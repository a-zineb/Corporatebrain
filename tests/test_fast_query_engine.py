import time

from canonical_rag import (
    ActiveDocumentService,
    CatalogIndex,
    DocumentQueryIndexCache,
    FastDirectAnswerEngine,
)
from document_normalizer import normalize_document


def _context(text, name="facts.csv"):
    document = normalize_document(text.encode(), name)
    return ActiveDocumentService().select(document)


def test_dynamic_field_index_and_nested_port_span_are_local():
    context = _context("System,Transfer Mechanism,Connection Protocol\nCRBT,PUSH,SFTP (port: 22)\n")
    engine = FastDirectAnswerEngine()
    index = engine.prepare(context.canonical_document)
    assert "transfer mechanism" in index.raw_field_index
    port, trace = engine.query(context, "What port is used for collection?")
    mechanism, second_trace = engine.query(context, "How does CRBT send files?")
    assert port.answer == "22"
    assert mechanism.answer == "PUSH"
    assert trace.ollama_calls == second_trace.ollama_calls == 0
    assert trace.chroma_calls == second_trace.chroma_calls == 0


def test_prose_negation_connection_and_direct_transfer_without_llm():
    context = _context(
        "Description,Topic\n"
        '"MZ n’établit pas de connexion avec la plateforme source. Les fichiers sont poussés vers MZ sans aucune transformation.",Collection\n'
    )
    engine = FastDirectAnswerEngine()
    connection, _ = engine.query(context, "Does MZ establish a collection connection?")
    transformed, _ = engine.query(context, "Are files transformed?")
    transfer, _ = engine.query(context, "How are files sent?")
    assert connection.answer == "No"
    assert transformed.answer == "No"
    assert transfer.answer == "PUSH"


def test_immutable_document_index_cache_is_hash_keyed_and_queries_are_recomputed():
    context = _context("System,Host\nBI,10.0.0.1\n")
    cache = DocumentQueryIndexCache()
    engine = FastDirectAnswerEngine(index_cache=cache)
    first, first_trace = engine.query(context, "What is the BI host?")
    started = time.perf_counter()
    second, second_trace = engine.query(context, "What is the BI host?")
    warm_ms = (time.perf_counter() - started) * 1000
    assert first.answer == second.answer == "10.0.0.1"
    assert len(cache) == 1 and not second_trace.cache_hit
    assert warm_ms < 50


def test_catalog_index_filters_without_rag_or_generation():
    catalog = CatalogIndex.from_metadatas([
        {"source_file": "P2P_spec_v1.1.docx", "file_hash": "a", "application": "MZ", "geographical_entity": "OCM", "version": "1.1"},
        {"source_file": "other.pdf", "file_hash": "b", "application": "KPSA", "geographical_entity": "OCI"},
        {"source_file": "P2P_spec_v1.1.docx", "file_hash": "a", "application": "MZ"},
    ])
    assert len(catalog.entries) == 2
    assert [entry.source_file for entry in catalog.search("P2P")] == ["P2P_spec_v1.1.docx"]
    assert [entry.source_file for entry in catalog.search(file_type="pdf")] == ["other.pdf"]
    assert [entry.source_file for entry in catalog.search(application="MZ")] == ["P2P_spec_v1.1.docx"]
    assert [entry.source_file for entry in catalog.search("Which files are version 1.1?")] == ["P2P_spec_v1.1.docx"]
    assert [entry.source_file for entry in catalog.search("List all DOCX documents")] == ["P2P_spec_v1.1.docx"]


def test_version_history_qualifier_selects_the_matching_author_row():
    context = _context("Version,Author\nV1.1,Alice Martin\nV1.2,Nawfal ENNAJI\n", "history.csv")
    result, trace = FastDirectAnswerEngine().query(context, "Who authored V1.2?")
    assert result.answer == "Nawfal ENNAJI"
    assert trace.ollama_calls == 0


def test_debug_trace_scans_all_blocks_and_never_contains_raw_secret():
    context = _context("System,Host,Password\nBI,10.0.0.1,actual-secret\n")
    result, trace = FastDirectAnswerEngine().query(context, "What is the BI host?")
    assert result.answer == "10.0.0.1"
    assert trace.active_block_count == trace.candidate_count == len(context.block_ids)
    assert "actual-secret" not in str(trace.top_candidates)
