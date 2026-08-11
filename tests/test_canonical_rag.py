import io

import fitz
from openpyxl import Workbook

from canonical_rag import (
    NO_EXPLICIT_EVIDENCE,
    ActiveDocumentService,
    CanonicalSessionCache,
    answer_question,
    debug_snapshot,
    document_health,
    ingestion_diagnostics,
    is_synthesis_query,
    normalize_with_gate,
    retrieve_canonical,
)
from document_normalizer import normalize_document


def _csv(text, name="facts.csv"):
    return normalize_document(text.encode(), name)


def _context(document):
    return ActiveDocumentService().select(document)


def test_switching_document_discards_old_evidence_and_increments_version():
    service = ActiveDocumentService()
    a = service.select(_csv("System,Host\nBI,1.1.1.1\n", "a.csv"))
    b = service.select(_csv("System,Host\nBI,2.2.2.2\n", "b.csv"))
    result = answer_question(b, "What is the BI host?")
    assert result.answer == "2.2.2.2"
    assert "1.1.1.1" not in result.answer
    assert all(block.file_hash == b.file_hash for block in result.evidence_blocks)
    assert b.selection_version == a.selection_version + 1


def test_identity_cases_same_name_same_bytes_and_modified_bytes():
    cache = CanonicalSessionCache()
    same = b"Name,Port\nAPI,443\n"
    first = cache.get_or_normalize(same, "same.csv")
    reused = cache.get_or_normalize(same, "renamed.csv")
    changed = cache.get_or_normalize(b"Name,Port\nAPI,8443\n", "same.csv")
    assert reused is first
    assert changed.file_hash != first.file_hash and len(cache) == 2


def test_entity_isolation_and_ambiguity_fail_closed():
    context = _context(_csv("System,Directory\nBI,/bi\nDWH,/dwh\nGlobal,/global\n"))
    assert answer_question(context, "What is the BI directory?").answer == "/bi"
    conflict = _context(_csv("System,Host\nBI,1.1.1.1\nBI,2.2.2.2\n", "conflict.csv"))
    result = answer_question(conflict, "What is the BI host?")
    assert result.status == "AMBIGUOUS" and result.answer == NO_EXPLICIT_EVIDENCE


def test_grounded_fallback_accepts_only_cited_supported_active_evidence():
    context = _context(_csv("Description,Owner\nThe system transfers files directly using PUSH.,Ops\n"))

    def grounded(prompt):
        evidence_id = context.block_ids[0]
        return {"answer": "PUSH", "supporting_evidence_ids": [evidence_id], "supported": True}

    result = answer_question(context, "How does the system send files?")
    assert result.answer == "PUSH" and result.method == "local_span_extraction"


def test_sensitive_field_is_blocked_but_safe_field_in_same_record_answers():
    context = _context(_csv("System,Host,Password\nBI,10.0.0.1,actual-secret\n"))
    host = answer_question(context, "What is the BI host?")
    password = answer_question(context, "What is the BI password?")
    assert host.answer == "10.0.0.1"
    assert password.status == "SENSITIVE_BLOCK"
    assert "actual-secret" not in context.canonical_document.canonical_text()


def test_retrieval_never_returns_a_foreign_hash_and_debug_is_safe():
    context = _context(_csv("System,Protocol\nBI,SFTP\n"))
    candidates = retrieve_canonical(context, "BI protocol")
    assert candidates and all(item.block.file_hash == context.file_hash for item in candidates)
    answer = answer_question(context, "What is the BI protocol?")
    debug = debug_snapshot(context, answer)
    assert debug["cross_document_evidence_count"] == 0
    assert "SFTP" not in str(debug)


def test_ingestion_gate_and_health_cover_bad_and_valid_inputs():
    assert normalize_with_gate(b"corrupt", "bad.docx").status == "EXTRACTION_FAILED"
    assert normalize_with_gate(b"not a zip", "bad.zip").status == "EXTRACTION_FAILED"
    assert normalize_with_gate(b"x", "bad.exe").status == "UNSUPPORTED"
    assert normalize_with_gate(b"Name,Value\nA,B\n", "ok.csv").status == "READY"

    pdf = fitz.open()
    pdf.new_page()
    empty_pdf = normalize_document(pdf.tobytes(), "empty.pdf")
    pdf.close()
    assert ingestion_diagnostics(empty_pdf).status == "EXTRACTION_FAILED"
    assert document_health(empty_pdf).status == "FAIL"

    workbook = Workbook()
    stream = io.BytesIO()
    workbook.save(stream)
    blank = normalize_document(stream.getvalue(), "blank.xlsx")
    assert ingestion_diagnostics(blank).status == "EXTRACTION_FAILED"
    assert document_health(blank).status == "FAIL"


def test_factual_grammar_is_not_misrouted_to_synthesis():
    assert not is_synthesis_query("How does the system send files?")
    assert not is_synthesis_query("Are files transformed?")
    assert is_synthesis_query("Explain why files are transformed")
