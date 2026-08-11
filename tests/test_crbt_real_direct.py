import random
from pathlib import Path

import pytest

from canonical_rag import ActiveDocumentService, FastDirectAnswerEngine, answer_direct
from document_normalizer import normalize_document


CRBT_CASES = {
    "How does CRBT send CDR files to MZ?": "PUSH",
    "How often are CRBT files handled for FTP-CRA?": "05 minutes",
    "How often are CRBT files handled for DWH, BI and Big Data?": "02 minutes",
    "Does MZ establish a collection connection to CRBT?": "No",
    "What is the CRBT CDR format?": "Brut",
    "What destinations are shown for the CRBT workflow?": "DWH, BI, SVR CRA",
    "What is the CRBT collection directory?": "TO BE DEFINED",
    "Who wrote the CRBT specification?": "Omar EL HIMASS",
    "Who reviewed the CRBT specification?": "Nawfal ENNAJI",
    "What is the BI output directory?": "/data/input/mz/sva/crbt",
}


def _real_crbt_document():
    matches = list(Path("doc_storage_v2").glob("*CRBT*.docx"))
    if not matches:
        pytest.skip("real CRBT validation document is unavailable")
    path = matches[0]
    return normalize_document(path.read_bytes(), path.name)


def test_real_crbt_exhaustive_results_have_constant_complete_coverage():
    document = _real_crbt_document()
    context = ActiveDocumentService().select(document)
    engine = FastDirectAnswerEngine()
    observed_counts = set()
    for question, expected in CRBT_CASES.items():
        result, trace = engine.query(context, question)
        assert result.status == "ANSWER"
        assert result.answer == expected
        assert result.file_hash == document.file_hash
        assert result.evidence_blocks
        assert all(block.file_hash == document.file_hash for block in result.evidence_blocks)
        assert trace.active_block_count == len(document.blocks)
        assert trace.candidate_count == len(document.blocks)
        assert trace.ollama_calls == trace.chroma_calls == 0
        observed_counts.add(trace.candidate_count)
    assert observed_counts == {len(document.blocks)}


def test_real_crbt_query_order_and_repetition_are_deterministic():
    document = _real_crbt_document()
    questions = list(CRBT_CASES)
    orders = [questions, list(reversed(questions))]
    shuffled = list(questions)
    random.Random(42).shuffle(shuffled)
    orders.append(shuffled)
    baseline = {question: answer_direct(question, document) for question in questions}
    for order in orders:
        assert {question: answer_direct(question, document) for question in order} == baseline
    for question in questions:
        assert all(answer_direct(question, document) == baseline[question] for _ in range(5))
