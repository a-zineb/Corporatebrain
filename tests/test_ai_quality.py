from __future__ import annotations

import json

from backend.services.ai_quality import (
    classify_intent,
    generate_grounded,
    retrieve_evidence,
    rewrite_follow_up,
)
from document_normalizer import CanonicalBlock, CanonicalDocument, normalize_document


def csv_document() -> CanonicalDocument:
    return normalize_document(
        b"System,Host,Collection directory\nBI,10.0.0.1,/abc\nDWH,10.0.0.2,/dwh\n",
        "systems.csv",
    )


def test_intent_distinguishes_exhaustive_and_follow_up():
    assert classify_intent("Give me all systems") == "EXHAUSTIVE_LIST"
    assert classify_intent("Give me the EKYC test cases") == "EXHAUSTIVE_LIST"
    assert classify_intent("and its directory?") == "FOLLOW_UP"


def test_follow_up_uses_only_same_document_context():
    history = [{"role": "user", "content": "What is the BI host?", "document_hash": "a"}]
    assert "BI" in rewrite_follow_up("and its directory?", history, "a")
    assert rewrite_follow_up("and its directory?", history, "b") == "and its directory?"


def test_cross_language_alias_retrieval_finds_collection_directory():
    blocks = retrieve_evidence(csv_document(), "Quel est le répertoire de collecte de BI ?", "SINGLE_FACT")
    assert any("/abc" in block.text for block in blocks)


def test_workbook_sheet_query_returns_every_sheet_in_order():
    blocks = tuple(
        CanonicalBlock(str(index), f"Name = item {index}", "table_row", "book.xlsx", "hash",
                       sheet=sheet, row_index=2)
        for index, sheet in enumerate(("Sheet A", "Sheet B", "Sheet C", "Sheet D"))
    )
    document = CanonicalDocument("book", "hash", "book.xlsx", "xlsx", blocks)
    result = retrieve_evidence(document, "What sheets are in this workbook?", "TABLE_QUERY")
    assert [block.sheet for block in result] == ["Sheet A", "Sheet B", "Sheet C", "Sheet D"]


def test_workbook_completeness_validator_repairs_missing_sheet():
    blocks = tuple(CanonicalBlock(str(i), f"Value = {i}", "table_row", "book.xlsx", "hash",
                                  sheet=sheet, row_index=2)
                   for i, sheet in enumerate(("Sheet A", "Sheet B", "Sheet C", "Sheet D")))
    document = CanonicalDocument("book", "hash", "book.xlsx", "xlsx", blocks)
    calls = 0
    def generate(prompt: str) -> str:
        nonlocal calls
        calls += 1
        names = "Sheet A, Sheet B" if calls == 1 else "Sheet A, Sheet B, Sheet C, Sheet D"
        return json.dumps({"answer": names, "claims":[{"text":names,"evidence_ids":[b.block_id for b in blocks]}]})
    answer, _, _, _ = generate_grounded("What sheets are in this workbook?", document, [], generate)
    assert answer.repaired and calls == 2 and "Sheet D" in answer.answer


def test_invalid_citation_is_repaired_once_and_sources_remain_verbatim():
    document = csv_document()
    calls: list[str] = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        blocks = retrieve_evidence(document, "What is the BI host?", "SINGLE_FACT")
        evidence_id = blocks[0].block_id
        if len(calls) == 1:
            evidence_id = "invented"
        return json.dumps({
            "answer": "The BI host is `10.0.0.1`.",
            "claims": [{"text": "BI uses 10.0.0.1", "evidence_ids": [evidence_id]}],
        })

    answer, evidence, language, _ = generate_grounded(
        "What is the BI host?", document, [], generate,
    )
    assert answer.repaired and len(calls) == 2
    assert language == "English"
    assert any("10.0.0.1" in block.text for block in evidence)


def test_generated_answer_can_paraphrase_while_evidence_stays_exact():
    document = csv_document()

    def generate(prompt: str) -> str:
        block = retrieve_evidence(document, "Explain the BI endpoint", "EXPLANATION")[0]
        return json.dumps({
            "answer": "BI receives its files at `/abc` on `10.0.0.1`.",
            "claims": [{"text": "BI endpoint", "evidence_ids": [block.block_id]}],
        })

    answer, evidence, _, _ = generate_grounded("Explain the BI endpoint", document, [], generate)
    assert answer.answer != evidence[0].text
    assert evidence[0].text in {block.text for block in document.blocks}
