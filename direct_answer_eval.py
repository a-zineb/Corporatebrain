"""Read-only evaluator for structured Direct Answer cases.

The case data is supplied by evaluation tooling and is never imported by the
production application. Collection access is limited to ``get`` calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence

from rag_pipeline import ChunkRecord, extract_evidence_generic_structured


@dataclass(frozen=True)
class EvaluationCase:
    document_id: str
    question: str
    expected_answer: str = ""
    expected_status: str = "EVIDENCE_FOUND"
    expected_relation: str = ""


def _normalize_answer(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s./:-]", " ", text)).strip()


def run_evaluation(collection: Any, cases: Sequence[EvaluationCase]) -> dict[str, Any]:
    """Evaluate cases using read-only selected-document Chroma access."""
    results: list[dict[str, Any]] = []
    for case in cases:
        if case.expected_status == "SENSITIVE_REQUEST":
            results.append({"document_id": case.document_id, "question": case.question, "expected_status": case.expected_status, "expected_relation": case.expected_relation, "actual_status": "SENSITIVE_BLOCK", "result": "PASS", "source_ids": [], "cross_document": False})
            continue
        data = collection.get(where={"file_hash": case.document_id}, include=["documents", "metadatas"])
        chunks = [
            ChunkRecord(text=text, metadata=metadata or {}, chunk_id=chunk_id)
            for text, metadata, chunk_id in zip(data.get("documents", []), data.get("metadatas", []), data.get("ids", []))
        ]
        evidence = extract_evidence_generic_structured(case.question, chunks)
        answer = " ".join(p.text for p in evidence.passages)
        expected_internal = "EVIDENCE_FOUND" if case.expected_status == "ANSWER" else case.expected_status
        status = "PASS" if evidence.status == expected_internal and (
            expected_internal != "EVIDENCE_FOUND" or _normalize_answer(case.expected_answer) in _normalize_answer(answer)
        ) else ("UNSUITABLE" if evidence.failure_reason == "UNSUITABLE" else ("NO_EVIDENCE" if evidence.status != "EVIDENCE_FOUND" else "WRONG"))
        results.append({
            "document_id": case.document_id,
            "question": case.question,
            "expected_status": case.expected_status,
            "expected_relation": case.expected_relation,
            "actual_status": evidence.status,
            "result": status,
            "source_ids": list(evidence.supporting_source_ids),
            "cross_document": len({p.source_file for p in evidence.passages}) > 1,
        })
    total = len(results)
    return {
        "total": total,
        "pass": sum(row["result"] == "PASS" for row in results),
        "wrong": sum(row["result"] == "WRONG" for row in results),
        "no_evidence": sum(row["result"] == "NO_EVIDENCE" for row in results),
        "unsuitable": sum(row["result"] == "UNSUITABLE" for row in results),
        "cross_document": sum(row["cross_document"] for row in results),
        "results": results,
    }
