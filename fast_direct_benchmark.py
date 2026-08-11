"""100-case latency and accuracy benchmark for zero-LLM Direct Answer."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from typing import Any

from canonical_eval import _documents
from canonical_rag import ActiveDocumentService, CatalogIndex, FastDirectAnswerEngine
from document_normalizer import normalize_document


QUESTIONS: tuple[tuple[str, str, str], ...] = (
    ("What is the BI host?", "10.0.0.1", "ANSWER"),
    ("Which host does BI use?", "10.0.0.1", "ANSWER"),
    ("What is the BI server IP?", "10.0.0.1", "ANSWER"),
    ("What is the BI protocol?", "SFTP", "ANSWER"),
    ("Which protocol is used by BI?", "SFTP", "ANSWER"),
    ("Quel est le protocole de BI ?", "SFTP", "ANSWER"),
    ("What is the BI directory?", "/data/bi", "ANSWER"),
    ("What is the BI output directory?", "/data/bi", "ANSWER"),
    ("Quel est le répertoire de BI ?", "/data/bi", "ANSWER"),
    ("What is the collection frequency?", "5 minutes", "ANSWER"),
    ("Which frequency is configured?", "5 minutes", "ANSWER"),
    ("Quelle est la fréquence de collecte ?", "5 minutes", "ANSWER"),
    ("What is the version?", "2.1", "ANSWER"),
    ("Which revision is documented?", "2.1", "ANSWER"),
    ("What is the processing status?", "Processed", "ANSWER"),
    ("Which status is recorded?", "Processed", "ANSWER"),
    ("What is the duplicate mechanism?", "Checksum", "ANSWER"),
    ("How are duplicates detected?", "Checksum", "ANSWER"),
    ("What is the archive directory?", "/archive/bi", "ANSWER"),
    ("Which archive path is used?", "/archive/bi", "ANSWER"),
    ("How does the system send files?", "PUSH", "ANSWER"),
    ("How are files sent?", "PUSH", "ANSWER"),
    ("Are files transformed?", "No", "ANSWER"),
    ("What is the password?", "NO_EXPLICIT_EVIDENCE", "SENSITIVE_BLOCK"),
    ("What database is used?", "NO_EXPLICIT_EVIDENCE", "NO_EVIDENCE"),
)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def run_fast_benchmark() -> dict[str, Any]:
    engine = FastDirectAnswerEngine()
    warm_latencies: list[float] = []
    repeated_latencies: list[float] = []
    cold_prepare: dict[str, float] = {}
    format_metrics: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "latencies_ms": []})
    wrong = false_no_evidence = cross_leakage = secret_leakage = ollama_calls = chroma_calls = 0
    for file_format, data in _documents().items():
        started = time.perf_counter()
        document = normalize_document(data, f"fast-benchmark.{file_format}")
        context = ActiveDocumentService().select(document)
        engine.prepare(document)
        cold_prepare[file_format] = (time.perf_counter() - started) * 1000
        for question, expected, expected_status in QUESTIONS:
            started = time.perf_counter()
            result, trace = engine.query(context, question)
            latency = (time.perf_counter() - started) * 1000
            warm_latencies.append(latency)
            row = format_metrics[file_format]
            row["total"] += 1
            row["latencies_ms"].append(latency)
            correct = result.answer == expected and result.status == expected_status
            row["correct"] += correct
            wrong += result.status == "ANSWER" and not correct
            false_no_evidence += expected_status == "ANSWER" and result.status != "ANSWER"
            cross_leakage += sum(block.file_hash != context.file_hash for block in result.evidence_blocks)
            secret_leakage += "real-secret" in result.answer
            ollama_calls += trace.ollama_calls
            chroma_calls += trace.chroma_calls
            started = time.perf_counter()
            repeated_result, repeated_trace = engine.query(context, question)
            repeated_latencies.append((time.perf_counter() - started) * 1000)
            assert repeated_result == result and not repeated_trace.cache_hit
    per_format = {
        key: {
            "total": value["total"], "correct": value["correct"],
            "accuracy": value["correct"] / value["total"],
            "p50_ms": statistics.median(value["latencies_ms"]),
            "p95_ms": _percentile(value["latencies_ms"], .95),
            "max_ms": max(value["latencies_ms"]),
        }
        for key, value in format_metrics.items()
    }
    return {
        "total": len(warm_latencies), "correct": sum(value["correct"] for value in format_metrics.values()),
        "wrong": wrong, "false_no_evidence": false_no_evidence,
        "cross_document_leakage": cross_leakage, "secret_leakage": secret_leakage,
        "ollama_calls": ollama_calls, "chroma_calls": chroma_calls,
        "cold_prepare_ms": cold_prepare,
        "warm_p50_ms": statistics.median(warm_latencies),
        "warm_p90_ms": _percentile(warm_latencies, .90),
        "warm_p95_ms": _percentile(warm_latencies, .95),
        "warm_max_ms": max(warm_latencies),
        "repeated_p50_ms": statistics.median(repeated_latencies),
        "repeated_p95_ms": _percentile(repeated_latencies, .95),
        "per_format": per_format,
    }


def run_catalog_benchmark() -> dict[str, float | int]:
    entries = [
        {"source_file": f"MZ_P2P_spec_{index}.docx", "file_hash": str(index),
         "application": "MZ" if index % 2 == 0 else "KPSA",
         "geographical_entity": "OCM" if index % 3 == 0 else "OCI", "version": "1.1"}
        for index in range(1000)
    ]
    started = time.perf_counter()
    catalog = CatalogIndex.from_metadatas(entries)
    build_ms = (time.perf_counter() - started) * 1000
    latencies = []
    for index in range(100):
        started = time.perf_counter()
        if index % 3 == 0:
            catalog.search("P2P", application="MZ")
        elif index % 3 == 1:
            catalog.search(file_type="docx", geographical_entity="OCM")
        else:
            catalog.search(version="1.1")
        latencies.append((time.perf_counter() - started) * 1000)
    return {
        "entries": len(catalog.entries), "queries": len(latencies), "build_ms": build_ms,
        "p50_ms": statistics.median(latencies), "p95_ms": _percentile(latencies, .95),
        "max_ms": max(latencies), "ollama_calls": 0, "embedding_calls": 0,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_fast_benchmark(), indent=2))
