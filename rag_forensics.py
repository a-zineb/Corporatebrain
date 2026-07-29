"""Trace-only forensic classification and report rendering for RAG evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import rag_pipeline


@dataclass(frozen=True, slots=True)
class ForensicFinding:
    """One deterministic failure classification derived from an execution trace."""

    category: str
    severity: str
    reason: str
    evidence: Mapping[str, Any]


def classify_trace(
    trace: rag_pipeline.PipelineTrace,
    metrics: Mapping[str, Any],
    expected_behavior: Mapping[str, Any],
) -> ForensicFinding:
    """Classify an evaluated trace without executing or recreating pipeline work."""

    if trace.failure:
        return ForensicFinding(
            "generation_error", "error", trace.failure.message, {"code": trace.failure.code}
        )
    retrieval = trace.retrieval
    if not retrieval or not (retrieval.filtered_chunks or retrieval.fallback_chunks):
        return ForensicFinding("no_retrieval_results", "error", "No prompt sources were selected.", {})
    if expected_behavior.get("mode") == "answer" and metrics.get("recall_at_k", 0.0) < 1.0:
        return ForensicFinding(
            "expected_source_missing", "error", "Not every annotated relevant source was selected.",
            {"recall_at_k": metrics.get("recall_at_k")},
        )
    if metrics.get("citation_valid") is False:
        return ForensicFinding(
            "invalid_citation", "error", "The response cited a source ID outside the prompt source list.",
            {"invalid_source_ids": list(trace.citations.invalid_source_ids) if trace.citations else []},
        )
    if metrics.get("expected_source_match") is False:
        return ForensicFinding(
            "unexpected_citation_source", "warning", "Displayed citations do not match benchmark-approved sources.", {}
        )
    if expected_behavior.get("mode") == "refuse_no_coverage" and metrics.get("refusal_correct") is False:
        return ForensicFinding(
            "refusal_incorrect", "error", "An unanswerable case was not identified as no documentary coverage.",
            {"refusal_detected": trace.citations.no_coverage_detected if trace.citations else False},
        )
    return ForensicFinding("passed", "info", "No deterministic failure was detected.", {})


def trace_snapshot(trace: rag_pipeline.PipelineTrace) -> dict[str, Any]:
    """Return a JSON-safe, read-only trace excerpt for a forensic artifact."""

    retrieval = trace.retrieval
    return {
        "query": trace.query,
        "rewritten_query": trace.rewritten_query,
        "metadata_filter": trace.metadata_filter,
        "selected_source_files": [
            chunk.metadata.get("source_file") for chunk in retrieval.filtered_chunks
        ] if retrieval else [],
        "fallback_source_files": [
            chunk.metadata.get("source_file") for chunk in retrieval.fallback_chunks
        ] if retrieval else [],
        "fallback_used": retrieval.fallback_used if retrieval else False,
        "response": trace.generation.response if trace.generation else "",
        "citation_ids": list(trace.citations.cited_source_ids) if trace.citations else [],
        "displayed_source_files": [
            source.file_name for source in trace.citations.display_sources
        ] if trace.citations else [],
        "refusal_detected": trace.citations.no_coverage_detected if trace.citations else False,
        "timings_ms": asdict(trace.timings),
    }


def render_markdown(case_id: str, finding: ForensicFinding, trace: rag_pipeline.PipelineTrace) -> str:
    """Render a concise, human-readable report for one evaluated failure."""

    snapshot = trace_snapshot(trace)
    return "\n".join([
        f"# Forensic report: {case_id}",
        "",
        f"- Category: `{finding.category}`",
        f"- Severity: `{finding.severity}`",
        f"- Reason: {finding.reason}",
        f"- Query: {snapshot['query']}",
        f"- Rewritten query: {snapshot['rewritten_query']}",
        f"- Selected sources: {snapshot['selected_source_files']}",
        f"- Fallback used: {snapshot['fallback_used']}",
        f"- Displayed sources: {snapshot['displayed_source_files']}",
        f"- Refusal detected: {snapshot['refusal_detected']}",
        "",
    ])


def write_failure_report(
    case_id: str,
    finding: ForensicFinding,
    trace: rag_pipeline.PipelineTrace,
    output_dir: Path,
) -> None:
    """Write structured JSON and Markdown artifacts for one non-passing trace."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "finding": asdict(finding),
        "trace": trace_snapshot(trace),
    }
    (output_dir / f"{case_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{case_id}.md").write_text(
        render_markdown(case_id, finding, trace), encoding="utf-8"
    )
