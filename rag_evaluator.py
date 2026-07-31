"""Deterministic, read-only benchmark runner for the certified RAG runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

import rag_forensics
import rag_judge
import rag_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = PROJECT_ROOT / "benchmarks" / "corporatebrain.v1.jsonl"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "evaluation_runs"


@dataclass(frozen=True, slots=True)
class EvaluationRuntime:
    """Read-only runtime resources supplied to the shared pipeline."""

    collection: rag_pipeline.VectorStore
    embedding_model: rag_pipeline.EmbeddingEncoder
    bm25: Any
    documents: list[str] | None
    metadatas: list[rag_pipeline.Metadata] | None
    config: rag_pipeline.RAGConfig


@dataclass(frozen=True, slots=True)
class RetrievalExperiment:
    """A fixed evaluator-only retrieval variant; production remains the control."""

    name: str
    vector_candidate_count: int
    bm25_candidate_count: int
    final_top_k: int
    fusion_depth: int
    fallback_threshold: int
    rrf_k: int = 60


def certified_control(config: rag_pipeline.RAGConfig) -> RetrievalExperiment:
    """Represent the approved production configuration without changing it."""

    return RetrievalExperiment("control", config.vector_candidate_count, config.bm25_candidate_count,
        config.production_top_k, config.vector_candidate_count, config.min_results_before_relax, config.rrf_k)


def fixed_experiment_matrix(config: rag_pipeline.RAGConfig) -> tuple[RetrievalExperiment, ...]:
    """Return reviewed fixed variants; this function never selects a winner."""

    control = certified_control(config)
    return (
        control,
        RetrievalExperiment("candidate_depth_20", 20, 20, control.final_top_k, 20, control.fallback_threshold),
        RetrievalExperiment("final_top_k_10", control.vector_candidate_count, control.bm25_candidate_count, 10, control.fusion_depth, control.fallback_threshold),
        RetrievalExperiment("fusion_depth_5", control.vector_candidate_count, control.bm25_candidate_count, control.final_top_k, 5, control.fallback_threshold),
        RetrievalExperiment("fallback_threshold_5", control.vector_candidate_count, control.bm25_candidate_count, control.final_top_k, control.fusion_depth, 5),
    )


class StageTimeoutError(TimeoutError):
    """Evaluator-only timeout that identifies the stage which exceeded its limit."""

    def __init__(self, stage: str, timeout_seconds: float) -> None:
        super().__init__(f"{stage} exceeded {timeout_seconds} seconds")
        self.stage = stage
        self.timeout_seconds = timeout_seconds


@dataclass(frozen=True, slots=True)
class EvaluatorGenerationBudget:
    """Evaluator-only streaming output cap; production generator calls are untouched."""

    generator: rag_pipeline.TextGenerator
    max_output_tokens: int

    def chat(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            options = dict(kwargs.get("options") or {})
            options["num_predict"] = self.max_output_tokens
            kwargs["options"] = options
        return self.generator.chat(**kwargs)


def run_stage(stage: str, operation: Callable[[], Any], timeout_seconds: float, *, clock: Callable[[], float] = time.perf_counter) -> tuple[Any, float]:
    """Run one read-only evaluator stage with a bounded wait and stage timing."""

    result: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
    started = clock()
    def worker() -> None:
        try:
            result.put((True, operation()))
        except BaseException as error:
            result.put((False, error))
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        succeeded, value = result.get(timeout=timeout_seconds)
    except queue.Empty as error:
        raise StageTimeoutError(stage, timeout_seconds) from error
    elapsed_ms = (clock() - started) * 1000
    if not succeeded:
        raise value
    return value, elapsed_ms


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read a versioned JSONL benchmark without changing it."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_runtime(config: rag_pipeline.RAGConfig, *, stage_timeout_seconds: float = 120.0) -> EvaluationRuntime:
    """Open the production collection read-only and build its certified BM25 view."""

    collection = chromadb.PersistentClient(path=config.chroma_path).get_collection(config.collection_name)
    embedding_model, _ = run_stage("model_loading", lambda: load_offline_embedding_model(config.embedding_model_name), stage_timeout_seconds)
    bm25, documents, metadatas = rag_pipeline.build_bm25_index(collection, collection.count())
    return EvaluationRuntime(collection, embedding_model, bm25, documents, metadatas, config)


def load_offline_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the production embedding model from local cache without network access."""

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as error:
        raise RuntimeError(
            f"Offline evaluation requires cached embedding model '{model_name}'. "
            "No model download was attempted."
        ) from error


def language_label(language: str) -> str:
    """Map the benchmark's documented language tag to the active prompt label."""

    return "English" if language == "en" else "French"


def source_hash(source: rag_pipeline.PromptSource) -> str:
    """Identify a displayed source by the exact content retained in the prompt."""

    return hashlib.sha256(source.text.encode("utf-8")).hexdigest()


def retrieval_metrics(
    sources: Sequence[rag_pipeline.PromptSource],
    relevance: Sequence[Mapping[str, Any]],
    top_k: int,
) -> dict[str, float]:
    """Calculate deterministic Recall@K, Precision@K, Hit Rate, MRR, and NDCG."""

    labels = {item["content_sha256"]: int(item["label"]) for item in relevance if item["label"] > 0}
    expected = set(labels)
    retrieved = [source_hash(source) for source in sources[:top_k]]
    gains = [labels.get(content_hash, 0) for content_hash in retrieved]
    matched = set(retrieved) & expected
    recall = len(matched) / len(expected) if expected else 1.0
    precision = sum(gain > 0 for gain in gains) / top_k if top_k else 0.0
    first_rank = next((index + 1 for index, gain in enumerate(gains) if gain > 0), None)
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(labels.values(), reverse=True)[:top_k]
    idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
    return {
        "recall_at_k": recall,
        "precision_at_k": precision,
        "hit_rate_at_k": 1.0 if first_rank else 0.0,
        "mrr": mrr,
        "ndcg_at_k": dcg / idcg if idcg else 1.0,
    }


def citation_metrics(
    citations: rag_pipeline.CitationResult,
    expected_citations: Sequence[Mapping[str, Any]],
) -> dict[str, bool | None]:
    """Check only citation validity and expected-source matching deterministically."""

    if not citations.cited_source_ids:
        return {"citation_valid": None, "expected_source_match": None}
    actual_hashes = {source_hash(source) for source in citations.display_sources}
    expected_hashes = {item["content_sha256"] for item in expected_citations}
    return {
        "citation_valid": not citations.invalid_source_ids,
        "expected_source_match": bool(actual_hashes) and actual_hashes <= expected_hashes,
    }


def evaluate_case(
    case: Mapping[str, Any],
    runtime: EvaluationRuntime,
    generator: rag_pipeline.TextGenerator,
    *,
    clock: Callable[[], float] = time.perf_counter,
    stage_timeout_seconds: float = 120.0,
    experiment: RetrievalExperiment | None = None,
) -> dict[str, Any]:
    """Execute one benchmark case exclusively through certified pipeline calls."""

    started = clock()
    experiment = experiment or certified_control(runtime.config)
    stage_timings: dict[str, float] = {}
    try:
        rewrite, stage_timings["query_rewriting"] = run_stage("query_rewriting", lambda: rag_pipeline.rewrite_query(case["question"], case["conversation"], runtime.config.llm_model_name, generator, clock=clock), stage_timeout_seconds, clock=clock)
        metadata_filter = rag_pipeline.normalize_chroma_filter(case["metadata_filter"])
        hybrid, stage_timings["vector_retrieval_bm25_rrf"] = run_stage("vector_retrieval_bm25_rrf", lambda: rag_pipeline.hybrid_search(rewrite.query, runtime.collection, runtime.embedding_model, runtime.bm25, runtime.documents, runtime.metadatas, chroma_filter=metadata_filter, top_k=experiment.final_top_k, min_results_before_relax=experiment.fallback_threshold, vector_candidate_count=experiment.vector_candidate_count, bm25_candidate_count=experiment.bm25_candidate_count, fusion_depth=experiment.fusion_depth, rrf_k=experiment.rrf_k), stage_timeout_seconds, clock=clock)
    except StageTimeoutError as error:
        return timeout_result(case, error, started, stage_timings, clock)
    filtered_sources = rag_pipeline.build_source_list(
        [chunk.text for chunk in hybrid.filtered.selected_chunks],
        [chunk.metadata for chunk in hybrid.filtered.selected_chunks],
        runtime.config.storage_dir,
    )
    fallback_sources = rag_pipeline.build_source_list(
        [chunk.text for chunk in hybrid.fallback_chunks],
        [chunk.metadata for chunk in hybrid.fallback_chunks],
        runtime.config.storage_dir,
        relaxed_flag=True,
        start_id=len(filtered_sources) + 1,
    )
    sources = filtered_sources + fallback_sources
    prompt = None
    generation = rag_pipeline.GenerationResult()
    citations = rag_pipeline.CitationResult()
    failure = None
    if sources:
        try:
            prompt, stage_timings["prompt_construction"] = run_stage("prompt_construction", lambda: rag_pipeline.build_production_prompt(
            user_query=case["question"],
            filter_ent="Tous",
            filter_application="Tous",
            history=case["conversation"],
            sources=sources,
            current_lang=language_label(case["language"]),
            was_relaxed=hybrid.fallback_used), stage_timeout_seconds, clock=clock)
        except StageTimeoutError as error:
            return timeout_result(case, error, started, stage_timings, clock)
        try:
            generation, stage_timings["generation"] = run_stage("generation", lambda: rag_pipeline.stream_generate(prompt.prompt, runtime.config.llm_model_name, generator, clock=clock), stage_timeout_seconds, clock=clock)
            citations = rag_pipeline.select_display_sources(generation.response, sources)
        except StageTimeoutError as error:
            return timeout_result(case, error, started, stage_timings, clock)
        except Exception as error:
            generation = rag_pipeline.GenerationResult(error=str(error))
            failure = {"code": "generation_error", "message": str(error)}

    metrics = retrieval_metrics(sources, case["relevance"], experiment.final_top_k)
    metrics.update(citation_metrics(citations, case["acceptable_citations"]))
    expected_mode = case["expected_behavior"]["mode"]
    metrics["refusal_correct"] = (
        citations.no_coverage_detected and not citations.display_sources
        if expected_mode == "refuse_no_coverage"
        else None
    )
    metrics["latency_ms"] = (clock() - started) * 1000
    trace = rag_pipeline.PipelineTrace(
        query=case["question"],
        rewritten_query=rewrite.query,
        language=language_label(case["language"]),
        metadata_filter=metadata_filter,
        retrieval=rag_pipeline.RetrievalResult(
            filtered_chunks=hybrid.filtered.selected_chunks,
            fallback_chunks=hybrid.fallback_chunks,
            fallback_used=hybrid.fallback_used,
            vector_candidates=hybrid.filtered.vector_candidates,
            bm25_candidates=hybrid.filtered.bm25_candidates,
            rrf_scores=hybrid.filtered.rrf_scores,
        ),
        prompt=prompt,
        generation=generation,
        citations=citations,
        timings=rag_pipeline.PipelineTimings(
            rewrite_ms=rewrite.latency_ms,
            generation_ms=generation.latency_ms,
            total_ms=metrics["latency_ms"],
        ),
        failure=rag_pipeline.PipelineFailure(**failure) if failure else None,
    )
    return {
        "case_id": case["id"],
        "experiment": experiment.name,
        "language": case["language"],
        "query_type": case["category"],
        "metadata_filter_state": "filtered" if metadata_filter else "unfiltered",
        "answerability": case["answerability"],
        "expected_behavior": case["expected_behavior"],
        "metrics": metrics,
        "response": generation.response,
        "trace": trace,
        "stage_timings_ms": stage_timings,
    }


def experiment_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report comparable metrics and forensic counts without promoting a winner."""

    serialized = [result_to_json(result) for result in results]
    counts: dict[str, int] = {}
    for result in serialized:
        category = str(result.get("forensic_category", "unknown"))
        counts[category] = counts.get(category, 0) + 1
    fallback_rate = (sum(bool(result["trace"].get("fallback_used")) for result in serialized) / len(serialized)) if serialized else 0.0
    return {"metrics": aggregate(serialized), "fallback_rate": fallback_rate, "forensic_counts": counts}


def run_experiment_matrix(
    cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime, generator: rag_pipeline.TextGenerator,
    experiments: Sequence[RetrievalExperiment], *, stage_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Evaluate fixed variants through the certified runtime; no winner is selected."""

    reports = {}
    for experiment in experiments:
        results = [evaluate_case(case, runtime, generator, stage_timeout_seconds=stage_timeout_seconds, experiment=experiment) for case in cases]
        reports[experiment.name] = {
            "configuration": {"vector_candidate_count": experiment.vector_candidate_count, "bm25_candidate_count": experiment.bm25_candidate_count, "final_top_k": experiment.final_top_k, "fusion_depth": experiment.fusion_depth, "fallback_threshold": experiment.fallback_threshold, "rrf_k": experiment.rrf_k},
            **experiment_summary(results),
        }
    return {"control": certified_control(runtime.config).name, "variants": reports}


def timeout_result(case: Mapping[str, Any], error: StageTimeoutError, started: float, stage_timings: Mapping[str, float], clock: Callable[[], float]) -> dict[str, Any]:
    """Record an incomplete case deterministically instead of hanging the evaluator."""
    trace = rag_pipeline.PipelineTrace(query=case["question"], failure=rag_pipeline.PipelineFailure(code=f"{error.stage}_timeout", message=str(error)))
    return {"case_id": case["id"], "language": case["language"], "query_type": case["category"], "metadata_filter_state": "filtered" if case["metadata_filter"] else "unfiltered", "answerability": case["answerability"], "expected_behavior": case["expected_behavior"], "metrics": {"recall_at_k": None, "precision_at_k": None, "hit_rate_at_k": None, "mrr": None, "ndcg_at_k": None, "citation_valid": None, "expected_source_match": None, "refusal_correct": None, "latency_ms": (clock() - started) * 1000}, "response": "", "trace": trace, "stage_timings_ms": dict(stage_timings)}


def trace_to_json(trace: rag_pipeline.PipelineTrace) -> dict[str, Any]:
    """Serialize the evaluator-owned trace without changing pipeline behavior."""

    retrieval = trace.retrieval
    return {
        "query": trace.query,
        "rewritten_query": trace.rewritten_query,
        "metadata_filter": trace.metadata_filter,
        "selected_chunks": [chunk.text for chunk in retrieval.filtered_chunks] if retrieval else [],
        "candidate_pool": [
            {"content_sha256": hashlib.sha256(candidate.chunk.text.encode("utf-8")).hexdigest(), "source": "vector", "rank": candidate.rank + 1}
            for candidate in retrieval.vector_candidates
        ] + [
            {"content_sha256": hashlib.sha256(candidate.chunk.text.encode("utf-8")).hexdigest(), "source": "bm25", "rank": candidate.rank + 1}
            for candidate in retrieval.bm25_candidates
        ] if retrieval else [],
        "fallback_chunks": [chunk.text for chunk in retrieval.fallback_chunks] if retrieval else [],
        "fallback_used": retrieval.fallback_used if retrieval else False,
        "response": trace.generation.response if trace.generation else "",
        "citations": list(trace.citations.cited_source_ids) if trace.citations else [],
        "displayed_sources": [source.file_name for source in trace.citations.display_sources] if trace.citations else [],
        "refusal_detected": trace.citations.no_coverage_detected if trace.citations else False,
        "timings_ms": {
            "rewrite": trace.timings.rewrite_ms,
            "generation": trace.timings.generation_ms,
            "total": trace.timings.total_ms,
        },
        "failure": {"code": trace.failure.code, "message": trace.failure.message} if trace.failure else None,
    }


def timeout_code(result: Mapping[str, Any]) -> str | None:
    """Return the evaluator timeout code when a persisted case is incomplete."""

    trace = result["trace"]
    failure = trace.get("failure") if isinstance(trace, Mapping) else trace.failure
    code = failure.get("code") if isinstance(failure, Mapping) else (failure.code if failure else None)
    return code if code and code.endswith("_timeout") else None


def completeness_report(results: Sequence[Mapping[str, Any]], expected_case_count: int) -> dict[str, Any]:
    """Keep execution completeness separate from eligible quality metrics."""

    timed_out = [result["case_id"] for result in results if timeout_code(result)]
    completed = {result["case_id"] for result in results if not timeout_code(result)}
    return {
        "expected_case_count": expected_case_count,
        "completed_case_count": len(completed),
        "timeout_case_count": len(timed_out),
        "timeout_case_ids": timed_out,
        "quality_evaluable_case_count": len(completed),
        "baseline_comparison_allowed": len(completed) == expected_case_count and not timed_out,
    }


def aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    """Average numeric deterministic metrics while preserving unavailable values."""

    names = ["recall_at_k", "precision_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k", "latency_ms"]
    summary: dict[str, float | None] = {}
    for name in names:
        values = [result["metrics"][name] for result in results if result["metrics"][name] is not None]
        summary[name] = sum(values) / len(values) if values else None
    for name in ("citation_valid", "expected_source_match", "refusal_correct"):
        values = [result["metrics"][name] for result in results if result["metrics"][name] is not None]
        summary[name] = sum(values) / len(values) if values else None
    def failure_code(result: Mapping[str, Any]) -> str | None:
        trace = result["trace"]
        failure = trace.get("failure") if isinstance(trace, Mapping) else trace.failure
        return failure.get("code") if isinstance(failure, Mapping) else (failure.code if failure else None)
    summary["generation_timeout_count"] = sum(failure_code(result) == "generation_timeout" for result in results)
    summary["successful_generation_count"] = sum(
        bool(result.get("response")) and failure_code(result) is None for result in results
    )
    return summary


def result_to_json(result: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a completed result, preserving an already checkpointed record."""

    if isinstance(result.get("trace"), Mapping):
        return dict(result)
    return {
        "case_id": result["case_id"], "experiment": result.get("experiment", "control"), "language": result["language"],
        "query_type": result["query_type"], "metadata_filter_state": result["metadata_filter_state"],
        "answerability": result["answerability"],
        "expected_behavior": result["expected_behavior"], "metrics": result["metrics"],
        "response": result["response"], "trace": trace_to_json(result["trace"]),
        "forensic_category": forensic_category(result),
    }


def forensic_category(result: Mapping[str, Any]) -> str:
    """Classify a result from the existing trace-only forensic pipeline."""

    trace = result["trace"]
    if isinstance(trace, Mapping):
        return str(result.get("forensic_category") or trace.get("failure", {}).get("code") or "unknown")
    return rag_forensics.classify_trace(trace, result["metrics"], result["expected_behavior"]).category


def segmented_diagnostics(cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate existing deterministic outputs by benchmark and trace dimensions."""

    dimensions = {
        "language": lambda case: case.get("language", "unknown"),
        "query_type": lambda case: case.get("query_type", "unknown"),
        "answerability": lambda case: case.get("answerability", "unknown"),
        "metadata_filter_state": lambda case: case.get("metadata_filter_state", "unknown"),
        "fallback_usage": lambda case: "used" if case.get("trace", {}).get("fallback_used") else "not_used",
        "forensic_category": lambda case: case.get("forensic_category", "unknown"),
    }
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, key_for in dimensions.items():
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for case in cases:
            groups.setdefault(str(key_for(case)), []).append(case)
        report[dimension] = {
            key: {"case_count": len(group), "metrics": aggregate(group)} for key, group in groups.items()
        }
    return report


def metadata_filter_audit(
    benchmark_cases: Sequence[Mapping[str, Any]],
    metadatas: Sequence[rag_pipeline.Metadata] | None,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit metadata and filter outcomes using existing collection and trace data only."""

    rows = list(metadatas or [])
    fields = sorted({str(key) for metadata in rows for key in metadata})
    values = {field: {json.dumps(metadata[field], ensure_ascii=False, sort_keys=True) for metadata in rows if field in metadata} for field in fields}
    missing = {field: sum(field not in metadata or metadata[field] in (None, "") for metadata in rows) for field in fields}
    types = {field: sorted({type(metadata[field]).__name__ for metadata in rows if field in metadata}) for field in fields}
    inconsistent = {
        field: {"missing_count": missing[field], "value_types": types[field]}
        for field in fields if missing[field] or len(types[field]) > 1
    }
    conditions = []
    for case in benchmark_cases:
        for key, value in (case.get("metadata_filter") or {}).items():
            status = "mapped"
            if key not in values:
                status = "unmapped_field"
            elif json.dumps(value, ensure_ascii=False, sort_keys=True) not in values[key]:
                status = "stale_value"
            conditions.append({"case_id": case["id"], "field": key, "value": value, "status": status})
    serialized = [result_to_json(result) for result in results]
    filtered = [result for result in serialized if result.get("metadata_filter_state") == "filtered"]
    fallback_used = sum(bool(result["trace"].get("fallback_used")) for result in filtered)
    return {
        "active_metadata": {
            "chunk_count": len(rows), "fields": {
                field: {"values": sorted(values[field]), "missing_count": missing[field], "value_types": types[field]}
                for field in fields
            },
        },
        "benchmark_filter_coverage": {
            "filtered_case_count": sum(bool(case.get("metadata_filter")) for case in benchmark_cases),
            "conditions": conditions,
            "unmapped_field_count": sum(item["status"] == "unmapped_field" for item in conditions),
            "stale_value_count": sum(item["status"] == "stale_value" for item in conditions),
        },
        "inconsistent_metadata": inconsistent,
        "filtered_query_outcomes": {"case_count": len(filtered), "metrics": aggregate(filtered)},
        "fallback_activation": {"filtered_case_count": len(filtered), "fallback_used_count": fallback_used},
    }


def write_metadata_filter_audit(
    benchmark_cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime,
    results: Sequence[Mapping[str, Any]], output_dir: Path,
) -> None:
    """Write the evaluator-owned metadata audit beside other ignored reports."""

    audit = metadata_filter_audit(benchmark_cases, runtime.metadatas, results)
    (output_dir / "metadata_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def reranking_opportunity_diagnostics(
    benchmark_cases: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]],
    metadatas: Sequence[rag_pipeline.Metadata] | None,
) -> dict[str, Any]:
    """Identify ranking-only misses from certified retrieval traces without reranking."""

    audit = metadata_filter_audit(benchmark_cases, metadatas, [])
    issue_cases = {item["case_id"] for item in audit["benchmark_filter_coverage"]["conditions"] if item["status"] != "mapped"}
    cases_by_id = {case["id"]: case for case in benchmark_cases}
    opportunities = []
    candidate_recalls = []
    selected_recalls = []
    answerable_count = 0
    for raw in results:
        result = result_to_json(raw)
        case = cases_by_id.get(result["case_id"])
        if not case or case["answerability"] != "answerable":
            continue
        answerable_count += 1
        expected = {item["content_sha256"] for item in case["relevance"] if item["label"] > 0}
        pool = result["trace"].get("candidate_pool", [])
        pool_hashes = {item["content_sha256"] for item in pool}
        selected_hashes = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in result["trace"].get("selected_chunks", [])}
        candidate_recall = len(expected & pool_hashes) / len(expected) if expected else 1.0
        selected_recall = len(expected & selected_hashes) / len(expected) if expected else 1.0
        candidate_recalls.append(candidate_recall)
        selected_recalls.append(selected_recall)
        if expected & pool_hashes and not expected & selected_hashes and result["case_id"] not in issue_cases:
            ranks = [item for item in pool if item["content_sha256"] in expected]
            opportunities.append({
                "case_id": result["case_id"], "candidate_ranks": ranks,
                "candidate_pool_recall": candidate_recall, "selected_context_recall": selected_recall,
                "exclusion_reason": "relevant_candidate_ranked_below_final_context_cutoff",
            })
    opportunity_rate = len(opportunities) / answerable_count if answerable_count else 0.0
    pool_recall = sum(candidate_recalls) / len(candidate_recalls) if candidate_recalls else 0.0
    selected_recall = sum(selected_recalls) / len(selected_recalls) if selected_recalls else 0.0
    gates = {
        "candidate_opportunity": {"met": len(opportunities) >= 3 and opportunity_rate >= 0.10, "threshold": "at least 3 cases and 10% of answerable cases"},
        "ranking_potential": {"met": pool_recall - selected_recall >= 0.10, "threshold": "candidate-pool recall exceeds selected-context recall by at least 0.10"},
        "filter_hygiene": {"met": True, "threshold": "affected cases have no stale or unmapped benchmark filter condition"},
        "matrix_exhaustion": {"met": "NOT_EVALUATED", "threshold": "requires full fixed-matrix evidence"},
        "baseline_protection": {"met": "NOT_EVALUATED", "threshold": "requires a reranking experiment"},
    }
    return {"answerable_case_count": answerable_count, "candidate_pool_recall": pool_recall, "selected_context_recall": selected_recall, "opportunity_count": len(opportunities), "opportunity_percentage": opportunity_rate, "affected_cases": opportunities, "gates": gates, "reranking_recommendation": "not_ready_pending_matrix_and_reranker_evidence"}


def write_reranking_opportunity_diagnostics(
    benchmark_cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime,
    results: Sequence[Mapping[str, Any]], output_dir: Path,
) -> None:
    """Write read-only reranking opportunity evidence beside evaluator reports."""

    payload = reranking_opportunity_diagnostics(benchmark_cases, results, runtime.metadatas)
    (output_dir / "reranking_opportunities.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(output_dir: Path) -> list[dict[str, Any]]:
    """Load completed case records from an evaluator-owned checkpoint."""

    path = output_dir / "cases.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_reports(results: Sequence[Mapping[str, Any]], output_dir: Path, *, expected_case_count: int | None = None) -> None:
    """Write reproducible JSON and Markdown reports below the ignored run directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [result_to_json(result) for result in results]
    summary = aggregate(results)
    (output_dir / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = "# Deterministic RAG evaluation\n\n" + "\n".join(
        f"- {name}: {value}" for name, value in summary.items()
    ) + "\n"
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    completeness = completeness_report(results, expected_case_count if expected_case_count is not None else len(results))
    (output_dir / "completeness.json").write_text(
        json.dumps(completeness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "segments.json").write_text(
        json.dumps(segmented_diagnostics(cases), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for result in results:
        if isinstance(result["trace"], Mapping):
            continue
        finding = rag_forensics.classify_trace(
            result["trace"], result["metrics"], result["expected_behavior"]
        )
        if finding.category != "passed":
            rag_forensics.write_failure_report(
                result["case_id"], finding, result["trace"], output_dir / "forensics"
            )


def run_cases(
    cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime, generator: rag_pipeline.TextGenerator,
    output_dir: Path, *, resume: bool = False, case_timeout_seconds: float | None = None,
    clock: Callable[[], float] = time.perf_counter, stage_timeout_seconds: float = 120.0,
) -> list[Mapping[str, Any]]:
    """Checkpoint each completed case and skip it on an explicit resume run."""

    if case_timeout_seconds is not None and case_timeout_seconds <= 0:
        raise ValueError("case_timeout_seconds must be greater than zero.")
    results: list[Mapping[str, Any]] = load_checkpoint(output_dir) if resume else []
    completed = {result["case_id"] for result in results if not timeout_code(result)}
    for case in cases:
        if case["id"] in completed:
            continue
        effective_timeout = min(stage_timeout_seconds, case_timeout_seconds) if case_timeout_seconds is not None else stage_timeout_seconds
        result = evaluate_case(case, runtime, generator, clock=clock, stage_timeout_seconds=effective_timeout)
        results = [existing for existing in results if existing["case_id"] != case["id"]]
        results.append(result)
        write_reports(results, output_dir, expected_case_count=len(cases))
    return results


def evaluate_judge_results(
    results: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    judge: rag_judge.JudgeAdapter,
) -> list[dict[str, Any]]:
    """Evaluate completed traces without changing their deterministic metrics."""

    cases_by_id = {case["id"]: case for case in cases}
    outcomes = []
    for result in results:
        case = cases_by_id[result["case_id"]]
        retrieval = result["trace"].retrieval
        contexts = [chunk.text for chunk in retrieval.filtered_chunks + retrieval.fallback_chunks] if retrieval else []
        outcome = judge.evaluate(
            question=result["trace"].query,
            answer=result["response"],
            contexts=contexts,
            reference_answer=case.get("expected_answer"),
        )
        outcomes.append({"case_id": result["case_id"], "outcome": outcome.to_json()})
    return outcomes


def write_judge_reports(outcomes: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Write separate optional-judge artifacts beside deterministic reports."""

    scored = [item["outcome"] for item in outcomes if item["outcome"]["status"] == "SCORED"]
    summary = {
        "scored_cases": len(scored),
        "not_run_cases": len(outcomes) - len(scored),
        "metrics": {
            name: (sum(item["metrics"][name] for item in scored) / len(scored) if scored else None)
            for name in rag_judge.JUDGE_METRICS
        },
    }
    (output_dir / "judge_cases.json").write_text(
        json.dumps(list(outcomes), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "judge_summary.md").write_text(
        "# Optional local judge\n\n" + "\n".join(f"- {key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the versioned benchmark against the active local production runtime."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge", action="store_true", help="Run optional local-only model judging.")
    parser.add_argument("--judge-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true", help="Resume from completed cases in --output-dir.")
    parser.add_argument("--case-timeout-seconds", type=float)
    parser.add_argument("--require-complete", action="store_true", help="Block baseline comparison when persisted cases are incomplete or timed out.")
    parser.add_argument("--stage-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--max-generation-tokens", type=int, default=256)
    parser.add_argument("--experiment-matrix", action="store_true", help="Run fixed evaluator-only retrieval variants.")
    args = parser.parse_args()
    config = rag_pipeline.RAGConfig()
    runtime = load_runtime(config, stage_timeout_seconds=args.stage_timeout_seconds)
    if args.max_generation_tokens <= 0:
        parser.error("--max-generation-tokens must be greater than zero")
    evaluator_generator = EvaluatorGenerationBudget(ollama, args.max_generation_tokens)
    output_dir = args.output_dir or DEFAULT_RUNS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    benchmark_cases = load_cases(args.benchmark)
    active_cases = benchmark_cases[:args.limit_cases] if args.limit_cases else benchmark_cases
    results = run_cases(
        active_cases, runtime, evaluator_generator, output_dir, resume=args.resume,
        case_timeout_seconds=args.case_timeout_seconds, stage_timeout_seconds=args.stage_timeout_seconds,
    )
    write_metadata_filter_audit(active_cases, runtime, results, output_dir)
    write_reranking_opportunity_diagnostics(active_cases, runtime, results, output_dir)
    if args.require_complete and not completeness_report(results, len(active_cases))["baseline_comparison_allowed"]:
        raise SystemExit("Baseline comparison blocked: inspect completeness.json and resume incomplete cases.")
    if args.experiment_matrix:
        matrix = run_experiment_matrix(active_cases, runtime, evaluator_generator, fixed_experiment_matrix(config), stage_timeout_seconds=args.stage_timeout_seconds)
        (output_dir / "retrieval_experiments.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.judge:
        judge = rag_judge.LocalOllamaJudgeAdapter(
            ollama, config.llm_model_name, args.judge_timeout_seconds
        )
        write_judge_reports(evaluate_judge_results(results, load_cases(args.benchmark), judge), output_dir)
    print(f"Deterministic evaluation report written to {output_dir}")


if __name__ == "__main__":
    main()
