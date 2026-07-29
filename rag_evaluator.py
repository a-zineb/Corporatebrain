"""Deterministic, read-only benchmark runner for the certified RAG runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
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


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read a versioned JSONL benchmark without changing it."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_runtime(config: rag_pipeline.RAGConfig) -> EvaluationRuntime:
    """Open the production collection read-only and build its certified BM25 view."""

    collection = chromadb.PersistentClient(path=config.chroma_path).get_collection(config.collection_name)
    embedding_model = load_offline_embedding_model(config.embedding_model_name)
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
) -> dict[str, Any]:
    """Execute one benchmark case exclusively through certified pipeline calls."""

    started = clock()
    rewrite = rag_pipeline.rewrite_query(
        case["question"], case["conversation"], runtime.config.llm_model_name, generator, clock=clock
    )
    metadata_filter = rag_pipeline.normalize_chroma_filter(case["metadata_filter"])
    hybrid = rag_pipeline.hybrid_search(
        rewrite.query,
        runtime.collection,
        runtime.embedding_model,
        runtime.bm25,
        runtime.documents,
        runtime.metadatas,
        chroma_filter=metadata_filter,
        top_k=runtime.config.production_top_k,
        min_results_before_relax=runtime.config.min_results_before_relax,
    )
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
        prompt = rag_pipeline.build_production_prompt(
            user_query=case["question"],
            filter_ent="Tous",
            filter_application="Tous",
            history=case["conversation"],
            sources=sources,
            current_lang=language_label(case["language"]),
            was_relaxed=hybrid.fallback_used,
        )
        try:
            generation = rag_pipeline.stream_generate(
                prompt.prompt, runtime.config.llm_model_name, generator, clock=clock
            )
            citations = rag_pipeline.select_display_sources(generation.response, sources)
        except Exception as error:
            generation = rag_pipeline.GenerationResult(error=str(error))
            failure = {"code": "generation_error", "message": str(error)}

    metrics = retrieval_metrics(sources, case["relevance"], runtime.config.production_top_k)
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
        "answerability": case["answerability"],
        "expected_behavior": case["expected_behavior"],
        "metrics": metrics,
        "response": generation.response,
        "trace": trace,
    }


def trace_to_json(trace: rag_pipeline.PipelineTrace) -> dict[str, Any]:
    """Serialize the evaluator-owned trace without changing pipeline behavior."""

    retrieval = trace.retrieval
    return {
        "query": trace.query,
        "rewritten_query": trace.rewritten_query,
        "metadata_filter": trace.metadata_filter,
        "selected_chunks": [chunk.text for chunk in retrieval.filtered_chunks] if retrieval else [],
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


def aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Average numeric deterministic metrics while preserving unavailable values."""

    names = ["recall_at_k", "precision_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k", "latency_ms"]
    summary: dict[str, float | None] = {}
    for name in names:
        values = [result["metrics"][name] for result in results if result["metrics"][name] is not None]
        summary[name] = sum(values) / len(values) if values else None
    for name in ("citation_valid", "expected_source_match", "refusal_correct"):
        values = [result["metrics"][name] for result in results if result["metrics"][name] is not None]
        summary[name] = sum(values) / len(values) if values else None
    return summary


def result_to_json(result: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a completed result, preserving an already checkpointed record."""

    if isinstance(result.get("trace"), Mapping):
        return dict(result)
    return {
        "case_id": result["case_id"], "answerability": result["answerability"],
        "expected_behavior": result["expected_behavior"], "metrics": result["metrics"],
        "response": result["response"], "trace": trace_to_json(result["trace"]),
    }


def load_checkpoint(output_dir: Path) -> list[dict[str, Any]]:
    """Load completed case records from an evaluator-owned checkpoint."""

    path = output_dir / "cases.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_reports(results: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
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
    clock: Callable[[], float] = time.perf_counter,
) -> list[Mapping[str, Any]]:
    """Checkpoint each completed case and skip it on an explicit resume run."""

    if case_timeout_seconds is not None and case_timeout_seconds <= 0:
        raise ValueError("case_timeout_seconds must be greater than zero.")
    results: list[Mapping[str, Any]] = load_checkpoint(output_dir) if resume else []
    completed = {result["case_id"] for result in results}
    for case in cases:
        if case["id"] in completed:
            continue
        started = clock()
        result = evaluate_case(case, runtime, generator, clock=clock)
        elapsed = clock() - started
        if case_timeout_seconds is not None and elapsed > case_timeout_seconds:
            raise TimeoutError(f"Case {case['id']} exceeded {case_timeout_seconds} seconds after completion.")
        results.append(result)
        write_reports(results, output_dir)
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
    args = parser.parse_args()
    config = rag_pipeline.RAGConfig()
    runtime = load_runtime(config)
    output_dir = args.output_dir or DEFAULT_RUNS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = run_cases(
        load_cases(args.benchmark), runtime, ollama, output_dir, resume=args.resume,
        case_timeout_seconds=args.case_timeout_seconds,
    )
    if args.judge:
        judge = rag_judge.LocalOllamaJudgeAdapter(
            ollama, config.llm_model_name, args.judge_timeout_seconds
        )
        write_judge_reports(evaluate_judge_results(results, load_cases(args.benchmark), judge), output_dir)
    print(f"Deterministic evaluation report written to {output_dir}")


if __name__ == "__main__":
    main()
