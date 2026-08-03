"""Deterministic, read-only benchmark runner for the certified RAG runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import multiprocessing
import pickle
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import unicodedata

import chromadb
import ollama

import rag_forensics
import rag_judge
import rag_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = PROJECT_ROOT / "benchmarks" / "corporatebrain.v1.jsonl"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "evaluation_runs"
CERTIFIED_BASELINE = PROJECT_ROOT / "baselines" / "corporatebrain.v1.baseline.json"


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


@dataclass(frozen=True, slots=True)
class ContextGroundingExperiment:
    """Evaluator-only presentation of one unchanged retrieved context."""

    name: str
    strategy: str
    max_sources: int | None = None
    prompt_suffix: str = ""


G4_EXPLICIT_FACTS_SUFFIX = """[EVALUATOR-ONLY GROUNDING RULES]
Answer only with facts explicitly stated in the supplied SOURCE sections.
Do not use outside knowledge, assumptions, or facts from a source that is not supplied.
If the supplied sources do not explicitly answer the question, state that the information is not available in the provided documents.
Do not infer, generalize, or substitute a related fact for the requested fact.
[/EVALUATOR-ONLY GROUNDING RULES]"""


G5_CITATION_REQUIRED_SUFFIX = G4_EXPLICIT_FACTS_SUFFIX + """

[EVALUATOR-ONLY CITATION RULES]
Every sentence containing a factual claim must end with one or more supporting citations in the exact form [SOURCE n].
Use only source numbers that appear in the supplied context.
A clarification or no-coverage response must not invent a citation.
[/EVALUATOR-ONLY CITATION RULES]"""


GROUNDING_EXPERIMENTS: tuple[ContextGroundingExperiment, ...] = (
    ContextGroundingExperiment("G0_control", "control"),
    ContextGroundingExperiment("G1_evidence_first", "evidence_first"),
    ContextGroundingExperiment("G2_deduplicated_context", "deduplicated"),
    ContextGroundingExperiment("G3_focused_context", "focused", max_sources=5),
    ContextGroundingExperiment("G4_explicit_facts", "prompt_suffix", prompt_suffix=G4_EXPLICIT_FACTS_SUFFIX),
    ContextGroundingExperiment("G5_citation_required", "prompt_suffix", prompt_suffix=G5_CITATION_REQUIRED_SUFFIX),
)

# The approved prompt-grounding benchmark compares only G0/G4/G5.  The prior
# G1-G3 context experiments remain available to their existing callers.
PROMPT_GROUNDING_EXPERIMENTS: tuple[ContextGroundingExperiment, ...] = tuple(
    experiment for experiment in GROUNDING_EXPERIMENTS
    if experiment.name in {"G0_control", "G4_explicit_facts", "G5_citation_required"}
)

# Compact-context experiments retain the first N already-ranked sources.  They
# are evaluator-only views over one certified trace; they never rerun retrieval.
COMPACT_CONTEXT_EXPERIMENTS: tuple[ContextGroundingExperiment, ...] = (
    ContextGroundingExperiment("C15_control", "top_n", max_sources=15),
    ContextGroundingExperiment("C5_compact", "top_n", max_sources=5),
    ContextGroundingExperiment("C3_compact", "top_n", max_sources=3),
)


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


@dataclass(frozen=True, slots=True)
class LocalModelStageRequest:
    """Serializable evaluator-only request for one local Ollama model stage."""

    stage: str
    model_name: str
    question: str = ""
    conversation: tuple[rag_pipeline.ChatMessage | rag_pipeline.Metadata, ...] = ()
    prompt: str = ""
    clarification_language: str | None = None
    max_output_tokens: int | None = None
    timeout_seconds: float = 120.0


def _child_process_entry(operation: Callable[[Any], Any], payload: Any, result: Any) -> None:
    """Return a serializable child-process outcome without leaking exceptions."""

    try:
        result.put((True, operation(payload)))
    except BaseException as error:
        result.put((False, (type(error).__name__, str(error))))


def run_isolated_stage(
    stage: str,
    operation: Callable[[Any], Any],
    payload: Any,
    timeout_seconds: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, float]:
    """Run one evaluator-only model operation in a child that can be terminated.

    A timed-out child is terminated and joined before this function returns.  This
    deliberately prevents queued local-Ollama work from surviving into the next
    benchmark case.
    """

    context = multiprocessing.get_context("spawn")
    result = context.Queue(maxsize=1)
    process = context.Process(target=_child_process_entry, args=(operation, payload, result))
    started = clock()
    process.start()
    try:
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            raise StageTimeoutError(stage, timeout_seconds)
        try:
            succeeded, value = result.get(timeout=1.0)
        except queue.Empty as error:
            raise RuntimeError(f"{stage} child exited without returning an outcome") from error
        if not succeeded:
            error_type, message = value
            raise RuntimeError(f"{stage} child failed with {error_type}: {message}")
        return value, (clock() - started) * 1000
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        result.close()
        result.join_thread()


def _execute_local_model_stage(request: LocalModelStageRequest) -> rag_pipeline.QueryRewriteResult | rag_pipeline.GenerationResult:
    """Call certified pipeline model APIs through an evaluator-local Ollama client."""

    client = ollama.Client(host="http://127.0.0.1:11434", timeout=request.timeout_seconds)
    if request.stage == "query_rewriting":
        return rag_pipeline.rewrite_query(
            request.question, request.conversation, request.model_name, client,
        )
    if request.stage == "generation":
        generator: rag_pipeline.TextGenerator = client
        if request.max_output_tokens is not None:
            generator = EvaluatorGenerationBudget(client, request.max_output_tokens)
        return rag_pipeline.stream_generate(
            request.prompt, request.model_name, generator,
            clarification_language=request.clarification_language,
        )
    raise ValueError(f"Unsupported local model stage: {request.stage}")


def uses_local_ollama(generator: rag_pipeline.TextGenerator) -> bool:
    """Identify the evaluator's real local-Ollama generator without affecting fakes."""

    underlying = generator.generator if isinstance(generator, EvaluatorGenerationBudget) else generator
    return underlying is ollama or isinstance(underlying, ollama.Client)


def run_model_stage(
    request: LocalModelStageRequest,
    generator: rag_pipeline.TextGenerator,
    fallback_operation: Callable[[], Any],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, float]:
    """Use a cancellable process for real local Ollama and in-process fakes for tests."""

    if uses_local_ollama(generator):
        return run_isolated_stage(request.stage, _execute_local_model_stage, request, request.timeout_seconds, clock=clock)
    return run_stage(request.stage, fallback_operation, request.timeout_seconds, clock=clock)


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


def load_offline_embedding_model(model_name: str):
    """Compatibility wrapper around the shared certified offline loader."""
    return rag_pipeline.load_embedding_model_offline(
        rag_pipeline.RAGConfig(embedding_model_name=model_name)
    )


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
    citations: rag_pipeline.CitationResult | None,
    expected_citations: Sequence[Mapping[str, Any]],
) -> dict[str, bool | None | str | int]:
    """Check only citation validity and expected-source matching deterministically."""

    if citations is None:
        return {
            "citation_valid": None, "expected_source_match": None,
            "citation_status": "NOT_EVALUABLE", "citation_evaluable_count": 0,
        }
    if not citations.cited_source_ids:
        return {
            "citation_valid": None, "expected_source_match": None,
            "citation_status": "NO_CITATIONS", "citation_evaluable_count": 0,
        }
    actual_hashes = {source_hash(source) for source in citations.display_sources}
    expected_hashes = {item["content_sha256"] for item in expected_citations}
    return {
        "citation_valid": not citations.invalid_source_ids,
        "expected_source_match": bool(actual_hashes) and actual_hashes <= expected_hashes,
        "citation_status": "EVALUATED",
        "citation_evaluable_count": 1,
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
        rewrite_request = LocalModelStageRequest(
            stage="query_rewriting", model_name=runtime.config.llm_model_name,
            question=case["question"], conversation=tuple(case["conversation"]),
            timeout_seconds=stage_timeout_seconds,
        )
        rewrite, stage_timings["query_rewriting"] = run_model_stage(
            rewrite_request, generator,
            lambda: rag_pipeline.rewrite_query(
                case["question"], case["conversation"], runtime.config.llm_model_name, generator, clock=clock,
            ),
            clock=clock,
        )
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
            generation_request = LocalModelStageRequest(
                stage="generation", model_name=runtime.config.llm_model_name, prompt=prompt.prompt,
                clarification_language=language_label(case["language"]),
                max_output_tokens=(
                    generator.max_output_tokens if isinstance(generator, EvaluatorGenerationBudget) else None
                ),
                timeout_seconds=stage_timeout_seconds,
            )
            generation, stage_timings["generation"] = run_model_stage(
                generation_request, generator,
                lambda: rag_pipeline.stream_generate(
                    prompt.prompt, runtime.config.llm_model_name, generator, clock=clock,
                    clarification_language=language_label(case["language"]),
                ),
                clock=clock,
            )
            citations = rag_pipeline.select_display_sources(generation.response, sources)
        except StageTimeoutError as error:
            return timeout_result(case, error, started, stage_timings, clock)
        except Exception as error:
            generation = rag_pipeline.GenerationResult(error=str(error))
            failure = {"code": "generation_error", "message": str(error)}

    metrics = retrieval_metrics(sources, case["relevance"], experiment.final_top_k)
    metrics.update(citation_metrics(citations, case["acceptable_citations"]))
    expected_mode = case["expected_behavior"]["mode"]
    is_refusal = citations.no_coverage_detected and not citations.display_sources
    is_clarification = generation.response == rag_pipeline.build_clarification_message(
        language_label(case["language"])
    )
    allows_no_source_outcome = case["expected_behavior"].get("source_display") == "none"
    metrics["refusal_correct"] = (
        is_refusal or (allows_no_source_outcome and is_clarification)
        if expected_mode == "refuse_no_coverage"
        else None
    )
    metrics["clarification_correct"] = (
        is_clarification
        if expected_mode == "request_clarification" or allows_no_source_outcome
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


def source_record(source: rag_pipeline.PromptSource) -> dict[str, Any]:
    """Serialize one evaluator context source without changing its content."""

    return {
        "source_id": source.source_id,
        "content_sha256": source_hash(source),
        "file_name": source.file_name,
        "location": source.location,
    }


def grounding_context(
    sources: Sequence[rag_pipeline.PromptSource], relevance: Sequence[Mapping[str, Any]],
    experiment: ContextGroundingExperiment,
) -> tuple[tuple[rag_pipeline.PromptSource, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return an evaluator-only ordering/subset of already selected sources.

    G1 and G3 use benchmark annotations only to measure an upper-bound context
    presentation.  They never alter retrieval candidates, RRF scores, or the
    production source list.
    """

    original = tuple(sources)
    expected = {item["content_sha256"] for item in relevance if item.get("label", 0) > 0}
    if experiment.strategy in ("control", "prompt_suffix"):
        retained = original
        reasons: dict[int, str] = {}
    elif experiment.strategy == "evidence_first":
        retained = tuple(sorted(
            original,
            key=lambda source: (source_hash(source) not in expected, source.source_id),
        ))
        reasons = {}
    elif experiment.strategy == "deduplicated":
        seen: set[str] = set()
        retained_items = []
        reasons = {}
        for source in original:
            content_id = source_hash(source)
            if content_id in seen:
                reasons[source.source_id] = "duplicate_content"
                continue
            seen.add(content_id)
            retained_items.append(source)
        retained = tuple(retained_items)
    elif experiment.strategy == "focused":
        ranked = tuple(sorted(
            original,
            key=lambda source: (source_hash(source) not in expected, source.source_id),
        ))
        retained = ranked[:experiment.max_sources]
        reasons = {source.source_id: "outside_focused_bound" for source in ranked[experiment.max_sources:]}
    elif experiment.strategy == "top_n":
        retained = original[:experiment.max_sources]
        reasons = {source.source_id: "outside_top_n_bound" for source in original[experiment.max_sources:]}
    else:
        raise ValueError(f"Unsupported grounding strategy: {experiment.strategy}")
    retained_ids = {source.source_id for source in retained}
    dropped = [
        {**source_record(source), "reason": reasons.get(source.source_id, "not_retained")}
        for source in original if source.source_id not in retained_ids
    ]
    return retained, [source_record(source) for source in retained], dropped


_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "�")
_APOSTROPHE_VARIANTS = "'`´‘’‛ʼ＇"


def normalize_evaluation_text(value: object) -> str:
    """Return a conservative, comparison-only canonical form of evaluation text.

    The benchmark is never rewritten.  This representation repairs common UTF-8
    decoded-as-Latin-1 artifacts, then removes presentation-only differences:
    accents, apostrophe glyphs, case, punctuation, and repeated whitespace.
    """

    # Repair mojibake before Unicode compatibility normalization; NFKC can
    # collapse the intermediate Latin-1 markers needed for byte recovery.
    text = str(value or "")
    for _ in range(3):
        if not any(marker in text for marker in _MOJIBAKE_MARKERS):
            break
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            break
        if repaired == text:
            break
        text = repaired
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({character: " " for character in _APOSTROPHE_VARIANTS}))
    text = "".join(
        character for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.casefold())).strip()


_EVIDENCE_STOPWORDS = {
    "the", "and", "for", "what", "where", "which", "how", "many", "are", "is", "to",
    "les", "des", "une", "quel", "quelle", "quels", "quelles", "combien", "ou", "est", "sont",
}


def _split_evidence_sentences(text: str) -> tuple[str, ...]:
    """Split text without changing any retained passage characters."""

    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return tuple(piece.strip() for piece in pieces if piece.strip())


def _evidence_match_features(query: str, passage: str) -> tuple[float, tuple[str, ...]]:
    normalized_query = normalize_evaluation_text(query)
    normalized_passage = normalize_evaluation_text(passage)
    query_terms = {
        term for term in normalized_query.split()
        if len(term) > 2 and term not in _EVIDENCE_STOPWORDS
    }
    passage_terms = set(normalized_passage.split())
    matched = set(query_terms & passage_terms)
    # Complementary lexical coverage handles inflectional variants such as
    # ``ouverture``/``ouverte`` without introducing a synonym model.
    for query_term in query_terms:
        if len(query_term) < 5:
            continue
        if any(
            passage_term.startswith(query_term[:5]) or query_term.startswith(passage_term[:5])
            for passage_term in passage_terms if len(passage_term) >= 5
        ):
            matched.add(query_term)
    exact_values = set(re.findall(r"\b\d+(?:[.,:]\d+)?\b", query))
    passage_values = set(re.findall(r"\b\d+(?:[.,:]\d+)?\b", passage))
    matched_values = exact_values & passage_values
    matched.update(matched_values)
    query_acronyms = set(re.findall(r"\b[A-Z][A-Za-z0-9]{1,}\b", query))
    passage_acronyms = set(re.findall(r"\b[A-Z][A-Za-z0-9]{1,}\b", passage))
    matched_acronyms = query_acronyms & passage_acronyms
    matched.update(matched_acronyms)
    time_patterns = set(re.findall(r"\b\d{1,2}h\d{2}\b", passage, flags=re.IGNORECASE))
    time_intent = any(term.startswith("heure") or term in {"ouverture", "horaire"} for term in query_terms)
    if time_intent and time_patterns:
        matched.update(query_term for query_term in query_terms if query_term.startswith("heure") or query_term in {"ouverture", "horaire"})
        matched.update(f"time:{value}" for value in sorted(time_patterns))
    denominator = max(1, len(query_terms) + len(exact_values) + len(query_acronyms))
    score = (len(matched) + len(matched_values) + len(matched_acronyms)) / denominator
    return score, tuple(sorted(matched, key=lambda value: (value.casefold(), value)))


def extract_evidence(
    trace: rag_pipeline.PipelineTrace, *, max_passages: int = 3,
) -> EvidenceExtractionResult:
    """Extract explicit passages from a certified trace without benchmark leakage."""

    query = trace.rewritten_query or trace.query
    prompt = trace.prompt
    if prompt is None or not prompt.sources:
        return EvidenceExtractionResult(
            "NO_EXPLICIT_EVIDENCE", query, trace.language, (), (), False, "no_prompt_sources",
        )
    candidates: list[tuple[float, int, int, rag_pipeline.PromptSource, str, tuple[str, ...]]] = []
    seen: set[tuple[str, str]] = set()
    for source in prompt.sources:
        for sentence_index, passage in enumerate(_split_evidence_sentences(source.text)):
            key = (source_hash(source), normalize_evaluation_text(passage))
            if key in seen:
                continue
            seen.add(key)
            score, matched_terms = _evidence_match_features(query, passage)
            if score <= 0 or not matched_terms:
                continue
            candidates.append((score, source.source_id, sentence_index, source, passage, matched_terms))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], normalize_evaluation_text(item[4])))
    selected: list[tuple[float, int, int, rag_pipeline.PromptSource, str, tuple[str, ...]]] = []
    covered_terms: set[str] = set()
    for candidate in candidates:
        if len(selected) >= max(1, max_passages):
            break
        candidate_terms = set(candidate[5])
        if selected and not candidate_terms.difference(covered_terms):
            continue
        selected.append(candidate)
        covered_terms.update(candidate_terms)
    passages = tuple(
        EvidencePassage(
            evidence_id=f"E{index}", source_id=source.source_id, content_sha256=source_hash(source),
            source_file=source.file_name, location=source.location, text=passage,
            sentence_index=sentence_index, match_score=round(score, 8), matched_terms=matched_terms,
        )
        for index, (score, _source_id, sentence_index, source, passage, matched_terms) in enumerate(selected, 1)
    )
    source_ids = tuple(sorted({passage.source_id for passage in passages}))
    if not passages:
        return EvidenceExtractionResult(
            "NO_EXPLICIT_EVIDENCE", query, trace.language, (), (), False, "no_query_supported_passage",
        )
    return EvidenceExtractionResult("EVIDENCE_FOUND", query, trace.language, passages, source_ids, True)


def build_extractive_answer(
    evidence: EvidenceExtractionResult, language: str | None = None,
) -> ExtractiveAnswerResult:
    """Assemble an answer directly from evidence, without model generation."""

    started = time.perf_counter()
    if evidence.status != "EVIDENCE_FOUND" or not evidence.passages:
        return ExtractiveAnswerResult(
            status="NO_EXPLICIT_EVIDENCE",
            answer_text=rag_pipeline.build_clarification_message(language or "French"),
            evidence_ids=(), source_ids=(), sources=(), passage_hashes=(), citation_ids=(),
            latency_ms=(time.perf_counter() - started) * 1000,
            failure_reason=evidence.failure_reason,
        )
    answer_parts: list[str] = []
    source_records: list[dict[str, Any]] = []
    seen_passages: set[tuple[int, str]] = set()
    for passage in evidence.passages:
        key = (passage.source_id, passage.text)
        if key in seen_passages:
            continue
        seen_passages.add(key)
        answer_parts.append(f"{passage.text} [SOURCE {passage.source_id}]")
        source_records.append({
            "source_id": passage.source_id,
            "source_file": passage.source_file,
            "location": passage.location,
            "content_sha256": passage.content_sha256,
            "evidence_id": passage.evidence_id,
        })
    source_ids = tuple(record["source_id"] for record in source_records)
    return ExtractiveAnswerResult(
        status="ANSWER", answer_text="\n\n".join(answer_parts),
        evidence_ids=tuple(record["evidence_id"] for record in source_records),
        source_ids=source_ids, sources=tuple(source_records),
        passage_hashes=tuple(record["content_sha256"] for record in source_records),
        citation_ids=source_ids, latency_ms=(time.perf_counter() - started) * 1000,
    )


# Shared runtime is the single source of truth; these aliases preserve the
# evaluator's established public names without retaining a second behavior.
EvidencePassage = rag_pipeline.EvidencePassage
EvidenceExtractionResult = rag_pipeline.EvidenceExtractionResult
ExtractiveAnswerResult = rag_pipeline.ExtractiveAnswerResult
extract_evidence = rag_pipeline.extract_evidence
build_extractive_answer = rag_pipeline.build_extractive_answer


def run_extractive_answer_two_passes(
    cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime,
    output_dir: Path, certified_controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run deterministic control/extractive comparisons twice from certified traces."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_before = grounding_runtime_fingerprint(runtime)
    passes: dict[str, Any] = {}
    for pass_number in (1, 2):
        rows: dict[str, list[dict[str, Any]]] = {"control": [], "extractive": []}
        checkpoint = output_dir / f"run_{pass_number}_checkpoint.json"
        for case in cases:
            control = certified_controls[case["id"]]
            trace = control["trace"]
            evidence = extract_evidence(trace)
            support_sources = tuple(
                source for source in (trace.prompt.sources if trace.prompt else ())
                if source.source_id in evidence.supporting_source_ids
            )
            rows["control"].append({
                "case_id": case["id"], "variant": "control", "response": control.get("response", ""),
                "metrics": control.get("metrics", {}), "retrieval_parity": True,
            })
            answer = build_extractive_answer(evidence, language_label(case["language"]))
            expected_hashes = {
                item["content_sha256"] for item in case.get("acceptable_citations", [])
            }
            evidence_hashes = set(answer.passage_hashes)
            citation_valid = bool(answer.citation_ids) and set(answer.citation_ids) <= set(answer.source_ids)
            expected_source_match = bool(expected_hashes) and expected_hashes <= evidence_hashes
            quality = grounding_quality_metrics(case, answer.answer_text, support_sources)
            rows["extractive"].append({
                "case_id": case["id"], "variant": "extractive", "response": answer.answer_text,
                "status": answer.status, "metrics": {
                    **quality,
                    "citation_valid": citation_valid if case["answerability"] == "answerable" else None,
                    "expected_source_match": expected_source_match if case["answerability"] == "answerable" else None,
                    "latency_ms": answer.latency_ms, "generation_timeout_count": 0,
                    "unsupported_answer_point_count": answer.unsupported_claim_count,
                }, "retrieval_parity": retrieval_parity(trace, trace),
                "extractive_answer": answer.to_json(), "evidence": evidence.to_json(),
            })
            checkpoint.write_text(json.dumps({"status": "PARTIAL", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        fingerprint_after = grounding_runtime_fingerprint(runtime)
        if fingerprint_before != fingerprint_after:
            raise RuntimeError("Extractive answer fingerprint changed during evaluation")
        summary = {"status": "COMPLETE", "fingerprint_match": True, "rows": rows}
        checkpoint.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        passes[f"run_{pass_number}"] = summary
    return {"status": "COMPLETE", "fingerprint": fingerprint_before, "passes": passes}


def source_faithful_answer_variants(
    case: Mapping[str, Any], sources: Sequence[rag_pipeline.PromptSource],
) -> tuple[str, ...]:
    """Expose only benchmark reference answers explicitly present in support chunks.

    This does not alter benchmark facts or invent paraphrases.  It permits a
    source-faithful reference answer to satisfy a case whose short answer point
    uses a different, but equivalent, grammatical form.
    """

    expected_answer = str(case.get("expected_answer") or "").strip()
    if not expected_answer:
        return ()
    expected_hashes = {
        item["content_sha256"] for item in case.get("relevance", []) if item.get("label", 0) > 0
    }
    supporting_text = "\n".join(
        source.text for source in sources if source_hash(source) in expected_hashes
    )
    normalized_answer = normalize_evaluation_text(expected_answer)
    if normalized_answer and normalized_answer in normalize_evaluation_text(supporting_text):
        return (expected_answer,)
    return ()


def grounding_quality_metrics(
    case: Mapping[str, Any], response: str, sources: Sequence[rag_pipeline.PromptSource],
) -> dict[str, bool | int | None]:
    """Measure literal and canonical answer-point use with explicit prompt support."""

    if case["answerability"] != "answerable":
        return {
            "grounded_answer_correct": None, "answer_use": None,
            "grounded_answer_correct_exact": None, "grounded_answer_correct_normalized": None,
            "answer_use_exact": None, "answer_use_normalized": None,
            "blank_output": not response.strip(), "unsupported_answer_point_count": 0,
        }
    literal_response = response.casefold()
    literal_points = [str(point).casefold() for point in case.get("acceptable_answer_points", [])]
    normalized_response = normalize_evaluation_text(response)
    normalized_points = [normalize_evaluation_text(point) for point in case.get("acceptable_answer_points", [])]
    source_variants = source_faithful_answer_variants(case, sources)
    normalized_variants = [normalize_evaluation_text(variant) for variant in source_variants]
    exact_match = bool(literal_points) and all(point in literal_response for point in literal_points)
    normalized_points_match = bool(normalized_points) and all(
        point in normalized_response for point in normalized_points
    )
    source_variant_match = any(variant in normalized_response for variant in normalized_variants)
    normalized_match = normalized_points_match or source_variant_match
    expected_hashes = {item["content_sha256"] for item in case["relevance"] if item["label"] > 0}
    context_hashes = {source_hash(source) for source in sources}
    unsupported_answer_points = sum(
        point in normalized_response and not expected_hashes <= context_hashes for point in normalized_points
    )
    return {
        "grounded_answer_correct": normalized_match and expected_hashes <= context_hashes,
        "answer_use": normalized_match,
        "grounded_answer_correct_exact": exact_match and expected_hashes <= context_hashes,
        "grounded_answer_correct_normalized": normalized_match and expected_hashes <= context_hashes,
        "answer_use_exact": exact_match,
        "answer_use_normalized": normalized_match,
        "blank_output": not response.strip(),
        "unsupported_answer_point_count": unsupported_answer_points,
    }


def citation_obligation_metrics(
    case: Mapping[str, Any], citation: Mapping[str, bool | None],
) -> dict[str, bool | int | None]:
    """Evaluate citation compliance over every answerable case, not only cited cases."""

    if case["answerability"] != "answerable":
        return {"citation_obligation_required": None, "citation_obligation_met": None}
    return {
        "citation_obligation_required": 1,
        "citation_obligation_met": bool(citation["citation_valid"] and citation["expected_source_match"]),
    }


def append_evaluator_prompt_suffix(
    prompt: rag_pipeline.PromptResult, suffix: str,
) -> rag_pipeline.PromptResult:
    """Append an experiment instruction without altering the production prompt builder."""

    return rag_pipeline.PromptResult(
        prompt=f"{prompt.prompt}\n\n{suffix}", sources=prompt.sources, context=prompt.context,
    )


def retrieval_parity(control: rag_pipeline.PipelineTrace, variant: rag_pipeline.PipelineTrace) -> bool:
    """Prove a context experiment retained the exact certified retrieval trace."""

    return (
        control.query == variant.query
        and control.rewritten_query == variant.rewritten_query
        and control.metadata_filter == variant.metadata_filter
        and control.retrieval == variant.retrieval
    )


def evaluate_grounding_variant(
    case: Mapping[str, Any], control: Mapping[str, Any], runtime: EvaluationRuntime,
    generator: rag_pipeline.TextGenerator, experiment: ContextGroundingExperiment,
    *, clock: Callable[[], float] = time.perf_counter, stage_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Generate from an alternate evaluator context while preserving retrieval exactly."""

    control_trace = control["trace"]
    if not isinstance(control_trace, rag_pipeline.PipelineTrace):
        raise ValueError("Grounding experiments require an in-memory certified control trace.")
    if control_trace.failure or not control_trace.prompt:
        failure_code = control_trace.failure.code if control_trace.failure else "control_prompt_unavailable"
        trace = rag_pipeline.PipelineTrace(
            query=control_trace.query, rewritten_query=control_trace.rewritten_query,
            language=control_trace.language, metadata_filter=control_trace.metadata_filter,
            retrieval=control_trace.retrieval,
            failure=rag_pipeline.PipelineFailure(
                code="not_run_control_failed",
                message=f"{experiment.name} was not run because G0_control failed: {failure_code}",
            ),
        )
        citation = citation_metrics(None, case["acceptable_citations"])
        metrics = {
            **{name: None for name in ("recall_at_k", "precision_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k")},
            **citation,
            "refusal_correct": None,
            "clarification_correct": None,
            "latency_ms": None,
            "citation_obligation_required": None,
            "citation_obligation_met": None,
            "grounded_answer_correct": None,
            "answer_use": None,
            "blank_output": None,
            "unsupported_answer_point_count": 0,
        }
        return {
            "case_id": case["id"], "experiment": experiment.name, "status": "NOT_RUN",
            "not_run_reason": f"G0_control:{failure_code}", "language": case["language"],
            "query_type": case["category"],
            "metadata_filter_state": "filtered" if control_trace.metadata_filter else "unfiltered",
            "answerability": case["answerability"], "expected_behavior": case["expected_behavior"],
            "metrics": metrics, "response": "", "trace": trace,
            "stage_timings_ms": {},
            "grounding_context": {"retained_chunks": [], "dropped_chunks": [], "retrieval_parity": True},
        }
    sources, retained, dropped = grounding_context(control_trace.prompt.sources, case["relevance"], experiment)
    started = clock()
    stage_timings: dict[str, float] = {}
    stage = "prompt_construction"
    try:
        prompt, stage_timings["prompt_construction"] = run_stage("prompt_construction", lambda: rag_pipeline.build_production_prompt(
            user_query=case["question"], filter_ent="Tous", filter_application="Tous",
            history=case["conversation"], sources=sources,
            current_lang=language_label(case["language"]),
            was_relaxed=bool(control_trace.retrieval and control_trace.retrieval.fallback_used),
        ), stage_timeout_seconds, clock=clock)
        if experiment.prompt_suffix:
            prompt = append_evaluator_prompt_suffix(prompt, experiment.prompt_suffix)
        stage = "generation"
        generation_request = LocalModelStageRequest(
            stage="generation", model_name=runtime.config.llm_model_name, prompt=prompt.prompt,
            clarification_language=language_label(case["language"]),
            max_output_tokens=(generator.max_output_tokens if isinstance(generator, EvaluatorGenerationBudget) else None),
            timeout_seconds=stage_timeout_seconds,
        )
        generation, stage_timings["generation"] = run_model_stage(
            generation_request, generator,
            lambda: rag_pipeline.stream_generate(
                prompt.prompt, runtime.config.llm_model_name, generator, clock=clock,
                clarification_language=language_label(case["language"]),
            ),
            clock=clock,
        )
    except StageTimeoutError as error:
        result = timeout_result(case, error, started, stage_timings, clock)
        result["experiment"] = experiment.name
        result["grounding_context"] = {"retained_chunks": retained, "dropped_chunks": dropped, "retrieval_parity": True}
        return result
    except Exception as error:
        failure = rag_pipeline.PipelineFailure(code=f"{stage}_error", message=str(error))
        trace = rag_pipeline.PipelineTrace(
            query=control_trace.query, rewritten_query=control_trace.rewritten_query,
            language=control_trace.language, metadata_filter=control_trace.metadata_filter,
            retrieval=control_trace.retrieval, prompt=control_trace.prompt,
            generation=rag_pipeline.GenerationResult(error=str(error)), failure=failure,
        )
        metrics = dict(control["metrics"])
        citation = citation_metrics(None, case["acceptable_citations"])
        metrics.update(citation)
        metrics.update(citation_obligation_metrics(case, citation))
        metrics.update(grounding_quality_metrics(case, "", sources))
        metrics["latency_ms"] = (clock() - started) * 1000
        return {
            "case_id": case["id"], "experiment": experiment.name, "language": case["language"],
            "query_type": case["category"],
            "metadata_filter_state": "filtered" if control_trace.metadata_filter else "unfiltered",
            "answerability": case["answerability"], "expected_behavior": case["expected_behavior"],
            "metrics": metrics, "response": "", "trace": trace, "stage_timings_ms": stage_timings,
            "grounding_context": {"retained_chunks": retained, "dropped_chunks": dropped, "retrieval_parity": retrieval_parity(control_trace, trace)},
        }
    try:
        citations = rag_pipeline.select_display_sources(generation.response, sources)
        failure = None
    except Exception as error:
        citations = None
        failure = rag_pipeline.PipelineFailure(code="citation_error", message=str(error))
    metrics = dict(control["metrics"])
    citation = citation_metrics(citations, case["acceptable_citations"])
    metrics.update(citation)
    metrics.update(citation_obligation_metrics(case, citation))
    expected_mode = case["expected_behavior"]["mode"]
    is_refusal = citations.no_coverage_detected and not citations.display_sources
    is_clarification = generation.response == rag_pipeline.build_clarification_message(language_label(case["language"]))
    allows_no_source_outcome = case["expected_behavior"].get("source_display") == "none"
    metrics["refusal_correct"] = (
        is_refusal or (allows_no_source_outcome and is_clarification)
        if expected_mode == "refuse_no_coverage" else None
    )
    metrics["clarification_correct"] = (
        is_clarification if expected_mode == "request_clarification" or allows_no_source_outcome else None
    )
    metrics.update(grounding_quality_metrics(case, generation.response, sources))
    metrics["latency_ms"] = (clock() - started) * 1000
    trace = rag_pipeline.PipelineTrace(
        query=control_trace.query, rewritten_query=control_trace.rewritten_query,
        language=control_trace.language, metadata_filter=control_trace.metadata_filter,
        retrieval=control_trace.retrieval, prompt=prompt, generation=generation, citations=citations,
        timings=rag_pipeline.PipelineTimings(
            rewrite_ms=control_trace.timings.rewrite_ms,
            generation_ms=generation.latency_ms, total_ms=metrics["latency_ms"],
        ), failure=failure,
    )
    return {
        "case_id": case["id"], "experiment": experiment.name, "language": case["language"],
        "query_type": case["category"],
        "metadata_filter_state": "filtered" if control_trace.metadata_filter else "unfiltered",
        "answerability": case["answerability"], "expected_behavior": case["expected_behavior"],
        "metrics": metrics, "response": generation.response, "trace": trace,
        "stage_timings_ms": stage_timings,
        "grounding_context": {
            "retained_chunks": retained, "dropped_chunks": dropped,
            "retrieval_parity": retrieval_parity(control_trace, trace),
        },
    }


def grounding_experiment_report(results: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Summarize G0-G3 quality evidence without selecting a production winner."""

    variants: dict[str, Any] = {}
    for name, rows in results.items():
        serialized = [result_to_json(row) for row in rows]
        quality = {
            metric: (
                sum(bool(row["metrics"].get(metric)) for row in rows if row["metrics"].get(metric) is not None)
                / sum(row["metrics"].get(metric) is not None for row in rows)
                if any(row["metrics"].get(metric) is not None for row in rows) else None
            )
            for metric in ("grounded_answer_correct", "answer_use", "blank_output")
        }
        obligation_rows = [row for row in rows if row["metrics"].get("citation_obligation_required")]
        quality["citation_obligation_case_count"] = len(obligation_rows)
        quality["citation_obligation_coverage"] = (
            sum(bool(row["metrics"].get("citation_obligation_met")) for row in obligation_rows) / len(obligation_rows)
            if obligation_rows else None
        )
        quality["unsupported_answer_point_count"] = sum(
            int(row["metrics"].get("unsupported_answer_point_count") or 0) for row in rows
        )
        quality["not_run_case_count"] = sum(row.get("status") == "NOT_RUN" for row in rows)
        forensic_counts: dict[str, int] = {}
        for row in serialized:
            category = row.get("forensic_category", "unknown")
            forensic_counts[category] = forensic_counts.get(category, 0) + 1
        variants[name] = {
            "metrics": {**aggregate(rows), **quality},
            "retrieval_parity": all(row.get("grounding_context", {}).get("retrieval_parity", True) for row in rows),
            "forensic_counts": forensic_counts,
        }
    return {"control": "G0_control", "variants": variants}


def metric_direction(control: float | int | None, variant: float | int | None, *, lower_is_better: bool = False) -> str:
    """Describe a variant delta without claiming statistical significance."""

    if control is None or variant is None:
        return "NOT_EVALUABLE"
    if variant == control:
        return "UNCHANGED"
    improved = variant < control if lower_is_better else variant > control
    return "IMPROVED" if improved else "REGRESSED"


def compare_grounding_stability(
    first: Mapping[str, Any], second: Mapping[str, Any],
    first_fingerprint: Mapping[str, Any], second_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two completed experiment reports; this never promotes a variant."""

    metrics = {
        "grounded_answer_correct": False, "answer_use": False,
        "expected_source_match": False, "citation_valid": False,
        "refusal_correct": False, "clarification_correct": False,
        "blank_output": True, "unsupported_answer_point_count": True,
        "latency_ms": True, "citation_obligation_coverage": False,
    }
    variants: dict[str, Any] = {}
    first_variants = first.get("variants", {})
    second_variants = second.get("variants", {})
    for name in sorted(set(first_variants) & set(second_variants)):
        if name == "G0_control":
            continue
        directions: dict[str, Any] = {}
        for metric, lower_is_better in metrics.items():
            run_one = metric_direction(
                first_variants["G0_control"]["metrics"].get(metric),
                first_variants[name]["metrics"].get(metric), lower_is_better=lower_is_better,
            )
            run_two = metric_direction(
                second_variants["G0_control"]["metrics"].get(metric),
                second_variants[name]["metrics"].get(metric), lower_is_better=lower_is_better,
            )
            directions[metric] = {
                "first_run": run_one, "second_run": run_two,
                "stable": run_one == run_two and run_one != "NOT_EVALUABLE",
            }
        variants[name] = {
            "complete": all(
                report["metrics"].get("generation_timeout_count") == 0
                and report["metrics"].get("not_run_case_count", 0) == 0
                for report in (first_variants[name], second_variants[name])
            ),
            "retrieval_parity": all(
                report.get("retrieval_parity") is True
                for report in (first_variants[name], second_variants[name])
            ),
            "metric_directions": directions,
        }
    return {
        "fingerprint_match": dict(first_fingerprint) == dict(second_fingerprint),
        "variants": variants,
    }


def run_grounding_experiments(
    cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime, generator: rag_pipeline.TextGenerator,
    *, stage_timeout_seconds: float = 120.0,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Run G0-G3 from one retrieval execution per case and return ignored-run artifacts."""

    rows: dict[str, list[dict[str, Any]]] = {experiment.name: [] for experiment in GROUNDING_EXPERIMENTS}
    for case in cases:
        control = evaluate_case(case, runtime, generator, stage_timeout_seconds=stage_timeout_seconds)
        control_sources = control["trace"].prompt.sources if isinstance(control["trace"], rag_pipeline.PipelineTrace) and control["trace"].prompt else ()
        retained, retained_records, dropped = grounding_context(control_sources, case["relevance"], GROUNDING_EXPERIMENTS[0])
        control = dict(control)
        control["experiment"] = GROUNDING_EXPERIMENTS[0].name
        control_citations = citation_metrics(control["trace"].citations, case["acceptable_citations"])
        control["metrics"] = {
            **control["metrics"],
            **grounding_quality_metrics(case, control["response"], retained),
            **citation_obligation_metrics(case, control_citations),
        }
        control["grounding_context"] = {"retained_chunks": retained_records, "dropped_chunks": dropped, "retrieval_parity": True}
        rows[GROUNDING_EXPERIMENTS[0].name].append(control)
        for experiment in GROUNDING_EXPERIMENTS[1:]:
            rows[experiment.name].append(evaluate_grounding_variant(
                case, control, runtime, generator, experiment, stage_timeout_seconds=stage_timeout_seconds,
            ))
    return grounding_experiment_report(rows), rows


def write_grounding_experiment_report(report: Mapping[str, Any], rows: Mapping[str, Sequence[Mapping[str, Any]]], output_dir: Path) -> None:
    """Persist G0-G3 reports only below the ignored evaluator-run directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        name: [
            {**result_to_json(row), "grounding_context": row.get("grounding_context", {})}
            for row in variant_rows
        ]
        for name, variant_rows in rows.items()
    }
    (output_dir / "context_grounding_experiments.json").write_text(
        json.dumps({"summary": report, "cases": payload}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def grounding_runtime_fingerprint(runtime: EvaluationRuntime) -> dict[str, Any]:
    """Return the compact read-only corpus/runtime fingerprint for one pass.

    Grounding experiments must use one immutable corpus.  The fingerprint is
    calculated before and after each pass and persisted with the partial report;
    it never changes the collection or production runtime.
    """

    data = runtime.collection.get(include=["documents", "metadatas"])
    documents = list(data.get("documents") or [])
    metadatas = list(data.get("metadatas") or [])
    rows = [
        {
            "chunk_id": str(index),
            "content_sha256": hashlib.sha256(str(document).encode("utf-8")).hexdigest(),
            "metadata": metadata or {},
        }
        for index, (document, metadata) in enumerate(zip(documents, metadatas))
    ]
    canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    rows.sort(key=lambda row: row["chunk_id"])
    corpus_sha256 = hashlib.sha256("\n".join(canonical(row) for row in rows).encode("utf-8")).hexdigest()
    metadata_sha256 = hashlib.sha256(
        "\n".join(canonical(row["metadata"]) for row in rows).encode("utf-8")
    ).hexdigest()
    return {
        "corpus_sha256": corpus_sha256,
        "metadata_sha256": metadata_sha256,
        "runtime_sha256": hashlib.sha256(canonical({
            "collection_name": runtime.config.collection_name,
            "embedding_model_name": runtime.config.embedding_model_name,
            "llm_model_name": runtime.config.llm_model_name,
            "vector_candidate_count": runtime.config.vector_candidate_count,
            "bm25_candidate_count": runtime.config.bm25_candidate_count,
            "rrf_k": runtime.config.rrf_k,
            "production_top_k": runtime.config.production_top_k,
            "min_results_before_relax": runtime.config.min_results_before_relax,
            "chunk_count": len(rows),
            "corpus_sha256": corpus_sha256,
            "metadata_sha256": metadata_sha256,
        }).encode("utf-8")).hexdigest(),
        "chunk_count": len(rows),
    }


def _grounding_checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "grounding_checkpoint.pkl"


def _load_grounding_checkpoint(
    output_dir: Path, experiments: Sequence[ContextGroundingExperiment] = GROUNDING_EXPERIMENTS,
) -> dict[str, Any]:
    path = _grounding_checkpoint_path(output_dir)
    if not path.exists():
        return {"rows": {name: [] for name in (experiment.name for experiment in experiments)}, "completed": {}}
    with path.open("rb") as stream:
        state = pickle.load(stream)
    if not isinstance(state, dict) or not isinstance(state.get("rows"), dict):
        raise ValueError("Invalid grounding checkpoint format")
    return state


def _write_grounding_checkpoint(output_dir: Path, state: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _grounding_checkpoint_path(output_dir)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def _grounding_partial_reports(
    output_dir: Path, rows: Mapping[str, Sequence[Mapping[str, Any]]], *,
    expected_case_count: int, fingerprint_before: Mapping[str, Any], fingerprint_after: Mapping[str, Any] | None,
    status: str, experiments: Sequence[ContextGroundingExperiment] = GROUNDING_EXPERIMENTS,
) -> None:
    """Persist case rows and a partial summary after every variant."""

    report = grounding_experiment_report(rows)
    write_grounding_experiment_report(report, rows, output_dir)
    completed = sum(len(values) for values in rows.values())
    timeout_count = sum(
        timeout_code(result) is not None
        for values in rows.values() for result in values
    )
    error_count = sum(
        bool((result_to_json(result).get("trace") or {}).get("failure"))
        for values in rows.values() for result in values
    )
    summary = {
        "status": status,
        "completed_variant_count": completed,
        "expected_variant_count": expected_case_count * len(experiments),
        "timeout_count": timeout_count,
        "error_count": error_count,
        "fingerprint_before": dict(fingerprint_before),
        "fingerprint_after": dict(fingerprint_after) if fingerprint_after is not None else None,
        "fingerprint_match": fingerprint_after is not None and dict(fingerprint_before) == dict(fingerprint_after),
        "variants": report.get("variants", {}),
    }
    (output_dir / "partial_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "partial_summary.md").write_text(
        "# Grounding experiment partial summary\n\n" +
        "\n".join(f"- {key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )


def run_grounding_experiments_resumable(
    cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime, generator: rag_pipeline.TextGenerator,
    output_dir: Path, *, resume: bool = False, stage_timeout_seconds: float = 120.0,
    max_new_variants: int | None = None,
    experiments: Sequence[ContextGroundingExperiment] = PROMPT_GROUNDING_EXPERIMENTS,
    certified_controls: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Run G0/G4/G5 with variant checkpoints.

    Each variant is persisted immediately.  A resume skips completed
    ``case_id/variant`` keys, including a partially completed case.  The
    optional ``max_new_variants`` exists only to exercise interruption/resume
    behavior in tests; normal callers leave it unset.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = tuple(experiments)
    state = _load_grounding_checkpoint(output_dir, experiments) if resume else {
        "rows": {experiment.name: [] for experiment in experiments},
        "completed": {},
    }
    rows: dict[str, list[dict[str, Any]]] = {
        experiment.name: list(state.get("rows", {}).get(experiment.name, []))
        for experiment in experiments
    }
    completed = dict(state.get("completed", {}))
    fingerprint_before = grounding_runtime_fingerprint(runtime)
    written = 0

    for case in cases:
        control = next((row for row in rows[experiments[0].name] if row.get("case_id") == case["id"]), None)
        for experiment in experiments:
            key = f"{case['id']}::{experiment.name}"
            if key in completed:
                continue
            if experiment.name == experiments[0].name:
                control = dict(certified_controls[case["id"]]) if certified_controls and case["id"] in certified_controls else evaluate_case(
                    case, runtime, generator, stage_timeout_seconds=stage_timeout_seconds
                )
                control_sources = control["trace"].prompt.sources if isinstance(control["trace"], rag_pipeline.PipelineTrace) and control["trace"].prompt else ()
                retained, retained_records, dropped = grounding_context(control_sources, case["relevance"], experiment)
                control = dict(control)
                control["experiment"] = experiment.name
                control_citations = citation_metrics(control["trace"].citations, case["acceptable_citations"])
                control["metrics"] = {
                    **control["metrics"],
                    **grounding_quality_metrics(case, control["response"], retained),
                    **citation_obligation_metrics(case, control_citations),
                }
                control["grounding_context"] = {"retained_chunks": retained_records, "dropped_chunks": dropped, "retrieval_parity": True}
                result = control
            else:
                if control is None:
                    raise ValueError(f"Cannot resume {key}: missing persisted control result")
                result = evaluate_grounding_variant(case, control, runtime, generator, experiment, stage_timeout_seconds=stage_timeout_seconds)
            rows[experiment.name] = [row for row in rows[experiment.name] if row.get("case_id") != case["id"]]
            rows[experiment.name].append(result)
            completed[key] = {"case_id": case["id"], "experiment": experiment.name}
            written += 1
            checkpoint = {
                "schema_version": 1,
                "rows": rows,
                "completed": completed,
                "fingerprint_before": fingerprint_before,
            }
            _write_grounding_checkpoint(output_dir, checkpoint)
            _grounding_partial_reports(
                output_dir, rows, expected_case_count=len(cases), fingerprint_before=fingerprint_before,
                fingerprint_after=None, status="PARTIAL", experiments=experiments,
            )
            if max_new_variants is not None and written >= max_new_variants:
                return grounding_experiment_report(rows), rows

    fingerprint_after = grounding_runtime_fingerprint(runtime)
    if fingerprint_before != fingerprint_after:
        raise RuntimeError("Grounding pass fingerprint changed during evaluation")
    state = {"schema_version": 1, "rows": rows, "completed": completed, "fingerprint_before": fingerprint_before, "fingerprint_after": fingerprint_after}
    _write_grounding_checkpoint(output_dir, state)
    _grounding_partial_reports(
        output_dir, rows, expected_case_count=len(cases), fingerprint_before=fingerprint_before,
        fingerprint_after=fingerprint_after, status="COMPLETE", experiments=experiments,
    )
    return grounding_experiment_report(rows), rows


def run_grounding_two_passes_resumable(
    cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime, generator: rag_pipeline.TextGenerator,
    output_dir: Path, *, resume: bool = False, stage_timeout_seconds: float = 120.0,
    experiments: Sequence[ContextGroundingExperiment] = PROMPT_GROUNDING_EXPERIMENTS,
    certified_controls: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run two independent resumable G0/G4/G5 passes with live summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    passes: dict[str, Any] = {}
    for number in (1, 2):
        pass_dir = output_dir / f"run_{number}"
        report, _rows = run_grounding_experiments_resumable(
            cases, runtime, generator, pass_dir, resume=resume,
            stage_timeout_seconds=stage_timeout_seconds,
            experiments=experiments,
            certified_controls=certified_controls,
        )
        partial_path = pass_dir / "partial_summary.json"
        partial = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}
        passes[f"run_{number}"] = {"report": report, "partial_summary": partial}
        (output_dir / "two_pass_summary.json").write_text(
            json.dumps({
                "status": "COMPLETE" if number == 2 and partial.get("status") == "COMPLETE" else "PARTIAL",
                "passes": passes,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if partial.get("status") != "COMPLETE":
            break
    complete = len(passes) == 2 and all(
        item["partial_summary"].get("status") == "COMPLETE" for item in passes.values()
    )
    return {"status": "COMPLETE" if complete else "PARTIAL", "passes": passes}


def run_compact_context_two_passes_resumable(
    cases: Sequence[Mapping[str, Any]], runtime: EvaluationRuntime, generator: rag_pipeline.TextGenerator,
    output_dir: Path, *, resume: bool = False, stage_timeout_seconds: float = 120.0,
    certified_controls: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the approved C15/C5/C3 context experiment twice with checkpoints."""

    return run_grounding_two_passes_resumable(
        cases, runtime, generator, output_dir, resume=resume,
        stage_timeout_seconds=stage_timeout_seconds, experiments=COMPACT_CONTEXT_EXPERIMENTS,
        certified_controls=certified_controls,
    )


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
    return {"case_id": case["id"], "language": case["language"], "query_type": case["category"], "metadata_filter_state": "filtered" if case["metadata_filter"] else "unfiltered", "answerability": case["answerability"], "expected_behavior": case["expected_behavior"], "metrics": {"recall_at_k": None, "precision_at_k": None, "hit_rate_at_k": None, "mrr": None, "ndcg_at_k": None, "citation_valid": None, "expected_source_match": None, "refusal_correct": None, "clarification_correct": None, "latency_ms": (clock() - started) * 1000}, "response": "", "trace": trace, "stage_timings_ms": dict(stage_timings)}


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
    for name in ("citation_valid", "expected_source_match", "refusal_correct", "clarification_correct"):
        values = [result["metrics"][name] for result in results if result["metrics"][name] is not None]
        summary[name] = sum(values) / len(values) if values else None
    summary["citation_evaluable_case_count"] = sum(
        result["metrics"]["citation_valid"] is not None for result in results
    )
    summary["expected_source_evaluable_case_count"] = sum(
        result["metrics"]["expected_source_match"] is not None for result in results
    )
    def failure_code(result: Mapping[str, Any]) -> str | None:
        trace = result["trace"]
        failure = trace.get("failure") if isinstance(trace, Mapping) else trace.failure
        return failure.get("code") if isinstance(failure, Mapping) else (failure.code if failure else None)
    summary["generation_timeout_count"] = sum(failure_code(result) == "generation_timeout" for result in results)
    summary["successful_generation_count"] = sum(
        bool(result.get("response")) and failure_code(result) is None for result in results
    )
    return summary


def citation_metric_comparability(summary: Mapping[str, Any], certified_baseline: Mapping[str, Any]) -> dict[str, str]:
    """Prevent a denominator change from being misreported as a metric regression."""

    expected = certified_baseline["evaluability"]
    return {
        "citation_valid": "COMPARABLE" if summary["citation_evaluable_case_count"] == expected["citation_evaluable_case_count"] else "NOT_COMPARABLE",
        "expected_source_match": "COMPARABLE" if summary["expected_source_evaluable_case_count"] == expected["expected_source_evaluable_case_count"] else "NOT_COMPARABLE",
    }


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
    baseline = json.loads(CERTIFIED_BASELINE.read_text(encoding="utf-8"))
    summary["citation_metric_comparability"] = citation_metric_comparability(summary, baseline)
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
    parser.add_argument("--context-grounding-experiments", action="store_true", help="Run evaluator-only G0-G3 context experiments.")
    parser.add_argument("--grounding-two-pass", action="store_true", help="Run two resumable G0/G4/G5 passes.")
    args = parser.parse_args()
    config = rag_pipeline.RAGConfig()
    runtime = load_runtime(config, stage_timeout_seconds=args.stage_timeout_seconds)
    if args.max_generation_tokens <= 0:
        parser.error("--max-generation-tokens must be greater than zero")
    evaluator_generator = EvaluatorGenerationBudget(ollama, args.max_generation_tokens)
    output_dir = args.output_dir or DEFAULT_RUNS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    benchmark_cases = load_cases(args.benchmark)
    active_cases = benchmark_cases[:args.limit_cases] if args.limit_cases else benchmark_cases
    if args.context_grounding_experiments:
        if args.grounding_two_pass:
            two_pass = run_grounding_two_passes_resumable(
                active_cases, runtime, evaluator_generator, output_dir, resume=args.resume,
                stage_timeout_seconds=args.stage_timeout_seconds,
            )
            report = two_pass["passes"].get("run_2", two_pass["passes"]["run_1"])["report"]
            grounding_rows = _load_grounding_checkpoint(output_dir / ("run_2" if "run_2" in two_pass["passes"] else "run_1"))["rows"]
        else:
            report, grounding_rows = run_grounding_experiments_resumable(
                active_cases, runtime, evaluator_generator, output_dir, resume=args.resume,
                stage_timeout_seconds=args.stage_timeout_seconds,
            )
        results = grounding_rows["G0_control"]
    else:
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
