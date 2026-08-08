"""Stable, runtime-neutral contracts for the Corporate Brain RAG pipeline.

This module is deliberately declarative in Sub-phase 1.1.  It defines the
public data contracts that later phases will use to share production behavior
between the Streamlit application, diagnostics, and evaluation.  It must not
perform retrieval, access ChromaDB, call Ollama, construct prompts, or import
the Streamlit application.

The public API is versioned so internal implementation can evolve without
silently breaking future consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from rank_bm25 import BM25Okapi


__version__ = "1.0"
API_VERSION = "1.0"

__all__ = [
    "__version__",
    "API_VERSION",
    "Metadata",
    "MetadataFilter",
    "ChatMessage",
    "RAGConfig",
    "ChunkRecord",
    "VectorCandidate",
    "BM25Candidate",
    "RRFScore",
    "RetrievalResult",
    "VectorQueryCall",
    "RetrievalPassTrace",
    "HybridSearchResult",
    "PromptSource",
    "PromptResult",
    "CitationResult",
    "QueryRewriteResult",
    "GenerationResult",
    "PipelineTimings",
    "PipelineFailure",
    "PipelineTrace",
    "EvidencePassage",
    "EvidenceExtractionResult",
    "extract_evidence_exhaustive_specific",
    "ExtractiveAnswerResult",
    "EmbeddingEncoder",
    "EmbeddingModelLoadError",
    "VectorStore",
    "TextGenerator",
    "PipelineRuntime",
    "build_bm25_index",
    "normalize_chroma_filter",
    "metadata_matches_filter",
    "hybrid_search",
    "build_source_list",
    "build_context",
    "build_recent_chat_history",
    "build_no_match_message",
    "build_clarification_message",
    "build_production_prompt",
    "rewrite_query",
    "stream_generate",
    "parse_cited_source_ids",
    "detect_no_coverage",
    "select_display_sources",
    "deduplicate_sources_by_path",
    "extract_evidence",
    "build_extractive_answer",
    "resolve_embedding_snapshot",
    "load_embedding_model_offline",
]


Metadata = Mapping[str, Any]
"""Read-only metadata associated with an indexed chunk."""

MetadataFilter = Mapping[str, Any]
"""Read-only Chroma-compatible metadata filter, or an empty mapping."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One conversation message supplied to the production RAG runtime."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class RAGConfig:
    """Immutable defaults matching the current production RAG configuration.

    These values are declarative only in this sub-phase.  No runtime behavior
    consumes them until a later, separately approved extraction phase.
    """

    storage_dir: str = "doc_storage_v2"
    chroma_path: str = "chroma_db_local_v2"
    collection_name: str = "documents"
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    llm_model_name: str = "qwen3:8b"
    vector_candidate_count: int = 10
    bm25_candidate_count: int = 10
    rrf_k: int = 60
    default_top_k: int = 5
    production_top_k: int = 15
    min_results_before_relax: int = 3
    rewrite_temperature: float = 0.0
    generation_temperature: float = 0.2


class EmbeddingModelLoadError(RuntimeError):
    """Raised when the certified embedding model is unavailable offline."""


def resolve_embedding_snapshot(model_name: str, cache_root: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a locally cached Hugging Face snapshot without network access."""
    root = Path(cache_root) if cache_root else Path(
        os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    hub_root = root / "hub"
    repository = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    model_dir = hub_root / f"models--{repository.replace('/', '--')}"
    refs_main = model_dir / "refs" / "main"
    revision = refs_main.read_text(encoding="utf-8").strip() if refs_main.is_file() else ""
    snapshots = model_dir / "snapshots"
    snapshot = snapshots / revision if revision else None
    if snapshot is None or not snapshot.is_dir():
        candidates = sorted((path for path in snapshots.glob("*") if path.is_dir()), reverse=True)
        snapshot = candidates[0] if candidates else None
    required = ("config.json", "modules.json", "tokenizer.json")
    if snapshot is None or any(not (snapshot / name).is_file() for name in required):
        raise EmbeddingModelLoadError(
            f"Cached embedding snapshot missing for '{model_name}'. "
            "Expected a complete local Hugging Face snapshot."
        )
    if not any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise EmbeddingModelLoadError(
            f"Cached embedding weights missing for '{model_name}'."
        )
    return snapshot


def load_embedding_model_offline(config: RAGConfig):
    """Load the certified embedding model from its local snapshot only."""
    snapshot = resolve_embedding_snapshot(config.embedding_model_name)
    previous = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(str(snapshot), local_files_only=True)
            dimension = model.get_sentence_embedding_dimension()
        except Exception as error:
            raise EmbeddingModelLoadError(
                f"Unable to load cached embedding model '{config.embedding_model_name}' offline."
            ) from error
        if dimension != 384:
            raise EmbeddingModelLoadError(
                f"Embedding dimension mismatch for '{config.embedding_model_name}': "
                f"expected 384, got {dimension}."
            )
        return model
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One document chunk and the metadata needed to attribute its source."""

    text: str
    metadata: Metadata = field(default_factory=dict)
    chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class VectorCandidate:
    """A vector-search candidate, preserving its source rank and distance."""

    chunk: ChunkRecord
    rank: int
    distance: float | None = None


@dataclass(frozen=True, slots=True)
class BM25Candidate:
    """A BM25 candidate, preserving its source rank and lexical score."""

    chunk: ChunkRecord
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RRFScore:
    """One fused candidate and the evidence contributing to its RRF score."""

    chunk: ChunkRecord
    rank: int
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Complete retrieval outcome, including the existing relaxed fallback."""

    filtered_chunks: tuple[ChunkRecord, ...] = ()
    fallback_chunks: tuple[ChunkRecord, ...] = ()
    fallback_used: bool = False
    vector_candidates: tuple[VectorCandidate, ...] = ()
    bm25_candidates: tuple[BM25Candidate, ...] = ()
    rrf_scores: tuple[RRFScore, ...] = ()


@dataclass(frozen=True, slots=True)
class VectorQueryCall:
    """Arguments and raw candidate evidence for one Chroma vector query."""

    query_embeddings: tuple[tuple[float, ...], ...]
    n_results: int
    metadata_filter: MetadataFilter | None
    include: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class RetrievalPassTrace:
    """Evidence from one filtered or unfiltered hybrid retrieval pass."""

    vector_query: VectorQueryCall
    vector_candidates: tuple[VectorCandidate, ...] = ()
    bm25_candidates: tuple[BM25Candidate, ...] = ()
    rrf_scores: tuple[RRFScore, ...] = ()
    selected_chunks: tuple[ChunkRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """Shared hybrid-search outcome with a legacy-compatible result adapter."""

    filtered: RetrievalPassTrace
    fallback: RetrievalPassTrace | None = None
    fallback_used: bool = False
    fallback_chunks: tuple[ChunkRecord, ...] = ()

    def as_legacy_tuple(self) -> tuple[list[str], list[Metadata], list[str], list[Metadata], bool]:
        """Return the five values currently returned by ``app.py`` unchanged."""

        return (
            [chunk.text for chunk in self.filtered.selected_chunks],
            [chunk.metadata for chunk in self.filtered.selected_chunks],
            [chunk.text for chunk in self.fallback_chunks],
            [chunk.metadata for chunk in self.fallback_chunks],
            self.fallback_used,
        )


@dataclass(frozen=True, slots=True)
class PromptSource:
    """A numbered source made available to a production prompt."""

    source_id: int
    file_name: str
    location: str
    text: str
    path: str
    relaxed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromptResult:
    """The exact prompt and numbered sources prepared for a generation call."""

    prompt: str
    sources: tuple[PromptSource, ...] = ()
    context: str = ""


@dataclass(frozen=True, slots=True)
class CitationResult:
    """Citation parsing and source-display outcome for one generated response."""

    cited_source_ids: tuple[int, ...] = ()
    display_sources: tuple[PromptSource, ...] = ()
    no_coverage_detected: bool = False
    invalid_source_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    """The production query-rewrite outcome and optional measured duration."""

    query: str = ""
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The model result and generation metadata, independent of any UI stream."""

    response: str = ""
    streamed: bool = False
    error: str | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class PipelineTimings:
    """Optional monotonic durations captured by future runtime implementations."""

    rewrite_ms: float | None = None
    embedding_ms: float | None = None
    vector_retrieval_ms: float | None = None
    bm25_retrieval_ms: float | None = None
    rrf_fusion_ms: float | None = None
    prompt_build_ms: float | None = None
    generation_ms: float | None = None
    total_ms: float | None = None


@dataclass(frozen=True, slots=True)
class PipelineFailure:
    """A structured failure description without prescribing any recovery logic."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    """Versioned, immutable record of a future production pipeline execution.

    In this sub-phase it is a contract only.  Future approved phases may fill
    these fields from one execution of the shared production runtime.
    """

    schema_version: str = API_VERSION
    query: str = ""
    rewritten_query: str = ""
    language: str = "French"
    metadata_filter: MetadataFilter | None = None
    retrieval: RetrievalResult | None = None
    prompt: PromptResult | None = None
    generation: GenerationResult | None = None
    citations: CitationResult | None = None
    timings: PipelineTimings = field(default_factory=PipelineTimings)
    failure: PipelineFailure | None = None


@runtime_checkable
class EmbeddingEncoder(Protocol):
    """Minimal interface required by a future vector-search implementation."""

    def encode(self, texts: str | Sequence[str]) -> Any:
        """Return embeddings in the encoder's native representation."""


@runtime_checkable
class VectorStore(Protocol):
    """Minimal read interface required by a future Chroma-compatible adapter."""

    def get(self, *, include: Sequence[str]) -> Mapping[str, Any]:
        """Return stored documents and metadata without prescribing a backend."""

    def query(self, **kwargs: Any) -> Mapping[str, Any]:
        """Return vector candidates without prescribing a backend client type."""


@runtime_checkable
class TextGenerator(Protocol):
    """Minimal interface required by a future Ollama-compatible adapter."""

    def chat(self, **kwargs: Any) -> Any:
        """Generate a completion in the provider's native representation."""


class PipelineRuntime(Protocol):
    """Future shared-runtime interface consumed by UI, tracer, and evaluator.

    Method declarations are intentionally absent in Sub-phase 1.1 because
    their signatures must be introduced alongside the behavior they expose in
    later approved sub-phases.  The protocol exists now as a stable extension
    point without implementing or importing production behavior.
    """


def build_bm25_index(
    _collection: VectorStore,
    _count: int,
) -> tuple[BM25Okapi | None, list[str] | None, list[Metadata] | None]:
    """Build BM25 exactly as the current production application does.

    The ``_count`` parameter is intentionally unused because the active
    ``app.py`` cache key includes it while construction always reloads all
    documents and metadata from the collection.  Tokenization is deliberately
    kept as ``doc.lower().split()`` and collection order is retained exactly.
    """

    all_data = _collection.get(include=["documents", "metadatas"])

    if not all_data["documents"]:
        return None, None, None

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    tokenized_corpus = [document.lower().split() for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, documents, metadatas


def normalize_chroma_filter(chroma_filter: MetadataFilter | None) -> dict[str, Any] | None:
    """Convert direct multi-field equality filters to Chroma's ``$and`` form.

    The production UI already emits either one equality condition or an
    explicit ``$and`` filter.  Versioned benchmark cases may express the same
    conjunction as a plain multi-key JSON object, which Chroma rejects even
    though the BM25 predicate treats it as an AND.  This helper preserves the
    existing valid forms and gives both retrieval backends one canonical
    representation without changing their matching semantics.
    """

    if chroma_filter is None or chroma_filter == {}:
        return None
    if not isinstance(chroma_filter, Mapping):
        raise TypeError("Chroma metadata filters must be mappings.")

    keys = list(chroma_filter)
    operator_keys = [key for key in keys if isinstance(key, str) and key.startswith("$")]
    if operator_keys:
        if len(keys) != 1 or operator_keys[0] != "$and":
            raise ValueError("Chroma metadata filters must use one operator or equality fields.")
        conditions = chroma_filter["$and"]
        if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)) or not conditions:
            raise ValueError("Chroma $and filters must contain one or more condition mappings.")
        if any(not isinstance(condition, Mapping) or len(condition) != 1 for condition in conditions):
            raise ValueError("Each Chroma $and condition must be a single-field mapping.")
        return {"$and": [dict(condition) for condition in conditions]}

    if len(chroma_filter) == 1:
        return dict(chroma_filter)
    return {"$and": [{key: value} for key, value in chroma_filter.items()]}


def metadata_matches_filter(meta: Metadata, chroma_filter: MetadataFilter | None) -> bool:
    """Apply the current production BM25 metadata-filter predicate unchanged.

    This mirrors the nested single-condition and ``$and`` handling currently
    embedded inside ``app.py``'s hybrid search.  It intentionally supports no
    additional Chroma operators in this sub-phase.
    """

    if not chroma_filter:
        return True

    match = True
    if "$and" in chroma_filter:
        for condition in chroma_filter["$and"]:
            for key, value in condition.items():
                if meta.get(key) != value:
                    match = False
                    break
    else:
        for key, value in chroma_filter.items():
            if meta.get(key) != value:
                match = False
                break

    return match


def hybrid_search(
    query: str,
    collection: VectorStore,
    embedding_model: EmbeddingEncoder,
    bm25: BM25Okapi | None,
    docs: list[str] | None,
    metadatas: list[Metadata] | None,
    chroma_filter: MetadataFilter | None = None,
    top_k: int = 5,
    min_results_before_relax: int = 3,
    vector_candidate_count: int = 10,
    bm25_candidate_count: int = 10,
    fusion_depth: int = 10,
    rrf_k: int = 60,
) -> HybridSearchResult:
    """Run the current vector/BM25/RRF search without changing its behavior.

    This is a direct extraction of the active ``app.py`` orchestration.  In
    particular, Chroma is called with ``n_results=10`` and no ``include``
    argument; BM25 uses positive scores only; RRF keys candidates by document
    text; and the optional fallback removes duplicates by document text.
    """

    def run_once(active_filter: MetadataFilter | None, active_top_k: int) -> RetrievalPassTrace:
        rrf_scores: dict[str, float] = {}
        doc_to_meta: dict[str, Metadata] = {}
        vector_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}

        query_vector = embedding_model.encode(query).tolist()
        vector_query = VectorQueryCall(
            query_embeddings=(tuple(query_vector),),
            n_results=vector_candidate_count,
            metadata_filter=active_filter,
        )
        vec_results = collection.query(
            query_embeddings=[query_vector],
            n_results=vector_candidate_count,
            where=active_filter,
        )

        vector_candidates: list[VectorCandidate] = []
        vector_documents = vec_results.get("documents", [[]]) if vec_results else [[]]
        vector_metadatas = vec_results.get("metadatas", [[]]) if vec_results else [[]]
        vector_ids = vec_results.get("ids", [[]]) if vec_results else [[]]
        vector_distances = vec_results.get("distances", [[]]) if vec_results else [[]]

        if vector_documents and len(vector_documents[0]) > 0:
            for rank, (doc_text, meta) in enumerate(zip(vector_documents[0], vector_metadatas[0])):
                chunk_id = vector_ids[0][rank] if vector_ids and len(vector_ids[0]) > rank else None
                distance = vector_distances[0][rank] if vector_distances and len(vector_distances[0]) > rank else None
                chunk = ChunkRecord(text=doc_text, metadata=meta, chunk_id=chunk_id)
                vector_candidates.append(VectorCandidate(chunk=chunk, rank=rank, distance=distance))
                if rank < fusion_depth:
                    rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + (1 / (rank + 1 + rrf_k))
                doc_to_meta[doc_text] = meta
                vector_ranks[doc_text] = rank

        bm25_candidates: list[BM25Candidate] = []
        if bm25 is not None and docs is not None:
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            sorted_indices = sorted(range(len(bm25_scores)), key=lambda index: bm25_scores[index], reverse=True)

            bm25_count = 0
            for index in sorted_indices:
                if bm25_count >= bm25_candidate_count:
                    break
                if bm25_scores[index] <= 0:
                    break

                meta = metadatas[index]
                if active_filter and not metadata_matches_filter(meta, active_filter):
                    continue

                doc_text = docs[index]
                chunk = ChunkRecord(text=doc_text, metadata=meta)
                bm25_candidates.append(BM25Candidate(chunk=chunk, rank=bm25_count, score=float(bm25_scores[index])))
                if bm25_count < fusion_depth:
                    rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + (1 / (bm25_count + 1 + rrf_k))
                doc_to_meta[doc_text] = meta
                bm25_ranks[doc_text] = bm25_count
                bm25_count += 1

        sorted_documents = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        rrf_trace = tuple(
            RRFScore(
                chunk=ChunkRecord(text=doc_text, metadata=doc_to_meta[doc_text]),
                rank=rank,
                score=score,
                vector_rank=vector_ranks.get(doc_text),
                bm25_rank=bm25_ranks.get(doc_text),
            )
            for rank, (doc_text, score) in enumerate(sorted_documents)
        )
        selected_chunks = tuple(
            ChunkRecord(text=doc_text, metadata=doc_to_meta[doc_text])
            for doc_text, _score in sorted_documents[:active_top_k]
        )

        return RetrievalPassTrace(
            vector_query=vector_query,
            vector_candidates=tuple(vector_candidates),
            bm25_candidates=tuple(bm25_candidates),
            rrf_scores=rrf_trace,
            selected_chunks=selected_chunks,
        )

    normalized_filter = normalize_chroma_filter(chroma_filter)
    filtered = run_once(normalized_filter, top_k)
    fallback: RetrievalPassTrace | None = None
    fallback_chunks: tuple[ChunkRecord, ...] = ()
    fallback_used = False

    if normalized_filter is not None and len(filtered.selected_chunks) < min_results_before_relax:
        fallback_used = True
        fallback = run_once(None, top_k)
        seen = {chunk.text for chunk in filtered.selected_chunks}
        fallback_chunks = tuple(chunk for chunk in fallback.selected_chunks if chunk.text not in seen)

    return HybridSearchResult(
        filtered=filtered,
        fallback=fallback,
        fallback_used=fallback_used,
        fallback_chunks=fallback_chunks,
    )


def build_source_list(
    chunks: Sequence[str],
    metas: Sequence[Metadata],
    storage_dir: str,
    relaxed_flag: bool = False,
    start_id: int = 1,
) -> list[PromptSource]:
    """Build numbered prompt sources exactly as the current app.py helper does."""

    output: list[PromptSource] = []
    for index, (doc_text, meta) in enumerate(zip(chunks, metas)):
        filename = meta.get("source_file", "Fichier source")
        output.append(
            PromptSource(
                source_id=start_id + index,
                file_name=filename,
                location=meta.get("location", "N/A"),
                text=doc_text,
                path=os.path.abspath(os.path.join(storage_dir, filename)),
                relaxed=relaxed_flag,
            )
        )
    return output


def build_context(sources: Sequence[PromptSource]) -> str:
    """Format the exact source-labelled context currently sent to the LLM."""

    context_chunks_formatted: list[str] = []
    for source in sources:
        tag = " (hors des filtres actifs — piste proche)" if source.relaxed else ""
        context_chunks_formatted.append(f"[SOURCE {source.source_id}]{tag}\n{source.text}")
    return "\n---\n".join(context_chunks_formatted)


def build_recent_chat_history(history: Sequence[ChatMessage | Metadata]) -> str:
    """Render recent history using the exact role labels in the active prompt."""

    recent_chat_history = ""
    for message in history[-4:]:
        role = "Utilisateur" if message["role"] == "user" else "Assistant"
        recent_chat_history += f"{role}: {message['content']}\n"
    return recent_chat_history


def build_no_match_message(current_lang: str) -> str:
    """Return the active empty-context response without changing its wording."""

    return (
        "I couldn't find anything close to that in the indexed documents, even outside the current "
        "filters. Could you rephrase your question, try a related keyword, or tell me the department/"
        "topic you're aiming for? That would help me point you in the right direction."
        if current_lang == "English"
        else "Je n'ai rien trouvé de proche dans les documents indexés, même en élargissant la recherche "
             "au-delà des filtres actifs. Peux-tu reformuler ta question, essayer un mot-clé associé, ou "
             "me préciser le service/sujet visé ? Ça m'aiderait à t'orienter."
    )


def build_clarification_message(current_lang: str) -> str:
    """Return the deterministic clarification fallback for a blank generation."""

    return (
        "Could you clarify the application, document, or context you mean?"
        if current_lang == "English"
        else "Pouvez-vous préciser l'application, le document ou le contexte concerné ?"
    )


def build_production_prompt(
    *,
    user_query: str,
    filter_ent: str,
    filter_application: str,
    history: Sequence[ChatMessage | Metadata],
    sources: Sequence[PromptSource],
    current_lang: str,
    was_relaxed: bool,
) -> PromptResult:
    """Construct the current production prompt byte-for-byte from its inputs."""

    context_str = build_context(sources)
    recent_chat_history = build_recent_chat_history(history)
    relaxed_note = (
        "\nNOTE IMPORTANTE : certaines sources ci-dessus sont marquées '(hors des filtres actifs — piste "
        "proche)'. Elles ne correspondent pas exactement aux filtres Zone/Application sélectionnés, mais "
        "peuvent constituer une piste utile à proposer à l'utilisateur (mentionne-le clairement, par "
        "exemple : \"je n'ai rien trouvé pile dans [filtre], mais j'ai trouvé quelque chose de proche dans "
        "[autre catégorie], est-ce que ça pourrait t'intéresser ?\")."
        if was_relaxed else ""
    )
    prompt = f"""Tu es l'assistant technique d'entreprise 'Corporate Brain'. Ton ton est celui d'un collègue serviable qui engage la discussion, pas celui d'un moteur de recherche binaire qui répond juste "trouvé" ou "pas trouvé".

PROJET & FILTRES ACTIFS :
- Zone Géographique (Filiale) : {filter_ent}
- Application : {filter_application}

HISTORIQUE RÉCENT DE LA CONVERSATION :
{recent_chat_history}

CONTEXTE DOCUMENTAIRE :
{context_str}
{relaxed_note}

DERNIÈRE QUESTION DE L'UTILISATEUR :
{user_query}

INSTRUCTIONS :
1. Si l'information exacte demandée est présente dans le CONTEXTE, réponds directement et cite la ou les sources au format [SOURCE X].
2. Si l'information exacte n'est pas présente mais que tu repères des éléments proches, apparentés, ou des synonymes/catégories voisines dans le CONTEXTE (par exemple : l'utilisateur cherche "département IT" et tu trouves des mentions de "Technologie", "Informatique", "Systèmes d'Information", ou un acronyme lié), NE REFUSE PAS SÈCHEMENT. Propose plutôt ces pistes de façon naturelle et conversationnelle, en citant leur source [SOURCE X], et explique en quoi elles pourraient correspondre à la demande.
3. Tu as le droit de faire des rapprochements logiques entre plusieurs fragments du CONTEXTE pour construire ta réponse ou tes suggestions.
4. Termine par une question ouverte ou une invitation à préciser si tu n'es pas sûr à 100% (ex : "Est-ce que ça correspond à ce que tu cherches ?" ou "Veux-tu que je regarde plus précisément du côté de X ?"), afin d'entretenir la discussion plutôt que de la clore abruptement.
5. Utilise uniquement les informations du CONTEXTE DOCUMENTAIRE ci-dessus (pas de connaissances externes/génériques), mais reste ouvert et exploratoire avec ce qui s'y trouve plutôt que strictement littéral.
6. Ne dis "je n'ai rien trouvé" que si le CONTEXTE ne contient vraiment rien qui se rapporche même de loin au sujet de la question — et dans ce cas, propose quand même une piste (reformulation, mot-clé à essayer, ou filtre à changer) plutôt que de t'arrêter là.
7. Ne fais aucune remarque sur ton identité d'IA ou tes limites de date.
8. Réponds dans la même langue que la question de l'utilisateur ({current_lang}).
"""
    return PromptResult(prompt=prompt, sources=tuple(sources), context=context_str)


def rewrite_query(
    user_query: str,
    chat_history: Sequence[ChatMessage | Metadata],
    model_name: str,
    generator: TextGenerator,
    *,
    error_reporter: Callable[[str], None] = print,
    clock: Callable[[], float] = time.perf_counter,
) -> QueryRewriteResult:
    """Apply the active app.py conversational rewrite behavior unchanged."""

    start = clock()
    if not chat_history:
        return QueryRewriteResult(query=user_query, latency_ms=(clock() - start) * 1000)

    recent_history = ""
    for message in chat_history[-3:]:
        role = "Utilisateur" if message["role"] == "user" else "Assistant"
        recent_history += f"{role}: {message['content']}\n"

    prompt_rewrite = f"""Compte tenu de l'historique de conversation suivant et de la dernière question de l'utilisateur, reformule la dernière question pour qu'elle soit totalement AUTONOME et COMPRÉHENSIBLE sans l'historique (remplace les pronoms comme 'it', 'ce terme', 'celui-ci', 'the answer' par les acronymes ou sujets réels abordés précédemment).
Si la question est déjà autonome, renvoie-la exactement à l'identique.
Ne réponds pas à la question, renvoie UNIQUEMENT la question reformulée.

HISTORIQUE :
{recent_history}

QUESTION : {user_query}
QUESTION REFORMULÉE :"""

    try:
        response = generator.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_rewrite}],
            options={"temperature": 0.0},
        )
        reformulated = response["message"]["content"].strip()
        query = reformulated if reformulated else user_query
    except Exception as error:
        error_reporter(f"Ollama Error in contextualize_query: {error}")
        query = user_query

    return QueryRewriteResult(query=query, latency_ms=(clock() - start) * 1000)


def stream_generate(
    prompt: str,
    model_name: str,
    generator: TextGenerator,
    *,
    on_token: Callable[[str], None] | None = None,
    clarification_language: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> GenerationResult:
    """Stream the active production Ollama request without owning UI rendering."""

    start = clock()
    stream = generator.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
        stream=True,
    )
    response = ""
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            response += content
            if on_token:
                on_token(response)
    if not response.strip() and clarification_language:
        response = build_clarification_message(clarification_language)
        if on_token:
            on_token(response)
    return GenerationResult(
        response=response,
        streamed=True,
        latency_ms=(clock() - start) * 1000,
    )


def parse_cited_source_ids(response: str) -> tuple[int, ...]:
    """Parse citation IDs with the current set-based deduplication behavior."""

    return tuple(list(set(int(number) for number in re.findall(r"\[SOURCE (\d+)\]", response))))


def detect_no_coverage(response: str) -> bool:
    """Recognize confirmed no-coverage answers without suppressing qualified answers."""

    response_lower = unicodedata.normalize("NFKC", response).lower().replace("’", "'")
    if re.search(r"\b(?:however|cependant|toutefois|néanmoins)\b", response_lower):
        return False
    no_coverage_patterns = [
        r"\b(?:désolé[,. ]*)?(?:je )?n[' ]ai pas trouv[ée]?(?: d[' ]?informations?)?\b",
        r"\bje n[' ]ai trouv[ée]? aucune mention de .{1,160}\bdans le contexte fourni\b",
        r"\baucun des documents fournis ne contient d[' ]?informations? relatives? à .{1,160}\b",
        r"\bje ne trouve pas.{0,160}\b(document|contexte|source|corpus)\b",
        r"information.*(non couverte|absente|indisponible)",
        r"le contexte( fourni)? ne contient aucune information",
        r"le contexte fourni ne contient pas",
        r"les documents ne contiennent aucune information",
        r"ne trouve pas de r.ponse dans le contexte( fourni)?",
        r"la question pos.e ne trouve pas de r.ponse dans le contexte( fourni)?",
        r"ne trouve pas de r.ponse dans les documents",
        r"\b(?:i cannot|i can't|i did not find|not found|not covered).{0,160}\b(document|context|source|corpus)\b",
    ]
    return any(re.search(pattern, response_lower, flags=re.DOTALL) for pattern in no_coverage_patterns)


def select_display_sources(response: str, sources: Sequence[PromptSource]) -> CitationResult:
    """Reproduce current citation, refusal, and source-selection behavior."""

    cited_source_ids = parse_cited_source_ids(response)
    source_ids = {source.source_id for source in sources}
    invalid_source_ids = tuple(source_id for source_id in cited_source_ids if source_id not in source_ids)
    no_coverage_detected = detect_no_coverage(response) and not cited_source_ids

    if no_coverage_detected:
        display_sources: tuple[PromptSource, ...] = ()
    elif cited_source_ids:
        display_sources = tuple(source for source in sources if source.source_id in cited_source_ids)
    else:
        display_sources = ()

    return CitationResult(
        cited_source_ids=cited_source_ids,
        display_sources=display_sources,
        no_coverage_detected=no_coverage_detected,
        invalid_source_ids=invalid_source_ids,
    )


def deduplicate_sources_by_path(sources: Sequence[PromptSource]) -> tuple[PromptSource, ...]:
    """Keep the first source per path, matching the current Streamlit display."""

    unique_sources: dict[str, PromptSource] = {}
    for source in sources:
        if source.path not in unique_sources:
            unique_sources[source.path] = source
    return tuple(unique_sources.values())


# Deterministic evidence/extractive APIs.  These functions are runtime-neutral:
# they consume an already-built trace and never perform retrieval or generation.
@dataclass(frozen=True, slots=True)
class EvidencePassage:
    """One verbatim passage selected from an existing prompt source."""

    evidence_id: str
    source_id: int
    content_sha256: str
    source_file: str
    location: str
    text: str
    sentence_index: int
    match_score: float
    matched_terms: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id, "source_id": self.source_id,
            "content_sha256": self.content_sha256, "source_file": self.source_file,
            "location": self.location, "text": self.text,
            "sentence_index": self.sentence_index, "match_score": self.match_score,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True, slots=True)
class EvidenceExtractionResult:
    """Stable evidence extraction result with no benchmark dependencies."""

    status: str
    query: str
    language: str | None
    passages: tuple[EvidencePassage, ...]
    supporting_source_ids: tuple[int, ...]
    explicit_evidence: bool
    failure_reason: str | None = None
    match_status: str = "NO_EXPLICIT_EVIDENCE"

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0", "status": self.status, "query": self.query,
            "language": self.language, "passages": [p.to_json() for p in self.passages],
            "supporting_source_ids": list(self.supporting_source_ids),
            "explicit_evidence": self.explicit_evidence, "failure_reason": self.failure_reason,
        }
        if self.match_status != "EXPLICIT_ENTITY_ATTRIBUTE_MATCH":
            payload["match_status"] = self.match_status
        return payload


@dataclass(frozen=True, slots=True)
class ExtractiveAnswerResult:
    """Deterministic answer assembled from exact extracted passage text."""

    status: str
    answer_text: str
    evidence_ids: tuple[str, ...]
    source_ids: tuple[int, ...]
    sources: tuple[dict[str, Any], ...]
    passage_hashes: tuple[str, ...]
    citation_ids: tuple[int, ...]
    latency_ms: float
    unsupported_claim_count: int = 0
    failure_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status, "answer_text": self.answer_text,
            "evidence_ids": list(self.evidence_ids), "source_ids": list(self.source_ids),
            "sources": [dict(source) for source in self.sources],
            "passage_hashes": list(self.passage_hashes), "citation_ids": list(self.citation_ids),
            "latency_ms": self.latency_ms, "unsupported_claim_count": self.unsupported_claim_count,
            "failure_reason": self.failure_reason,
        }


_EVIDENCE_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "�")
_EVIDENCE_APOSTROPHE_VARIANTS = "'`´‘’‛ʼ＇"
_EVIDENCE_STOPWORDS = {
    "the", "and", "for", "what", "where", "which", "how", "who", "when", "why", "many",
    "are", "is", "do", "does", "did", "to", "les", "des", "une", "quel", "quelle", "quels",
    "quelles", "combien", "ou", "est", "sont", "qui", "demande", "question",
}


def _normalize_evidence_text(value: object) -> str:
    text = str(value or "")
    for _ in range(3):
        if not any(marker in text for marker in _EVIDENCE_MOJIBAKE_MARKERS):
            break
        try:
            repaired = text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            break
        if repaired == text:
            break
        text = repaired
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({character: " " for character in _EVIDENCE_APOSTROPHE_VARIANTS}))
    text = "".join(character for character in unicodedata.normalize("NFKD", text) if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text.casefold())).strip()


def _evidence_source_hash(source: PromptSource) -> str:
    return hashlib.sha256(source.text.encode("utf-8")).hexdigest()


def _split_evidence_sentences(text: str) -> tuple[str, ...]:
    pieces = re.split(r"(?<=[.!?ã€‚ï¼ï¼Ÿ])\s+|\n+", text)
    return tuple(piece.strip() for piece in pieces if piece.strip())


def _evidence_match_features(query: str, passage: str) -> tuple[float, tuple[str, ...]]:
    normalized_query = _normalize_evidence_text(query)
    normalized_passage = _normalize_evidence_text(passage)
    query_terms = {term for term in normalized_query.split() if len(term) > 2 and term not in _EVIDENCE_STOPWORDS}
    opening_time_aliases = {
        "open", "opened", "opening", "openings", "hours", "hour", "when",
        "ouvert", "ouverte", "ouverts", "ouverture", "horaires", "horaire", "heure",
        "abierto", "abierta", "horario", "horarios",
    }
    opening_time_intent = bool(set(normalized_query.split()) & opening_time_aliases)
    passage_terms = set(normalized_passage.split())
    matched = set(query_terms & passage_terms)
    for query_term in query_terms:
        if len(query_term) >= 5 and any(
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
    time_patterns = set(re.findall(r"\b\d{1,2}(?:h|:)\d{2}\b", passage, flags=re.IGNORECASE))
    time_intent = any(
        term.startswith("heure") or term.startswith("horaire") or term == "ouverture"
        for term in query_terms
    )
    if (time_intent or opening_time_intent) and time_patterns:
        matched.update(
            term for term in query_terms
            if term.startswith("heure") or term.startswith("horaire") or term == "ouverture"
        )
        matched.update(f"time:{value}" for value in sorted(time_patterns))
    denominator = max(1, len(query_terms) + len(exact_values) + len(query_acronyms))
    score = (len(matched) + len(matched_values) + len(matched_acronyms)) / denominator
    return score, tuple(sorted(matched, key=lambda value: (value.casefold(), value)))


_DIRECT_ENTITY_ALIASES = {
    "inzsmart": ("inzsmart",), "simbox": ("simbox",), "vpn": ("vpn",),
    "mbf": ("mbf", "mach billing format"), "cafeteria": ("cafeteria", "cafeterie", "caf t ria"),
    "ggsn": ("ggsn",), "p2p": ("p2p",), "crbt": ("crbt",),
    "huawei_msc": ("huawei msc", "huawei",),
}
_DIRECT_ATTRIBUTE_ALIASES = {
    "count": ("count", "number", "nombre", "combien", "instances", "total"),
    "duration": ("duration", "duree", "durée", "cache", "age", "maximum age", "maximal"),
    "location": ("where", "located", "location", "situe", "situ", "trouve", "etage", "étage", "floor", "tage"),
    "opening_time": ("open", "opened", "ouverte", "opening", "hours", "hour", "when", "ouverture", "horaires", "heure"),
    "approval": ("approve", "approval", "approuve", "approbation", "manager", "responsibility"),
    "version": ("version", "specification"),
    "parameter": ("parameter", "parametre", "paramètre", "duplicate", "doublon", "identifier"),
    "header": ("header",),
    "trailer": ("trailer",),
    "document": ("document", "file", "described"),
}
_DIRECT_ATTRIBUTE_ALIASES["location"] = _DIRECT_ATTRIBUTE_ALIASES["location"] + ("path", "directory", "repertoire", "share")


def _evidence_entity_attribute_profile(query: str, passage: str) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return named entities and factual attributes present in normalized text."""
    nq, np = _normalize_evidence_text(query), _normalize_evidence_text(passage)
    def has_alias(text: str, alias: str) -> bool:
        pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
        return bool(re.search(pattern, text))
    query_entities = {name for name, aliases in _DIRECT_ENTITY_ALIASES.items() if any(has_alias(nq, alias) for alias in aliases)}
    query_attrs = {name for name, aliases in _DIRECT_ATTRIBUTE_ALIASES.items() if any(has_alias(nq, alias) for alias in aliases)}
    passage_entities = {name for name, aliases in _DIRECT_ENTITY_ALIASES.items() if any(has_alias(np, alias) for alias in aliases)}
    passage_attrs = {name for name, aliases in _DIRECT_ATTRIBUTE_ALIASES.items() if any(has_alias(np, alias) for alias in aliases)}
    if re.search(r"\b\d{1,2}(?:h|:)\d{2}\b", passage, flags=re.IGNORECASE):
        passage_attrs.add("opening_time")
    return query_entities, query_attrs, passage_entities, passage_attrs


def _technical_attribute_value_status(query: str, passage: str) -> str | None:
    """Validate that technical attributes include an explicit value."""
    nq = _normalize_evidence_text(query)
    np = _normalize_evidence_text(passage)
    if re.search(r"\bfichier source\s*:", passage, flags=re.IGNORECASE) or re.search(r"\bpath\s+filename\b", np):
        if any(term in nq for term in ("filename", "pattern", "modele", "mod le de nom", "motif de nom", "format du nom", "version", "server", "hostname", "host")):
            return "METADATA_ONLY_MATCH"
    if "version" in nq or "revision" in nq or "release" in nq:
        if re.search(r"\b(?:v\s*)?\d+(?:\.\d+)+[a-z]?\b", passage, flags=re.IGNORECASE) and not re.search(r"\bfichier source\s*:", passage, flags=re.IGNORECASE) and not re.search(r"table des matieres|table of contents", np):
            return None
        return "ATTRIBUTE_PRESENT_VALUE_MISSING"
    if any(term in nq for term in ("filename pattern", "modele de nom de fichier", "mod le de nom de fichier", "motif de nom de fichier", "format du nom de fichier", "pattern de fichier", "file name pattern", "pattern")):
        if re.search(r"(?:\b[A-Za-z0-9][A-Za-z0-9_.-]*[*?][A-Za-z0-9_.-]*|\^?[A-Za-z0-9_]+\[.*?\]|pattern\s*=)", passage, flags=re.IGNORECASE) or "prefix timestamp" in np:
            return None
        return "ATTRIBUTE_PRESENT_VALUE_MISSING"
    if any(term in nq for term in ("duplicate", "doublon", "dupliqu", "duplication")):
        if any(term in np for term in ("param_check_dup_batch", "duplicate batch check", "crc", "controle de redondance cyclique", "vérification des doublons")) and not re.search(r"table des matieres|table of contents", np):
            return None
        return "HEADING_ONLY_MATCH" if len(np.split()) < 24 or "table des matieres" in np else "ATTRIBUTE_PRESENT_VALUE_MISSING"
    if any(term in nq for term in ("directory", "repertoire", "folder", "path", "chemin", "output files", "output directory", "sortie")):
        if re.search(r"(?:[A-Za-z]:\\|\\\\[A-Za-z0-9_.-]+\\[^\s]+|/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|(?:s?ftp|https?)://[^\s]+)", passage, flags=re.IGNORECASE):
            return None
        return "ATTRIBUTE_PRESENT_VALUE_MISSING"
    if "table" in nq:
        if "table des matieres" in np or "table of contents" in np:
            return "HEADING_ONLY_MATCH"
        if re.search(r"\b(?:[A-Z][A-Z0-9_]{2,}|[A-Za-z0-9_]+\.[A-Za-z0-9_]+)\b", passage):
            return None
        return "HEADING_ONLY_MATCH"
    if any(term in nq for term in ("server", "hostname", "host")):
        if re.search(r"\b(?:[A-Za-z][A-Za-z0-9-]*\.[A-Za-z]{2,}|\d{1,3}(?:\.\d{1,3}){3}|[A-Za-z]+\d{2,})\b", passage):
            return None
        return "ATTRIBUTE_PRESENT_VALUE_MISSING"
    if "protocol" in nq:
        if re.search(r"\b(?:ftp|sftp|http|https|tcp|udp|nfs)\b", np, flags=re.IGNORECASE):
            return None
        return "ATTRIBUTE_PRESENT_VALUE_MISSING"
    if "port" in nq:
        if re.search(r"\bport\s*[:=]?\s*[1-9]\d{1,4}\b|\b(?:sftp|ftp|http|https)\s*[:=]\s*[1-9]\d{1,4}\b", passage, flags=re.IGNORECASE):
            return None
        return "ATTRIBUTE_PRESENT_VALUE_MISSING"
    if any(term in nq for term in ("how often", "frequency", "schedule", "frequence", "fréquence", "a quelle frequence", "tous les combien")):
        frequency_value = re.search(r"\b(?:every|daily|weekly|monthly|minutes?|hours?|quotidien|journaliere|journalière|une fois par jour|tous les jours|cron)\b|\b\d{1,2}:\d{2}\b", np, flags=re.IGNORECASE)
        duration_only = bool(re.search(r"\b\d+\s*(?:jours?|days?|months?|mois)\b", np, flags=re.IGNORECASE)) and not frequency_value
        if frequency_value and not duration_only:
            return None
        return "ATTRIBUTE_PRESENT_VALUE_MISSING"
    return None


def _classify_evidence_query(query: str) -> str:
    """Classify whether a query requests one or multiple explicit facts."""

    tokens = _normalize_evidence_text(query).split()
    interrogatives = {
        "qui", "who", "what", "which", "where", "when", "why", "how", "ou",
        "quel", "quelle", "quels", "quelles", "combien", "o",
    }
    interrogative_count = sum(token in interrogatives for token in tokens)
    if interrogative_count >= 2:
        return "multi_fact"
    if any(token in {"and", "et"} for token in tokens):
        token_set = set(tokens)
        if ({"location", "hours"} <= token_set or {"location", "horaire"} <= token_set
                or {"version", "date"} <= token_set
                or {"filename", "directory"} <= token_set):
            return "multi_fact"
    return "single_fact"


def extract_evidence(trace: PipelineTrace, *, max_passages: int = 3) -> EvidenceExtractionResult:
    """Extract deterministic passages from a certified trace without leakage."""

    query = trace.rewritten_query or trace.query
    if trace.prompt is None or not trace.prompt.sources:
        return EvidenceExtractionResult("NO_EXPLICIT_EVIDENCE", query, trace.language, (), (), False, "no_prompt_sources")
    # A bare document/entity label is not a factual target.  Treat it as
    # genuinely vague so callers can ask for clarification instead of
    # surfacing a filename or other metadata sentence as an answer.
    normalized_query = _normalize_evidence_text(query)
    query_terms = normalized_query.split()
    factual_markers = {
        "version", "specification", "parametre", "parameter", "duplicate",
        "doublon", "dupliqu", "duree", "duration", "age", "maximum",
        "maximal", "combien", "nombre", "count", "how", "many", "where",
        "ou", "etage", "location", "located", "horaire", "horaires",
        "ouverture", "open", "opening", "when", "qui", "who", "approuve",
        "approval", "approve",
    }
    if len(query_terms) <= 2 and not set(query_terms) & factual_markers:
        return EvidenceExtractionResult("NO_EXPLICIT_EVIDENCE", query, trace.language, (), (), False, "vague_query")
    candidates: list[tuple[float, int, int, PromptSource, str, tuple[str, ...]]] = []
    seen: set[tuple[str, str]] = set()
    for source in trace.prompt.sources:
        for sentence_index, passage in enumerate(_split_evidence_sentences(source.text)):
            key = (_evidence_source_hash(source), _normalize_evidence_text(passage))
            if key in seen:
                continue
            seen.add(key)
            score, matched_terms = _evidence_match_features(query, passage)
            technical_request = any(
                term in _normalize_evidence_text(query)
                for term in ("protocol", "port", "hostname", "server", "table", "directory", "repertoire", "filename pattern", "modele de nom de fichier", "mod le de nom de fichier", "frequency", "schedule", "output files", "output directory", "sortie", "author", "wrote", "purge", "value enables", "how are duplicate", "suffix", "archived filename")
            )
            if score == 0 and technical_request and _technical_attribute_value_status(query, passage) is None:
                score, matched_terms = 0.01, ("technical_explicit_value",)
            if source.metadata.get("block_type") and score == 0:
                nq = _normalize_evidence_text(query)
                np = _normalize_evidence_text(passage)
                structured_signal = (
                    ("value enables" in nq and "param_check_dup_batch" in np and re.search(r"\b[yY]\b", passage))
                    or ("who wrote" in nq and any(term in np for term in ("ecrit par", "author", "written by")))
                    or ("purge" in nq and any(term in np for term in ("journaliere", "daily", "every")))
                    or ("bi" in nq and "filedirectory" in np and "system name bi" in np)
                )
                if structured_signal:
                    score, matched_terms = 0.02, ("structured_explicit_value",)
            if score > 0 and matched_terms:
                candidates.append((score, source.source_id, sentence_index, source, passage, matched_terms))
    query_entities, query_attributes, _, _ = _evidence_entity_attribute_profile(query, query)
    query_fact_type = _classify_evidence_query(query)
    normalized_query_for_priority = _normalize_evidence_text(query)
    validated_candidates: list[tuple[float, int, int, PromptSource, str, tuple[str, ...]]] = []
    rejected_statuses: set[str] = set()
    for candidate in candidates:
        score, source_id, sentence_index, source, passage, matched_terms = candidate
        structured = bool(source.metadata.get("block_type"))
        normalized_passage = _normalize_evidence_text(passage)
        if structured:
            if any(term in normalized_query_for_priority for term in ("who wrote", "author", "written by", "auteur", "ecrit par")) and not any(term in normalized_passage for term in ("ecrit par", "author", "written by", "auteur")):
                rejected_statuses.add("ENTITY_ONLY_MATCH")
                continue
            if any(term in normalized_query_for_priority for term in ("who wrote", "author", "written by", "auteur", "ecrit par")):
                validated_candidates.append(candidate)
                continue
            if "value enables" in normalized_query_for_priority or "valeur active" in normalized_query_for_priority:
                if not ("param_check_dup_batch" in normalized_passage and ("valeur" in normalized_passage or "=" in normalized_passage) and re.search(r"\b[yY]\b", passage)):
                    rejected_statuses.add("ATTRIBUTE_PRESENT_VALUE_MISSING")
                    continue
                validated_candidates.append(candidate)
                continue
            if "how are duplicate" in normalized_query_for_priority or "comment les fichiers dupliqu" in normalized_query_for_priority:
                if not ("crc" in normalized_passage or "param_check_dup_batch" in normalized_passage):
                    rejected_statuses.add("ATTRIBUTE_PRESENT_VALUE_MISSING")
                    continue
                validated_candidates.append(candidate)
                continue
            if "purge" in normalized_query_for_priority or "purg" in normalized_query_for_priority:
                if not any(term in normalized_passage for term in ("journaliere", "daily", "every")):
                    rejected_statuses.add("ATTRIBUTE_PRESENT_VALUE_MISSING")
                    continue
                validated_candidates.append(candidate)
                continue
            if "suffix" in normalized_query_for_priority or "archived filename" in normalized_query_for_priority:
                if not re.search(r"\.gz\b", passage, flags=re.IGNORECASE):
                    rejected_statuses.add("ATTRIBUTE_PRESENT_VALUE_MISSING")
                    continue
                validated_candidates.append(candidate)
                continue
            if "bi" in normalized_query_for_priority and any(term in normalized_query_for_priority for term in ("output", "directory", "folder")):
                if not ("system name bi" in normalized_passage and "filedirectory" in normalized_passage):
                    rejected_statuses.add("ENTITY_ONLY_MATCH")
                    continue
                validated_candidates.append(candidate)
                continue
        if "suffix" in normalized_query_for_priority or "archived filename" in normalized_query_for_priority:
            if not re.search(r"\.gz\b", passage, flags=re.IGNORECASE):
                rejected_statuses.add("ATTRIBUTE_PRESENT_VALUE_MISSING")
                continue
        if "purge" in normalized_query_for_priority or "purg" in normalized_query_for_priority:
            if not any(term in normalized_passage for term in ("journaliere", "daily", "every")):
                rejected_statuses.add("ATTRIBUTE_PRESENT_VALUE_MISSING")
                continue
        technical_status = _technical_attribute_value_status(query, passage)
        if technical_status is not None:
            rejected_statuses.add(technical_status)
            continue
        _, _, passage_entities, passage_attributes = _evidence_entity_attribute_profile(query, passage)
        source_sentences = _split_evidence_sentences(source.text)
        adjacent_text = " ".join(
            source_sentences[index]
            for index in (sentence_index - 1, sentence_index + 1)
            if 0 <= index < len(source_sentences)
        )
        heading_context = " ".join(
            source_sentences[index]
            for index in range(max(0, sentence_index - 3), sentence_index)
        )
        adjacent_entities = _evidence_entity_attribute_profile(query, adjacent_text)[2]
        contextual_entities = _evidence_entity_attribute_profile(query, heading_context)[2]
        entity_ok = not query_entities or bool(passage_entities & query_entities) or bool(adjacent_entities & query_entities) or bool(contextual_entities & query_entities)
        attribute_ok = not query_attributes or bool(passage_attributes & query_attributes)
        conflicting_entity = bool(query_entities and passage_entities and not (passage_entities & query_entities))
        opposing_pairs = (
            ({"location"}, {"opening_time"}),
            ({"opening_time"}, {"location"}),
            ({"header"}, {"trailer"}),
        )
        conflicting_attribute = query_fact_type == "single_fact" and any(
            left & query_attributes and right & passage_attributes
            for left, right in opposing_pairs
        )
        if conflicting_entity or conflicting_attribute:
            rejected_statuses.add("CONFLICTING_ENTITY_OR_ATTRIBUTE")
        elif query_entities and not entity_ok:
            rejected_statuses.add("ATTRIBUTE_ONLY_MATCH")
        elif query_attributes and not attribute_ok:
            rejected_statuses.add("ENTITY_ONLY_MATCH")
        else:
            validated_candidates.append(candidate)
    candidates = validated_candidates
    opening_time_query = bool(
        set(_normalize_evidence_text(query).split())
        & {"open", "opened", "opening", "openings", "hours", "hour", "when", "ouvert", "ouverte", "ouverture", "horaires", "horaire", "heure", "abierto", "abierta", "horario", "horarios"}
    )
    normalized_query_for_priority = _normalize_evidence_text(query)

    def attribute_priority(passage_text: str) -> int:
        normalized_passage = _normalize_evidence_text(passage_text)
        if "version" in normalized_query_for_priority and "version" in normalized_passage and re.search(r"\bv?\d+(?:\.\d+)+[a-z]?\b", passage_text, flags=re.IGNORECASE):
            return 3
        if (
            any(term in normalized_query_for_priority for term in ("duree", "duration", "age maximal", "maximum age"))
            or ("maximum" in normalized_query_for_priority and "age" in normalized_query_for_priority)
        ):
            if ("cache" in normalized_passage or "age" in normalized_passage) and re.search(r"\b\d+(?:[.,]\d+)?\s+(?:jours?|heures?|minutes?)\b", normalized_passage):
                return 3
        if any(term in normalized_query_for_priority for term in ("parametre", "parameter")):
            if any(term in normalized_passage for term in ("batch check", "verification", "vérification")):
                return 4
            if any(term in normalized_passage for term in ("duplicate", "doublon", "dupliqu")):
                return 2
        return 0

    def structured_priority(item: tuple[float, int, int, PromptSource, str, tuple[str, ...]]) -> int:
        """Prefer explicit structured fields when metadata is available."""
        source, passage = item[3], _normalize_evidence_text(item[4])
        if not source.metadata.get("block_type"):
            return 0
        if any(term in normalized_query_for_priority for term in ("who wrote", "author", "written by", "auteur", "ecrit par")):
            return 5 if any(term in passage for term in ("ecrit par", "author", "written by", "auteur")) else -5
        if "value enables" in normalized_query_for_priority or "valeur active" in normalized_query_for_priority:
            return 6 if "param_check_dup_batch" in passage and re.search(r"\b[=:]\s*y\b", passage) else -4
        if "how are duplicate" in normalized_query_for_priority or "comment les fichiers dupliqu" in normalized_query_for_priority:
            return 6 if "crc" in passage else (3 if "param_check_dup_batch" in passage else -3)
        if "purge" in normalized_query_for_priority or "purg" in normalized_query_for_priority:
            return 6 if any(term in passage for term in ("journaliere", "daily", "every")) else -5
        if "bi" in normalized_query_for_priority and any(term in normalized_query_for_priority for term in ("output", "directory", "folder")):
            return 7 if "system name = bi" in passage and "filedirectory =" in passage else -6
        return 0

    candidates.sort(
        key=lambda item: (
            -attribute_priority(item[4]),
            -structured_priority(item),
            -(item[0] + (1.0 if opening_time_query and re.search(r"\b\d{1,2}(?:h|:)\d{2}\b", item[4], flags=re.IGNORECASE) else 0.0)),
            -item[0], item[1], item[2], _normalize_evidence_text(item[4]),
        )
    )
    if _classify_evidence_query(query) == "single_fact":
        selected = candidates[:1]
    else:
        selected = []
        covered_terms: set[str] = set()
        covered_attributes: set[str] = set()
        for candidate in candidates:
            if len(selected) >= max(1, max_passages):
                break
            terms = set(candidate[5])
            candidate_attributes = _evidence_entity_attribute_profile(query, candidate[4])[3] & query_attributes
            terms.update(f"attribute:{attribute}" for attribute in candidate_attributes)
            if selected and not terms.difference(covered_terms) and not candidate_attributes.difference(covered_attributes):
                continue
            selected.append(candidate)
            covered_terms.update(terms)
            covered_attributes.update(candidate_attributes)
        selected.sort(key=lambda item: (item[1], item[2], _normalize_evidence_text(item[4])))
    passages = tuple(
        EvidencePassage(f"E{index}", source.source_id, _evidence_source_hash(source), source.file_name, source.location, passage, sentence_index, round(score, 8), matched_terms)
        for index, (score, _source_id, sentence_index, source, passage, matched_terms) in enumerate(selected, 1)
    )
    source_ids = tuple(sorted({passage.source_id for passage in passages}))
    if not passages:
        reason = next(iter(sorted(rejected_statuses)), "no_query_supported_passage")
        return EvidenceExtractionResult("NO_EXPLICIT_EVIDENCE", query, trace.language, (), (), False, reason, reason if reason in {"ATTRIBUTE_ONLY_MATCH", "ENTITY_ONLY_MATCH", "CONFLICTING_ENTITY_OR_ATTRIBUTE"} else "NO_EXPLICIT_EVIDENCE")
    return EvidenceExtractionResult("EVIDENCE_FOUND", query, trace.language, passages, source_ids, True, None, "EXPLICIT_ENTITY_ATTRIBUTE_MATCH")


def _strict_exhaustive_attribute(query: str) -> str | None:
    """Identify one explicit technical attribute for selected-document scans."""
    nq = _normalize_evidence_text(query)
    if any(term in nq for term in ("who wrote", "written by", "author", "writer", "auteur", "ecrit par", "redige par", "redige par")):
        return "author"
    if any(term in nq for term in ("who approved", "approved by", "approver", "approuve par")):
        return "approval"
    if "duplicate" in nq or "doublon" in nq or "dupliqu" in nq:
        if any(term in nq for term in ("parameter", "parametre", "paramètre")):
            return "duplicate_parameter"
        if any(term in nq for term in ("how are", "how do", "mechanism", "detected", "detecte", "controle", "control")):
            return "duplicate_mechanism"
        if any(term in nq for term in ("parameter", "parametre", "value", "valeur")):
            return "duplicate_enable_value"
        return "duplicate_mechanism"
    if "version" in nq or "revision" in nq or "release" in nq:
        return "version"
    if "filename pattern" in nq or "modele de nom" in nq or "pattern de fichier" in nq:
        return "filename_pattern"
    if ("collection frequency" in nq or "frequence de collecte" in nq or "how often" in nq or "a quelle frequence" in nq or "tous les combien" in nq) and "purge" not in nq and "archive" not in nq:
        return "collection_frequency"
    if "distribution frequency" in nq or "frequence de distribution" in nq:
        return "distribution_frequency"
    if "protocol" in nq:
        return "protocol"
    if "port" in nq:
        return "port"
    if "input directory" in nq or "repertoire d entree" in nq or "collection directory" in nq:
        return "input_directory"
    if "output directory" in nq or "output files" in nq or "repertoire de sortie" in nq:
        return "output_directory"
    if "archive directory" in nq or "exact archive" in nq or "dossier d archivage" in nq:
        return "archive_directory"
    if "cdr format" in nq:
        return "cdr_format"
    if "cache age" in nq or "maximum cache" in nq:
        return "cache_age"
    if "compression" in nq:
        return "compression"
    if "retention" in nq:
        return "retention_period"
    if "suffix" in nq or "archived filename" in nq:
        return "archive_suffix"
    if "purge" in nq or ("how often" in nq and "archive" in nq):
        return "archive_purge_frequency"
    if "username" in nq or "user name" in nq:
        return "username"
    if "hostname" in nq or "server" in nq or " host" in f" {nq}":
        return "host"
    if "table" in nq:
        return "table"
    if "count" in nq or "number" in nq or "nombre" in nq:
        return "count"
    return None


def _strict_exhaustive_entities(query: str) -> set[str]:
    nq = _normalize_evidence_text(query)
    entities = {name for name, aliases in _DIRECT_ENTITY_ALIASES.items() if any(alias in nq for alias in aliases)}
    entities.update(name for name in ("dwh", "bi", "ftp_cra", "reqleg", "p2p") if re.search(rf"\b{re.escape(name)}\b", nq))
    return entities


def _strict_value_state(attribute: str, text: str) -> str:
    """Return the explicit-value state used by exhaustive admission."""
    normalized = _normalize_evidence_text(text)
    def labeled_value(pattern: str) -> str:
        match = re.search(pattern + r"\s*[:=]\s*(?:=\s*)?(.+)$", text, re.I)
        return match.group(1).strip() if match else ""
    field_value = re.search(r"(?:=|:)\s*(.*)$", text.strip())
    value = field_value.group(1).strip() if field_value else ""
    if attribute == "author":
        value = labeled_value(r"(?:ecrit par|écrit par|auteur|author|written by|redige par|rédigé par)")
        ok = bool(value)
    elif attribute == "approval":
        value = labeled_value(r"(?:approuve par|approuvé par|approved by)")
        ok = bool(value)
    elif attribute == "duplicate_mechanism":
        ok = bool(re.search(r"crc|cyclic redundancy check|controle de redondance cyclique|contrôle de redondance cyclique|checksum", text, re.I))
    elif attribute == "duplicate_parameter":
        ok = "param_check_dup_batch" in normalized
    elif attribute == "duplicate_enable_value":
        ok = "param_check_dup_batch" in normalized and bool(re.search(r"(?:=|valeur)\s*[«\"']?\s*y\b", text, re.I))
    elif attribute == "version":
        ok = bool(re.search(r"\b(?:v\s*)?\d+(?:\.\d+)+[a-z]?\b", text, re.I))
    elif attribute == "filename_pattern":
        ok = bool(re.search(r"[*?]|\^?[A-Za-z0-9_]+\[.*?\]|prefix\s*\+\s*timestamp|[A-Za-z]+(?:[-_]\w+){2,}", text, re.I))
    elif attribute in {"collection_frequency", "distribution_frequency", "archive_purge_frequency"}:
        ok = bool(re.search(r"daily|every|jours?|quotidien|journaliere|journalière|une fois par jour|\d{1,2}:\d{2}", normalized, re.I))
    elif attribute == "protocol":
        ok = bool(re.search(r"\b(?:sftp|ftp|http|https|tcp|udp|nfs)\b", normalized, re.I))
    elif attribute == "port":
        ok = bool(re.search(r"\bport\s*[:=]?\s*\d{1,5}\b", text, re.I))
    elif attribute in {"input_directory", "output_directory", "archive_directory"}:
        ok = attribute == "archive_directory" and bool(re.search(r"dossier d['’]?archivage|archive directory", normalized)) or bool(re.search(r"(?:[A-Za-z]:\\|\\\\|/(?:[A-Za-z0-9_.-]+/)+|(?:s?ftp|https?)://|to be defined)", text, re.I))
    elif attribute == "cdr_format":
        ok = bool(re.search(r"\b(?:brut|raw)\b", normalized, re.I))
    elif attribute == "cache_age" or attribute == "retention_period":
        ok = bool(re.search(r"\b\d+\s*(?:jours?|days?|mois|months?)\b", normalized, re.I))
    elif attribute == "compression":
        ok = bool(re.search(r"\b(?:gzip|zip|\.gz)\b", normalized, re.I))
    elif attribute == "archive_suffix":
        ok = bool(re.search(r"\.gz\b", text, re.I))
    elif attribute == "username":
        ok = bool(re.search(r"\b(?:username|user name)\s+(?!redacted)\S", normalized, re.I))
    elif attribute == "host":
        ok = bool(re.search(r"\b(?:host|hostname|server)\s*[:=]\s*(?:\d{1,3}(?:\.\d{1,3}){3}|\S+)", text, re.I))
    elif attribute == "table":
        ok = bool(re.search(r"\b[A-Z][A-Z0-9_]{2,}\b", text))
    elif attribute == "count":
        ok = bool(re.search(r"\b\d+\b", text))
    else:
        ok = bool(value)
    if not value and attribute in {"author", "approval", "archive_directory"}:
        return "EMPTY_VALUE" if attribute == "approval" else "ATTRIBUTE_WITHOUT_VALUE"
    if ok:
        return "EXPLICIT_PLACEHOLDER" if "to be defined" in normalized else "EXPLICIT_VALUE"
    return "ATTRIBUTE_WITHOUT_VALUE"


def extract_evidence_exhaustive_specific(
    query: str,
    chunks: Sequence[ChunkRecord],
    *,
    max_passages: int = 3,
) -> EvidenceExtractionResult:
    """Strict exhaustive evidence scan for Direct Answer/specific-document diagnostics.

    This API deliberately performs no retrieval and is not used by AI Answer or
    all-document execution. Candidates must satisfy the requested attribute,
    entity relation, and explicit-value rules before ranking.
    """
    attribute = _strict_exhaustive_attribute(query)
    if attribute is None:
        return EvidenceExtractionResult("NO_EXPLICIT_EVIDENCE", query, None, (), (), False, "NO_MATCH")
    nq = _normalize_evidence_text(query)
    entities = _strict_exhaustive_entities(query)
    admitted: list[tuple[int, float, ChunkRecord, str, tuple[str, ...]]] = []
    for index, chunk in enumerate(chunks):
        text = chunk.text
        normalized = _normalize_evidence_text(text)
        metadata = chunk.metadata
        if metadata.get("block_type") == "heading" or re.search(r"table of contents|table des matieres", normalized):
            continue
        state = _strict_value_state(attribute, text)
        if state not in {"EXPLICIT_VALUE", "EXPLICIT_PLACEHOLDER"}:
            continue
        labels = {
            "filename_pattern": ("filename pattern", "modele de nom", "nom de fichier"),
            "collection_frequency": ("collection frequency", "frequence de collecte"),
            "distribution_frequency": ("distribution frequency", "frequence de distribution"),
            "protocol": ("connection protocol", "protocol"),
            "input_directory": ("directory", "repertoire", "filedirectory"),
            "output_directory": ("filedirectory", "output directory", "repertoire de sortie"),
            "archive_directory": ("dossier d", "archive directory"),
            "cdr_format": ("cdr format",), "cache_age": ("cache", "age maximal"),
            "compression": ("compress", "gzip"), "retention_period": ("retention", "retention"),
            "archive_suffix": ("archivage", ".gz"), "archive_purge_frequency": ("purge",),
            "table": ("mz_param",), "duplicate_parameter": ("param_check_dup_batch",),
            "duplicate_enable_value": ("param_check_dup_batch",), "username": ("username", "user name"),
            "host": ("host", "hostname", "server"), "port": ("port",),
        }
        if attribute in labels and not any(label in normalized for label in labels[attribute]):
            continue
        if attribute == "author" and not re.search(r"(?:ecrit par|écrit par|auteur|author|written by|redige par|rédigé par)", text, re.I):
            continue
        if attribute == "approval" and not re.search(r"(?:approuve par|approved by)", text, re.I):
            continue
        if attribute == "archive_directory" and not re.search(r"dossier d\s*['’]?\s*archivage|archive directory", normalized):
            continue
        if entities:
            context_text = " ".join(str(metadata.get(key, "")) for key in ("source_file", "section", "location"))
            entity_context = normalized + " " + _normalize_evidence_text(context_text)
            entity_hit = any(
                re.search(rf"\b{re.escape(entity)}\b", entity_context)
                or re.search(rf"\b{re.escape(entity.replace('_', ' '))}\b", entity_context)
                for entity in entities
            )
            if not entity_hit and metadata.get("block_type") in {"table_row", "row_record", "column_record"}:
                continue
        collection_port_context = (
            attribute == "port"
            and "p2p" in nq
            and "collection" in nq
            and ("connection protocol" in normalized or "sftp" in normalized)
        )
        if attribute == "port" and entities and not any(entity in normalized for entity in entities) and not collection_port_context:
            continue
        if attribute == "username" and entities and not any(entity in normalized for entity in entities):
            continue
        score = 10.0 + (3.0 if metadata.get("block_type") in {"table_row", "row_record", "column_record"} else 0.0)
        if attribute in {"author", "approval", "archive_directory"}:
            score += 5.0
        if attribute == "table" and "mz_param" in normalized:
            score += 10.0
        if attribute == "duplicate_parameter" and "param_check_dup_batch" in normalized:
            score += 10.0
        if attribute == "duplicate_enable_value" and "param_check_dup_batch" in normalized:
            score += 10.0
        if attribute == "username" and entities:
            score += 5.0
        if attribute == "archive_purge_frequency" and "purge" in normalized:
            score += 10.0
        if attribute == "duplicate_mechanism" and re.search(r"crc|cyclic redundancy check|redondance cyclique", text, re.I):
            score += 5.0
        source = PromptSource(index + 1, str(metadata.get("source_file", "")), str(metadata.get("location", "")), text, str(metadata.get("source_file", "")), False, dict(metadata))
        matched = (attribute,) + tuple(sorted(entities))
        admitted.append((index, score, chunk, text, matched))
    admitted.sort(key=lambda item: (-item[1], item[0], _normalize_evidence_text(item[3])))
    selected = admitted[:1] if attribute not in {"location", "multi_fact"} else admitted[:max(1, max_passages)]
    passages: list[EvidencePassage] = []
    for ordinal, (index, score, chunk, text, matched) in enumerate(selected, 1):
        source = PromptSource(index + 1, str(chunk.metadata.get("source_file", "")), str(chunk.metadata.get("location", "")), text, str(chunk.metadata.get("source_file", "")), False, dict(chunk.metadata))
        passages.append(EvidencePassage(f"E{ordinal}", index + 1, _evidence_source_hash(source), source.file_name, source.location, text, 0, score, matched))
    if not passages:
        return EvidenceExtractionResult("NO_EXPLICIT_EVIDENCE", query, None, (), (), False, "NO_MATCH")
    return EvidenceExtractionResult("EVIDENCE_FOUND", query, None, tuple(passages), tuple(p.source_id for p in passages), True, None, "EXPLICIT_ENTITY_ATTRIBUTE_MATCH")


def build_extractive_answer(evidence: EvidenceExtractionResult, language: str | None = None) -> ExtractiveAnswerResult:
    """Build an exact-text answer with deterministic source citations."""

    started = time.perf_counter()
    if evidence.status != "EVIDENCE_FOUND" or not evidence.passages or (
        evidence.match_status and evidence.match_status != "EXPLICIT_ENTITY_ATTRIBUTE_MATCH"
    ):
        return ExtractiveAnswerResult("NO_EXPLICIT_EVIDENCE", build_clarification_message(language or "French"), (), (), (), (), (), (time.perf_counter() - started) * 1000, failure_reason=evidence.failure_reason)
    answer_parts: list[str] = []
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for passage in evidence.passages:
        key = (passage.source_id, passage.text)
        if key in seen:
            continue
        seen.add(key)
        answer_parts.append(f"{passage.text} [SOURCE {passage.source_id}]")
        records.append({"source_id": passage.source_id, "source_file": passage.source_file, "location": passage.location, "content_sha256": passage.content_sha256, "evidence_id": passage.evidence_id})
    source_ids = tuple(record["source_id"] for record in records)
    return ExtractiveAnswerResult("ANSWER", "\n\n".join(answer_parts), tuple(record["evidence_id"] for record in records), source_ids, tuple(records), tuple(record["content_sha256"] for record in records), source_ids, (time.perf_counter() - started) * 1000)
