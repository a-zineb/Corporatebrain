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
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

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
    "PromptSource",
    "PromptResult",
    "CitationResult",
    "GenerationResult",
    "PipelineTimings",
    "PipelineFailure",
    "PipelineTrace",
    "EmbeddingEncoder",
    "VectorStore",
    "TextGenerator",
    "PipelineRuntime",
    "build_bm25_index",
    "metadata_matches_filter",
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
class PromptSource:
    """A numbered source made available to a production prompt."""

    source_id: int
    file_name: str
    location: str
    text: str
    path: str
    relaxed: bool = False


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


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The model result and generation metadata, independent of any UI stream."""

    response: str = ""
    streamed: bool = False
    error: str | None = None


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
