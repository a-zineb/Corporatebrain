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
import os
import re
import time
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
    "EmbeddingEncoder",
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
    "build_production_prompt",
    "rewrite_query",
    "stream_generate",
    "parse_cited_source_ids",
    "detect_no_coverage",
    "select_display_sources",
    "deduplicate_sources_by_path",
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
    return GenerationResult(
        response=response,
        streamed=True,
        latency_ms=(clock() - start) * 1000,
    )


def parse_cited_source_ids(response: str) -> tuple[int, ...]:
    """Parse citation IDs with the current set-based deduplication behavior."""

    return tuple(list(set(int(number) for number in re.findall(r"\[SOURCE (\d+)\]", response))))


def detect_no_coverage(response: str) -> bool:
    """Apply the active app.py no-documentary-answer regex patterns unchanged."""

    response_lower = response.lower()
    no_coverage_patterns = [
        r"je ne trouve pas.*(document|contexte|source|corpus)",
        r"n.est pas.*(document|contexte|source|corpus)",
        r"information.*(non couverte|absente|indisponible)",
        r"le contexte( fourni)? ne contient aucune information",
        r"le contexte fourni ne contient pas",
        r"les documents ne contiennent aucune information",
        r"ne trouve pas de r.ponse dans le contexte( fourni)?",
        r"la question pos.e ne trouve pas de r.ponse dans le contexte( fourni)?",
        r"ne trouve pas de r.ponse dans les documents",
        r"n.est pas mentionn. dans les documents",
        r"n.est pas abord. dans les documents",
        r"l.information n.est pas pr.sente dans les documents",
        r"(i cannot|i can.t|not found|not covered).*(document|context|source|corpus)",
        r"(not mentioned|not available).*(document|context|source|corpus)",
    ]
    return any(re.search(pattern, response_lower, flags=re.DOTALL) for pattern in no_coverage_patterns)


def select_display_sources(response: str, sources: Sequence[PromptSource]) -> CitationResult:
    """Reproduce current citation, refusal, and source-selection behavior."""

    cited_source_ids = parse_cited_source_ids(response)
    source_ids = {source.source_id for source in sources}
    invalid_source_ids = tuple(source_id for source_id in cited_source_ids if source_id not in source_ids)
    no_coverage_detected = detect_no_coverage(response)

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
