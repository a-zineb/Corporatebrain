"""Small, deterministic product services shared by the Streamlit UI.

Nothing in this module calls an LLM.  Documents are prepared once per content
hash and canonical schema version, then reused by Direct Answer and Find Me.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
import json
import re
import statistics
import time
from typing import Iterable

import canonical_rag
from document_normalizer import CANONICAL_SCHEMA_VERSION, CanonicalBlock, CanonicalDocument


def normalize_text(value: str) -> str:
    import unicodedata

    folded = unicodedata.normalize("NFKD", str(value or ""))
    return " ".join(re.findall(r"[a-z0-9_.:/-]+", "".join(
        char for char in folded if not unicodedata.combining(char)
    ).casefold()))


def detect_query_language(query: str) -> str:
    """Detect French/English for the current message, never from chat history."""
    normalized = normalize_text(query)
    tokens = set(normalized.split())
    french = {"quel", "quelle", "quels", "quelles", "ou", "est", "sont", "dans",
              "avec", "des", "les", "une", "trouver", "repertoire", "fichier",
              "fichiers", "historique", "modifications", "glossaire", "archivage"}
    english = {"what", "which", "where", "is", "are", "the", "with", "find",
               "file", "files", "history", "glossary", "archive", "directory"}
    fr_score = len(tokens & french)
    en_score = len(tokens & english)
    if re.search(r"[éèêëàâçîïôùûüœ]", str(query).casefold()):
        fr_score += 2
    return "French" if fr_score > en_score else "English"


@dataclass(frozen=True, slots=True)
class SourceTarget:
    source_file: str
    file_hash: str
    block_id: str
    file_type: str
    page: int | None = None
    page_end: int | None = None
    sheet: str | None = None
    section: str | None = None
    table_index: int | None = None
    row_index: int | None = None
    row_end: int | None = None
    cell_range: str | None = None
    paragraph_index: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    evidence_text: str = ""

    @classmethod
    def from_block(cls, block: CanonicalBlock, file_type: str = "") -> "SourceTarget":
        raw_bbox = block.metadata.get("bbox")
        bbox = tuple(float(value) for value in raw_bbox) if raw_bbox and len(raw_bbox) == 4 else None
        return cls(
            block.source_file, block.file_hash, block.block_id, file_type,
            block.page, block.metadata.get("page_end"), block.sheet, block.section,
            block.table_index, block.row_index, block.metadata.get("row_end"),
            block.metadata.get("cell_range"), block.paragraph_index, bbox, block.text,
        )

    @property
    def location_label(self) -> str:
        parts: list[str] = []
        if self.page is not None:
            parts.append(f"Page {self.page}")
        if self.sheet:
            parts.append(f"Sheet: {self.sheet}")
        if self.row_index is not None:
            parts.append(f"Row: {self.row_index}")
        if self.cell_range:
            parts.append(f"Cells: {self.cell_range}")
        if self.section:
            parts.append(self.section)
        return " · ".join(parts) or "Document"


@dataclass(frozen=True, slots=True)
class GlobalSearchHit:
    document: CanonicalDocument
    target: SourceTarget
    score: float
    matched_topic: str
    relation: str = ""
    entity: str = ""
    display_title: str = ""
    display_value: str = ""
    preview: str = ""
    match_count: int = 1


@dataclass(frozen=True, slots=True)
class PreparationResult:
    state: str
    document: CanonicalDocument | None
    warnings: tuple[str, ...] = ()
    cached: bool = False


class PreparedDocumentRegistry:
    """In-memory preparation registry keyed by content hash + schema version."""

    def __init__(self) -> None:
        self._documents: dict[str, CanonicalDocument] = {}
        self._states: dict[str, PreparationResult] = {}
        self._engine = canonical_rag.FastDirectAnswerEngine()
        self._navigation: dict[tuple[str, str], SourceTarget] = {}

    @staticmethod
    def cache_key(data: bytes) -> str:
        return f"{sha256(data).hexdigest()}:{CANONICAL_SCHEMA_VERSION}"

    def prepare(self, data: bytes, source_file: str) -> PreparationResult:
        key = self.cache_key(data)
        cached = self._states.get(key)
        if cached is not None:
            return PreparationResult(cached.state, cached.document, cached.warnings, True)
        outcome = canonical_rag.normalize_with_gate(data, source_file)
        if outcome.document is None:
            result = PreparationResult("FAILED", None, outcome.warnings)
        else:
            health = canonical_rag.document_health(outcome.document)
            if health.status == "FAIL":
                result = PreparationResult("FAILED", outcome.document, health.reasons)
            else:
                self._engine.prepare(outcome.document)
                state = "READY_WITH_WARNINGS" if health.status == "WARN" else "READY"
                result = PreparationResult(state, outcome.document, health.reasons)
                self._documents[outcome.document.file_hash] = outcome.document
                self._index_navigation(outcome.document)
        self._states[key] = result
        return result

    def add(self, document: CanonicalDocument) -> None:
        self._engine.prepare(document)
        self._documents[document.file_hash] = document
        self._index_navigation(document)

    def _index_navigation(self, document: CanonicalDocument) -> None:
        for block in document.blocks:
            self._navigation[(document.file_hash, block.block_id)] = SourceTarget.from_block(
                block, document.file_type
            )

    def source_target(self, file_hash: str, block_id: str) -> SourceTarget | None:
        return self._navigation.get((file_hash, block_id))

    @property
    def documents(self) -> tuple[CanonicalDocument, ...]:
        return tuple(sorted(self._documents.values(), key=lambda item: item.source_file.casefold()))

    def global_search(self, query: str, limit: int = 50) -> tuple[GlobalSearchHit, ...]:
        query_tokens = set(normalize_text(query).split())
        if not query_tokens:
            return ()
        hits: list[GlobalSearchHit] = []
        for document in self.documents:
            index = self._engine.prepare(document)
            for ordinal, block in enumerate(document.blocks):
                block_tokens = set(index.tokenized_blocks[ordinal])
                section_tokens = set(normalize_text(block.section or "").split())
                overlap = query_tokens & (block_tokens | section_tokens)
                if not overlap:
                    continue
                coverage = len(overlap) / len(query_tokens)
                structure_boost = 0.25 if block.block_type in {"key_value", "table_row"} else 0.0
                phrase_boost = 0.5 if normalize_text(query) in normalize_text(block.text) else 0.0
                score = coverage + structure_boost + phrase_boost
                relation, entity, title, value, preview = _humanize_block(block, query_tokens)
                hits.append(GlobalSearchHit(
                    document, self._navigation[(document.file_hash, block.block_id)], score,
                    block.section or next(iter(sorted(overlap))), relation, entity,
                    title, value, preview,
                ))
        hits.sort(key=lambda item: (-item.score, item.document.source_file.casefold(), item.target.block_id))
        deduped: list[GlobalSearchHit] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            identity = (hit.document.file_hash, normalize_text(hit.target.evidence_text))
            if identity not in seen:
                seen.add(identity)
                deduped.append(hit)
        return tuple(deduped[:limit])


def _humanize_block(block: CanonicalBlock, query_tokens: set[str]) -> tuple[str, str, str, str, str]:
    """Turn an internal row representation into a compact user-facing card."""
    pairs = [(left.strip(), right.strip()) for left, right in re.findall(
        r"(?:^|\|)\s*([^=|:]+?)\s*(?:=|:)\s*([^|]+)", block.text
    )]
    meaningful = [(left, right) for left, right in pairs if not re.fullmatch(r"Column\s+\d+", left, re.I)]
    considered = meaningful or pairs
    matched = next(((left, right) for left, right in considered
                    if query_tokens & set(normalize_text(left).split())), None)
    identity_labels = {"system", "system name", "application", "entity", "destination", "workflow", "name"}
    identity = next((right for left, right in considered if normalize_text(left) in identity_labels), "")
    if matched:
        relation, value = matched
    elif considered:
        relation, value = considered[-1]
    else:
        relation, value = block.section or "Match", block.text
    title_parts = [part for part in (identity, block.section or block.sheet) if part]
    title = " · ".join(dict.fromkeys(title_parts)) or relation
    if not meaningful and pairs:
        relation = re.sub(r"(?i)^Column\s+(\d+)$", r"Value from column \1", relation)
    preview_pairs = meaningful[:3]
    preview = " · ".join(f"{left}: {right}" for left, right in preview_pairs)
    if not preview:
        preview = re.sub(r"(?i)Column\s+(\d+)\s*=", r"Column \1:", block.text)
    return relation, identity, title, value, preview[:240]


def suggestion_candidates(query: str, document: CanonicalDocument, limit: int = 3) -> tuple[str, ...]:
    """Suggest only labels/titles that are explicitly present in the document."""
    query_norm = normalize_text(query)
    candidates: set[str] = set()
    for block in document.blocks:
        if block.section:
            candidates.add(block.section.strip())
        for label in re.findall(r"(?:^|\|)\s*([^|=:\n]{2,60})\s*[:=]", block.text):
            candidates.add(label.strip())
    scored: list[tuple[float, str]] = []
    query_tokens = set(query_norm.split())
    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        if not candidate_norm:
            continue
        ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()
        token_score = len(query_tokens & set(candidate_norm.split())) / max(len(query_tokens), 1)
        score = max(ratio, token_score)
        if score >= 0.42:
            scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1].casefold()))
    return tuple(value for _, value in scored[:limit])


@dataclass
class LocalMetrics:
    latencies_ms: list[float] = field(default_factory=list)
    method_counts: dict[str, int] = field(default_factory=dict)
    no_evidence_count: int = 0
    multi_value_count: int = 0
    preparation_failures: int = 0

    def record_answer(self, result: canonical_rag.AnswerResult, latency_ms: float) -> None:
        self.latencies_ms.append(float(latency_ms))
        self.method_counts[result.method] = self.method_counts.get(result.method, 0) + 1
        self.no_evidence_count += int(result.status == "NO_EVIDENCE")
        self.multi_value_count += int(result.result_type in {"MULTI_VALUE", "MULTI_MENTION"})

    def snapshot(self) -> dict[str, object]:
        ordered = sorted(self.latencies_ms)
        percentile = lambda fraction: ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)] if ordered else 0.0
        return {
            "direct_p50_ms": round(statistics.median(ordered), 2) if ordered else 0.0,
            "direct_p95_ms": round(percentile(0.95), 2),
            "method_counts": dict(self.method_counts),
            "no_evidence_count": self.no_evidence_count,
            "multi_value_count": self.multi_value_count,
            "preparation_failures": self.preparation_failures,
        }
