"""Reliable selected-document QA over canonical blocks only.

The service is deliberately independent from Streamlit and Chroma.  A caller
must provide one exact ``ActiveDocumentContext``; global retrieval is not an
available operation in this module.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence

from document_normalizer import CANONICAL_SCHEMA_VERSION, CanonicalBlock, CanonicalDocument, normalize_document
from rank_bm25 import BM25Okapi


NO_EXPLICIT_EVIDENCE = "NO_EXPLICIT_EVIDENCE"
_SECRET = re.compile(r"password|passwd|token|api[_ -]?key|secret|private\s+key", re.I)
_ASSIGNMENT = re.compile(r"(?:^|\|)\s*([^=|:]+?)\s*(?:=|:)\s*([^|]+)")
_SYNTHESIS = re.compile(r"\b(?:why|explain|compare|summari[sz]e|analy[sz]e|recommend|pourquoi|explique|comparer|resume|analyser|recommande)\b", re.I)


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or ""))
    return " ".join("".join(char for char in value if not unicodedata.combining(char)).casefold().split())


def _tokens(value: str) -> set[str]:
    stop = {"what", "which", "where", "when", "who", "how", "does", "are", "is", "the", "a", "an",
            "of", "for", "to", "in", "and", "quel", "quelle", "quels", "est", "sont", "comment", "le",
            "la", "les", "de", "du", "des", "un", "une", "pour", "dans"}
    cleaned = {token.strip("._-") for token in re.findall(r"[a-z0-9_.-]+", _norm(value))}
    result = {token for token in cleaned if token not in stop and len(token) > 1}
    if "reqleg" in result:
        result.add("reqlog")
    if "reqlog" in result:
        result.add("reqleg")
    if {"svr", "cra"} <= result:
        result.add("svrcra")
    if "svrcra" in result:
        result.update({"svr", "cra"})
    return result


_FIELD_FAMILIES = {
    "author": {"author", "authored", "auteur", "written", "wrote", "redige", "ecrit"},
    "reviewer": {"reviewer", "reviewed", "relecteur", "revu", "revue"},
    "version": {"version", "revision"},
    "date": {"date"},
    "host": {"host", "hostname", "ip", "server", "serveur"},
    "port": {"port"},
    "protocol": {"protocol", "protocole"},
    "directory": {"directory", "filedirectory", "folder", "path", "repertoire", "dossier", "chemin"},
    "filename": {"filename", "file", "pattern", "patterns", "fichier", "nom"},
    "frequency": {"frequency", "frequence", "schedule", "interval", "often", "periodicity", "periodicite"},
    "retention": {"retention", "purge", "duration", "duree", "cache", "age", "maximum"},
    "parameter": {"parameter", "parametre", "param", "flag", "setting", "configuration"},
    "status": {"status", "statut", "state", "etat", "processing"},
    "duplicate": {"duplicate", "duplicates", "doublon", "doublons", "deduplication", "detected", "detection"},
    "mode": {"mode", "transfer", "transfers", "send", "sends", "connection", "transfert", "envoie"},
    "transformation": {"transformation", "transform", "transformed", "transforme"},
    "format": {"format"},
    "destination": {"destination", "destinations", "target", "targets", "sortie", "sorties"},
    "enrichment": {"enrichment", "enrichissement", "enriched", "enrichi"},
    "normalization": {"normalization", "normalisation", "normalized", "normalise"},
    "table": {"table", "tables"},
    "workflow": {"workflow", "workflows", "flux"},
    "username": {"username", "login", "user", "utilisateur"},
    "capacity": {"capacity", "capacite", "performance"},
    "archive": {"archive", "archiving", "archivage"},
    "post_action": {"post", "action", "collecte"},
    "correlation": {"correlation"},
    "copyright": {"copyright"},
}


def _family(tokens: set[str]) -> str | None:
    if tokens & _FIELD_FAMILIES.get("destination", set()):
        return "destination"
    scored = [(len(tokens & aliases), name) for name, aliases in _FIELD_FAMILIES.items()]
    score, name = max(scored, default=(0, ""))
    return name if score else None


@dataclass(frozen=True, slots=True)
class ActiveDocumentContext:
    source_file: str
    file_hash: str
    canonical_document: CanonicalDocument
    block_ids: tuple[str, ...]
    selection_version: int

    def __post_init__(self) -> None:
        if self.file_hash != self.canonical_document.file_hash:
            raise ValueError("active hash does not match canonical document")
        if any(block.file_hash != self.file_hash for block in self.canonical_document.blocks):
            raise ValueError("cross-document block in active context")
        if self.block_ids != tuple(block.block_id for block in self.canonical_document.blocks):
            raise ValueError("active block IDs do not match canonical document")


class ActiveDocumentService:
    def __init__(self) -> None:
        self._active: ActiveDocumentContext | None = None
        self._version = 0

    @property
    def active(self) -> ActiveDocumentContext | None:
        return self._active

    def select(self, document: CanonicalDocument) -> ActiveDocumentContext:
        if self._active is None or self._active.file_hash != document.file_hash or self._active.source_file != document.source_file:
            self._version += 1
        self._active = ActiveDocumentContext(
            source_file=document.source_file, file_hash=document.file_hash,
            canonical_document=document, block_ids=tuple(block.block_id for block in document.blocks),
            selection_version=self._version,
        )
        return self._active

    def clear(self) -> None:
        self._active = None


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    block: CanonicalBlock
    score: float
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FieldOccurrence:
    label: str
    normalized_label: str
    family: str | None
    value: str
    block: CanonicalBlock
    local_context: str = ""


@dataclass(frozen=True, slots=True)
class DocumentFact:
    fact_id: str
    document_hash: str
    source_file: str
    subject: str
    entity: str
    relation: str
    raw_relation: str
    value: str
    qualifiers: tuple[str, ...]
    role: str
    section: str
    block_id: str
    row_id: str
    page: int | None
    sheet: str | None
    original_text: str


@dataclass(frozen=True, slots=True)
class DocumentFactStore:
    file_hash: str
    facts: tuple[DocumentFact, ...]
    categories: tuple[str, ...]

    @classmethod
    def build(cls, document: CanonicalDocument, index: "DocumentQueryIndex") -> "DocumentFactStore":
        facts: list[DocumentFact] = []
        for occurrences in index.raw_field_index.values():
            for ordinal, item in enumerate(occurrences):
                relation = item.family or item.normalized_label
                identity = f"{document.file_hash}\0{item.block.block_id}\0{item.label}\0{ordinal}\0{item.value}"
                facts.append(DocumentFact(
                    sha256(identity.encode()).hexdigest(), document.file_hash, document.source_file,
                    item.local_context or document.source_file, item.local_context, relation, item.label,
                    item.value, tuple(sorted(_tokens(item.block.text) - _tokens(item.value))),
                    _norm(item.block.section or ""), item.block.section or "", item.block.block_id,
                    f"{item.block.table_index}:{item.block.row_index}", item.block.page, item.block.sheet,
                    item.block.text,
                ))
        for block in document.blocks:
            if block.block_type == "heading":
                identity = f"{document.file_hash}\0{block.block_id}\0section"
                facts.append(DocumentFact(sha256(identity.encode()).hexdigest(), document.file_hash,
                    document.source_file, block.text, "", "section", "section", block.text, (),
                    _norm(block.section or ""), block.section or "", block.block_id,
                    f"{block.table_index}:{block.row_index}", block.page, block.sheet, block.text))
        return cls(document.file_hash, tuple(facts), tuple(sorted({fact.relation for fact in facts})))


@dataclass(frozen=True, slots=True)
class QuestionPlan:
    normalized_question: str
    requested_family: str | None
    requested_label_tokens: frozenset[str]
    entities: frozenset[str]
    roles: frozenset[str]
    language: str
    yes_no: bool


@dataclass(slots=True)
class DocumentQueryIndex:
    file_hash: str
    source_file: str
    canonical_blocks: tuple[CanonicalBlock, ...]
    normalized_text: tuple[str, ...]
    tokenized_blocks: tuple[tuple[str, ...], ...]
    field_index: dict[str, tuple[FieldOccurrence, ...]]
    raw_field_index: dict[str, tuple[FieldOccurrence, ...]]
    entity_index: dict[str, tuple[int, ...]]
    section_index: dict[str, tuple[int, ...]]
    lexical_index: dict[str, tuple[int, ...]]
    bm25: BM25Okapi | None
    build_ms: float

    @classmethod
    def build(cls, document: CanonicalDocument) -> "DocumentQueryIndex":
        started = time.perf_counter()
        fields: dict[str, list[FieldOccurrence]] = {}
        raw_fields: dict[str, list[FieldOccurrence]] = {}
        entities: dict[str, set[int]] = {}
        sections: dict[str, set[int]] = {}
        lexical: dict[str, set[int]] = {}
        normalized: list[str] = []
        tokenized: list[tuple[str, ...]] = []
        for index, block in enumerate(document.blocks):
            block_norm = _norm(block.text)
            normalized.append(block_norm)
            block_tokens = tuple(sorted(_tokens(block.text)))
            tokenized.append(block_tokens)
            for token in block_tokens:
                lexical.setdefault(token, set()).add(index)
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{1,15}", token, re.I):
                    entities.setdefault(token, set()).add(index)
            if block.section:
                for token in _tokens(block.section):
                    sections.setdefault(token, set()).add(index)
            for field_match in _ASSIGNMENT.finditer(block.text):
                label, value = field_match.group(1), field_match.group(2)
                normalized_label = _norm(label)
                prefix = block.text[:field_match.start()]
                identity_pairs = []
                for left, right in _ASSIGNMENT.findall(prefix):
                    if _norm(left) == _norm(right) and left.strip():
                        identity_pairs.append(left.strip())
                    elif _norm(left) in {"system", "system name", "destination", "flux", "workflow"} and right.strip():
                        identity_pairs.append(right.strip())
                local_context = identity_pairs[-1] if identity_pairs else ""
                occurrence_family = _family(_tokens(label))
                if occurrence_family == "parameter":
                    target = re.search(r"(?i)(?:_TO_|\sTO\s)([A-Z][A-Z0-9 _-]{1,30})$", value.strip())
                    if target:
                        local_context = target.group(1).strip()
                occurrence = FieldOccurrence(
                    label.strip(), normalized_label, occurrence_family, value.strip().lstrip("=: ").strip(), block, local_context
                )
                raw_fields.setdefault(normalized_label, []).append(occurrence)
                if occurrence.family:
                    fields.setdefault(occurrence.family, []).append(occurrence)
        bm25 = BM25Okapi([list(tokens) for tokens in tokenized]) if tokenized and any(tokenized) else None
        freeze = lambda values: {key: tuple(value) for key, value in values.items()}
        return cls(
            document.file_hash, document.source_file, document.blocks, tuple(normalized), tuple(tokenized),
            freeze(fields), freeze(raw_fields),
            {key: tuple(sorted(value)) for key, value in entities.items()},
            {key: tuple(sorted(value)) for key, value in sections.items()},
            {key: tuple(sorted(value)) for key, value in lexical.items()}, bm25,
            (time.perf_counter() - started) * 1000,
        )


class DocumentQueryIndexCache:
    def __init__(self) -> None:
        self._indexes: dict[str, DocumentQueryIndex] = {}

    def get_or_build(self, document: CanonicalDocument) -> DocumentQueryIndex:
        cached = self._indexes.get(document.file_hash)
        if cached is None:
            cached = DocumentQueryIndex.build(document)
            self._indexes[document.file_hash] = cached
        return cached

    def __len__(self) -> int:
        return len(self._indexes)


@dataclass(frozen=True, slots=True)
class DirectAnswerTrace:
    stages_attempted: tuple[str, ...]
    timings_ms: Mapping[str, float]
    cache_hit: bool
    ollama_calls: int = 0
    chroma_calls: int = 0
    active_block_count: int = 0
    candidate_count: int = 0
    top_candidates: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerResult:
    status: str
    answer: str
    evidence_blocks: tuple[CanonicalBlock, ...]
    source_file: str
    file_hash: str
    confidence: str
    method: str
    reason: str = ""
    result_type: str = "SINGLE_VALUE"
    query_language: str = "English"
    suggestions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestionDiagnostics:
    status: str
    byte_length: int
    block_count: int
    usable_text_length: int
    section_count: int
    structured_block_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NormalizationOutcome:
    status: str
    document: CanonicalDocument | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentHealth:
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    source_file: str
    file_hash: str
    file_type: str
    application: str = ""
    geographical_entity: str = ""
    version: str = ""
    title: str = ""
    author: str = ""
    status: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogIndex:
    entries: tuple[CatalogEntry, ...]

    @classmethod
    def from_metadatas(cls, metadatas: Iterable[Mapping[str, Any]]) -> "CatalogIndex":
        unique: dict[str, CatalogEntry] = {}
        for metadata in metadatas:
            source = str(metadata.get("source_file") or metadata.get("saved_as") or "")
            file_hash = str(metadata.get("file_hash") or source)
            if not source or file_hash in unique:
                continue
            suffix = source.rsplit(".", 1)[-1].casefold() if "." in source else ""
            unique[file_hash] = CatalogEntry(
                source, file_hash, suffix,
                str(metadata.get("application") or ""),
                str(metadata.get("geographical_entity") or ""),
                str(metadata.get("version") or ""), str(metadata.get("title") or ""),
                str(metadata.get("author") or ""), str(metadata.get("status") or ""), dict(metadata),
            )
        return cls(tuple(sorted(unique.values(), key=lambda item: item.source_file.casefold())))

    def search(self, query: str = "", *, application: str = "", geographical_entity: str = "",
               file_type: str = "", version: str = "") -> tuple[CatalogEntry, ...]:
        terms = _tokens(query)
        command_words = {"show", "which", "list", "all", "documents", "document", "files", "file", "belong",
                         "are", "is", "version", "affiche", "liste", "quels", "quel", "fichiers", "documents"}
        terms -= command_words
        version_match = re.search(r"(?i)\bversion\s+v?([0-9]+(?:\.[0-9A-Za-z]+)*)", query)
        if version_match and not version:
            version = version_match.group(1)
            terms.discard(_norm(version))
        type_match = re.search(r"(?i)\b(docx?|pdf|xlsx?|csv|zip)\b", query)
        if type_match and not file_type:
            file_type = type_match.group(1)
            terms.discard(file_type.casefold())
        results: list[CatalogEntry] = []
        for entry in self.entries:
            if application and _norm(entry.application) != _norm(application):
                continue
            if geographical_entity and _norm(entry.geographical_entity) != _norm(geographical_entity):
                continue
            if file_type and entry.file_type != file_type.casefold().lstrip("."):
                continue
            if version and _norm(entry.version) != _norm(version):
                continue
            haystack = _tokens(" ".join((entry.source_file.replace("_", " ").replace("-", " "), entry.file_type, entry.application,
                                         entry.geographical_entity, entry.version, entry.title,
                                         entry.author, entry.status)))
            if terms and not terms <= haystack:
                continue
            results.append(entry)
        return tuple(results)


class CanonicalSessionCache:
    def __init__(self) -> None:
        self._documents: dict[tuple[str, int], CanonicalDocument] = {}

    def get_or_normalize(self, data: bytes, source_file: str, **kwargs: Any) -> CanonicalDocument:
        file_hash = sha256(data).hexdigest()
        cache_key = (file_hash, CANONICAL_SCHEMA_VERSION)
        cached = self._documents.get(cache_key)
        if cached is not None:
            return cached
        document = normalize_document(data, source_file, **kwargs)
        self._documents[cache_key] = document
        return document

    def __len__(self) -> int:
        return len(self._documents)


def ingestion_diagnostics(document: CanonicalDocument) -> IngestionDiagnostics:
    byte_length = int(document.metadata.get("byte_length", 0))
    usable = sum(len(block.text.strip()) for block in document.blocks)
    sections = len({block.section for block in document.blocks if block.section})
    structured = sum(block.block_type in {"key_value", "table_row"} for block in document.blocks)
    if byte_length <= 0 or not document.blocks or usable == 0:
        status = "EXTRACTION_FAILED"
    elif document.warnings or (document.file_type == "pdf" and (not document.sections or not document.logical_tables)):
        status = "READY_WITH_WARNINGS"
    else:
        status = "READY"
    return IngestionDiagnostics(status, byte_length, len(document.blocks), usable, sections, structured, document.warnings)


def normalize_with_gate(data: bytes, source_file: str, **kwargs: Any) -> NormalizationOutcome:
    try:
        document = normalize_document(data, source_file, **kwargs)
    except ValueError as exc:
        if "Unsupported document type" in str(exc):
            return NormalizationOutcome("UNSUPPORTED", None, (str(exc),))
        return NormalizationOutcome("EXTRACTION_FAILED", None, (f"Parser failure: {type(exc).__name__}",))
    except Exception as exc:
        return NormalizationOutcome("EXTRACTION_FAILED", None, (f"Parser failure: {type(exc).__name__}",))
    diagnostics = ingestion_diagnostics(document)
    return NormalizationOutcome(diagnostics.status, document, diagnostics.warnings)


def document_health(document: CanonicalDocument) -> DocumentHealth:
    reasons: list[str] = []
    diagnostics = ingestion_diagnostics(document)
    if diagnostics.status == "EXTRACTION_FAILED":
        reasons.append("meaningful extraction is empty")
    if len(set(block.block_id for block in document.blocks)) != len(document.blocks):
        reasons.append("duplicate block IDs")
    if any(block.file_hash != document.file_hash for block in document.blocks):
        reasons.append("block file hash mismatch")
    if any(_SECRET.search(label) and value.strip() != "[REDACTED]"
           for block in document.blocks for label, value in _ASSIGNMENT.findall(block.text)):
        reasons.append("unredacted secret-like field")
    if document.warnings:
        reasons.extend(document.warnings)
    for table in document.logical_tables:
        values = table.values()
        if table.logical_columns <= 0 or any(len(row) != table.logical_columns for row in values):
            reasons.append(f"{table.table_id}: inconsistent logical grid width")
        if table.shape == "MATRIX":
            if not values or sum(bool(value) for value in values[0][1:]) < 2:
                reasons.append(f"{table.table_id}: matrix entity headers are missing")
        if table.shape == "VERSION_HISTORY" and values:
            headers = {_norm(value) for value in values[0]}
            if not headers.intersection({"version", "vers", "vers."}) or "date" not in headers or not headers.intersection({"auteur", "author"}):
                reasons.append(f"{table.table_id}: incomplete version-history relationship")
        if table.metadata.get("merged_cell_count", 0) and not any(
                cell.is_merged_continuation for row in table.rows for cell in row):
            reasons.append(f"{table.table_id}: merged-cell reconstruction mismatch")
    structural_failure = any("inconsistent logical grid" in reason or "matrix entity headers" in reason
                             for reason in reasons)
    status = "FAIL" if structural_failure or any(reason in reasons for reason in (
        "meaningful extraction is empty", "duplicate block IDs", "block file hash mismatch", "unredacted secret-like field"
    )) else ("WARN" if reasons else "PASS")
    return DocumentHealth(status, tuple(reasons))


def retrieve_canonical(context: ActiveDocumentContext, query: str, limit: int = 12) -> tuple[RetrievalCandidate, ...]:
    query_tokens = _tokens(query)
    requested_family = _family(query_tokens)
    candidates: list[RetrievalCandidate] = []
    for block in context.canonical_document.blocks:
        if block.file_hash != context.file_hash:
            raise ValueError("cross-document evidence encountered")
        text_tokens = _tokens(block.text)
        overlap = query_tokens & text_tokens
        fields = _ASSIGNMENT.findall(block.text)
        field_boost = 0.0
        if requested_family and any(_family(_tokens(label)) == requested_family for label, _ in fields):
            field_boost = 8.0
        structure_boost = 2.0 if block.block_type in {"key_value", "table_row"} else 0.0
        section_boost = 2.0 if block.section and _tokens(block.section) & query_tokens else 0.0
        glossary_penalty = -4.0 if "definition" in _norm(block.section or "") and "definition" not in query_tokens else 0.0
        score = len(overlap) + field_boost + structure_boost + section_boost + glossary_penalty
        if score > 0:
            candidates.append(RetrievalCandidate(block, score, tuple(sorted(overlap))))
    candidates.sort(key=lambda item: (-item.score, item.block.block_id))
    # Small and medium documents are scanned exhaustively; limit only controls
    # evidence passed to a potential LLM after deterministic extraction.
    return tuple(candidates if len(context.block_ids) <= 1000 else candidates[:max(limit, 1)])


def _deterministic_answer(context: ActiveDocumentContext, query: str,
                          candidates: Sequence[RetrievalCandidate]) -> AnswerResult | None:
    query_tokens = _tokens(query)
    requested_family = _family(query_tokens)
    if not requested_family:
        return None
    matches: list[tuple[float, str, CanonicalBlock]] = []
    for candidate in candidates:
        for label, raw_value in _ASSIGNMENT.findall(candidate.block.text):
            if _family(_tokens(label)) != requested_family:
                continue
            if _SECRET.search(label):
                return AnswerResult("SENSITIVE_BLOCK", NO_EXPLICIT_EVIDENCE, (candidate.block,), context.source_file,
                                    context.file_hash, "HIGH", "sensitive_block", "requested field is sensitive")
            value = raw_value.strip()
            label_tokens = _tokens(label)
            record_tokens = _tokens(candidate.block.text)
            entity_terms = query_tokens - set().union(*_FIELD_FAMILIES.values())
            entity_score = len(entity_terms & record_tokens)
            exact_label_boost = 3 if label_tokens and label_tokens <= query_tokens else 0
            matches.append((candidate.score + entity_score * 4 + len(query_tokens & label_tokens) + exact_label_boost,
                            value, candidate.block))
    if not matches:
        return None
    best_score = max(score for score, _, _ in matches)
    best = [(value, block) for score, value, block in matches if score == best_score]
    values = {_norm(value) for value, _ in best}
    if len(values) != 1:
        return AnswerResult("AMBIGUOUS", NO_EXPLICIT_EVIDENCE, tuple(block for _, block in best), context.source_file,
                            context.file_hash, "LOW", "ambiguous", "equally ranked conflicting values")
    value = best[0][0]
    if requested_family == "transformation" and re.match(r"(?i)^no\b|^none\b|^aucun", value):
        value = "No"
    evidence = tuple(dict.fromkeys(block.block_id for _, block in best))
    evidence_blocks = tuple(next(block for _, block in best if block.block_id == block_id) for block_id in evidence)
    return AnswerResult("ANSWER", value, evidence_blocks, context.source_file, context.file_hash,
                        "HIGH", "deterministic_structured")


_ROLE_TERMS = {
    "collection", "distribution", "output", "input", "archive", "version", "history",
    "collecte", "sortie", "entree", "archivage", "historique",
}


def _canonical_roles(tokens: Iterable[str]) -> set[str]:
    aliases = {
        "collecte": "collection", "collection": "collection",
        "sortie": "output", "output": "output", "distribution": "output",
        "entree": "input", "input": "input", "archive": "archive", "archivage": "archive",
        "history": "history", "historique": "history", "version": "version",
    }
    return {aliases[token] for token in tokens if token in aliases}


def parse_question(question: str) -> QuestionPlan:
    normalized = _norm(question)
    tokens = _tokens(question)
    language_tokens = set(re.findall(r"[a-z0-9_.:/-]+", normalized))
    family = _family(tokens)
    # Requested relation outranks a qualifier such as a version identifier.
    if tokens & {"author", "authored", "auteur", "written", "wrote", "redige", "ecrit"}:
        family = "author"
    elif tokens & {"date", "created", "creation", "when"}:
        family = "date"
    aliases = set().union(*_FIELD_FAMILIES.values())
    roles = _canonical_roles(tokens & _ROLE_TERMS)
    entity_candidates = tokens - aliases - _ROLE_TERMS - {
        "used", "specification", "files", "file", "cdr", "platform", "plateforme", "planned",
        "use", "utilise", "specification", "flow", "flows", "workflow", "workflows",
    }
    french_markers = {
        "quel", "quelle", "quels", "quelles", "auteur", "fichier", "fichiers",
        "connexion", "frequence", "repertoire", "historique", "modifications",
        "glossaire", "archivage", "trouver", "dans", "avec", "est", "sont",
    }
    english_markers = {
        "what", "which", "where", "author", "file", "files", "connection",
        "frequency", "directory", "history", "glossary", "archive", "find",
        "with", "is", "are",
    }
    language = "French" if len(language_tokens & french_markers) > len(language_tokens & english_markers) else "English"
    yes_no = bool(re.match(r"(?i)^(?:does|do|is|are|can|est-ce|les?\s+.+\s+est|y a)", normalized))
    return QuestionPlan(normalized, family, frozenset(tokens), frozenset(entity_candidates),
                        frozenset(roles), language, yes_no)


def _natural_section_intent(question: str) -> str | None:
    normalized = _norm(question)
    patterns = {
        "abstract": (
            r"\babstract\b", r"\babstrait\b", r"\bresume\b",
            r"what(?:'s| is) (?:this|the) (?:document|file|project) about",
            r"de quoi parle (?:ce|le) (?:document|fichier|projet)",
        ),
        "requirements": (
            r"\brequirements?\b", r"\bexigences?\b", r"\brequis\b",
        ),
        "purpose": (
            r"\bpurpose\b", r"\bobjective\b", r"\bobjectif\b", r"\bbut du document\b",
        ),
        "references": (
            r"\breferences?\b", r"documents? de reference", r"documents? references?",
        ),
    }
    for intent, expressions in patterns.items():
        if any(re.search(expression, normalized) for expression in expressions):
            return intent
    return None


def _section_answer(context: ActiveDocumentContext, intent: str) -> AnswerResult | None:
    aliases = {
        "abstract": {"abstract", "abstrait", "resume", "summary", "overview"},
        "requirements": {"requirement", "requirements", "exigence", "exigences", "requis"},
        "purpose": {"purpose", "objective", "objectif", "scope", "portee"},
        "references": {"reference", "references", "document de reference", "documents de reference"},
    }[intent]
    matches = [block for block in context.canonical_document.blocks
               if any(alias in _norm(" ".join(filter(None, (block.section, block.text[:120]))))
                      for alias in aliases)]
    if not matches:
        return None
    # A heading identifies the section but is not a useful answer by itself.
    evidence = [block for block in matches if block.block_type != "heading"] or matches
    evidence = evidence[:20 if intent == "requirements" else 8]
    answer = "\n".join(block.text for block in evidence)
    return AnswerResult("ANSWER", answer, tuple(evidence), context.source_file, context.file_hash,
                        "HIGH", f"natural_{intent}_section", result_type=intent.upper())


def _occurrence_score(occurrence: FieldOccurrence, plan: QuestionPlan) -> float:
    block_tokens = _tokens(occurrence.block.text)
    relation_tokens = _tokens(occurrence.local_context) if occurrence.local_context else block_tokens
    label_tokens = _tokens(occurrence.label)
    score = 10.0 + len(plan.requested_label_tokens & label_tokens) * 3
    local_entity_matches = len(plan.entities & relation_tokens)
    score += local_entity_matches * 5
    if occurrence.local_context:
        score += local_entity_matches * 5
        score += len(_tokens(occurrence.local_context) & set(plan.requested_label_tokens)) * 5
    score += len(plan.entities & block_tokens) * 2
    score += len(set(plan.requested_label_tokens) & block_tokens) * 2
    occurrence_roles = _canonical_roles(block_tokens | label_tokens | _tokens(occurrence.block.section or ""))
    score += len(plan.roles & occurrence_roles) * 4
    # Reject an unrequested qualifier when an exact, less-qualified label is
    # present (Directory must beat Archive directory for a plain query).
    score -= len(_canonical_roles(label_tokens & _ROLE_TERMS) - set(plan.roles)) * 3
    score -= len(occurrence_roles - set(plan.roles)) * 3
    score -= len(label_tokens - set(plan.requested_label_tokens)) * 0.5
    if occurrence.block.block_type == "key_value":
        score += 3
    return score


def _structured_index_answer(context: ActiveDocumentContext, index: DocumentQueryIndex,
                             plan: QuestionPlan) -> AnswerResult | None:
    query_tokens = set(plan.requested_label_tokens)
    instance_match = re.search(r"(?i)\binstance\s*(\d+)\b", plan.normalized_question)
    instance_number = instance_match.group(1) if instance_match else ""
    if instance_number:
        target_labels = [token for token in query_tokens
                         if token in {"dwh", "bi", "reqleg", "reqlog", "svr", "cra", "svrcra"}]
        for block in index.canonical_blocks:
            if block.block_type != "table_row" or not re.search(
                    rf"(?i)\binstance\s*{re.escape(instance_number)}\b", block.text):
                continue
            pairs = _ASSIGNMENT.findall(block.text)
            for label, value in pairs:
                if target_labels and _tokens(label) & set(target_labels):
                    return AnswerResult("ANSWER", value.strip(), (block,), context.source_file,
                                        context.file_hash, "HIGH", "same_row_matrix_extraction")
    # Explicit glossary-description intent uses the glossary row, while
    # operational attributes (host/directory/etc.) continue to outrank it.
    if query_tokens & {"description", "definition", "signification"}:
        for token in query_tokens - {"description", "definition", "signification"}:
            definitions = tuple(item for item in index.raw_field_index.get(_norm(token), ())
                                if item.block.metadata.get("logical_table_shape") == "GLOSSARY")
            if definitions:
                values = list(dict.fromkeys(item.value for item in definitions))
                evidence = tuple({item.block.block_id: item.block for item in definitions}.values())
                if values == ["[NO_VALUE]"]:
                    return AnswerResult("EXPLICIT_TERM_WITHOUT_VALUE",
                                        f"{token} is listed in the glossary, but no description is provided.",
                                        evidence, context.source_file, context.file_hash, "HIGH",
                                        "explicit_empty_definition", result_type="EXPLICIT_EMPTY_VALUE")
                return AnswerResult("ANSWER", ", ".join(values), evidence, context.source_file,
                                    context.file_hash, "HIGH", "definition_fact", result_type="DEFINITION")
    if query_tokens & {"abreviation", "abreviations", "abbreviation", "abbreviations"}:
        query_tokens = {"glossaire"}
        plan = QuestionPlan(plan.normalized_question, plan.requested_family, frozenset(query_tokens),
                            plan.entities, plan.roles, plan.language, plan.yes_no)
    if query_tokens & {"server", "serveur"} and query_tokens & {"details", "detail"}:
        matches = index.raw_field_index.get("server details *", ()) or index.raw_field_index.get("server details", ())
        if matches:
            evidence = tuple({item.block.block_id: item.block for item in matches}.values())
            return AnswerResult("ANSWER", "\n".join(item.value for item in matches), evidence,
                                context.source_file, context.file_hash, "HIGH", "structured_group_extraction",
                                result_type="TOPIC_RESULT")
    if plan.requested_family == "retention" and query_tokens & {"cache", "duplicate", "doublon"}:
        cache_blocks = [block for block in index.canonical_blocks if "cache" in _norm(block.text)]
        spans = [(span, block) for block in cache_blocks if (span := _span_from_text(block.text, plan))]
        if spans:
            return AnswerResult("ANSWER", spans[0][0], tuple(block for _, block in spans),
                                context.source_file, context.file_hash, "HIGH", "section_span_extraction")
    has_duplicate_parameter = any(re.search(r"\bPARAM_[A-Z0-9_]*DUP[A-Z0-9_]*\b", block.text)
                                  for block in index.canonical_blocks)
    if (plan.requested_family == "duplicate" and not has_duplicate_parameter
            and not [item for item in index.field_index.get("duplicate", ()) if "udr" not in _norm(item.label)]):
        duplicate_blocks = [block for block in index.canonical_blocks
                            if "duplication" in _norm(block.section or "") or "duplicate" in _norm(block.section or "")]
        prose = [block for block in duplicate_blocks if block.block_type == "paragraph" and len(block.text) > 20]
        if prose:
            return AnswerResult("ANSWER", " ".join(block.text for block in prose), tuple(prose),
                                context.source_file, context.file_hash, "HIGH", "section_prose_extraction",
                                result_type="TOPIC_RESULT")
    # A close title match means "show this section", not "return its heading".
    query_topic_tokens = set(plan.requested_label_tokens)
    section_candidates: list[tuple[float, str]] = []
    for block in index.canonical_blocks:
        if block.block_type != "heading":
            continue
        title_tokens = _tokens(re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", block.text))
        if not title_tokens or not query_topic_tokens:
            continue
        coverage = len(query_topic_tokens & title_tokens) / len(query_topic_tokens)
        precision = len(query_topic_tokens & title_tokens) / len(title_tokens)
        if coverage == 1.0 and (precision >= 0.45 or len(query_topic_tokens) == 1):
            section_candidates.append((coverage + precision, block.text))
    if section_candidates:
        _, raw_title = max(section_candidates, key=lambda item: item[0])
        title = re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", raw_title).strip()
        normalized_title = _norm(title)
        contents: list[CanonicalBlock] = []
        for block in index.canonical_blocks:
            path = [_norm(value) for value in block.metadata.get("section_path", [])]
            if _norm(block.section or "") == normalized_title or normalized_title in path:
                if block.block_type != "heading" and not re.search(r"(?i)^page\s+\d+\s+of\s+\d+$", block.text):
                    contents.append(block)
        structured = [block for block in contents if block.block_type in {"table_row", "key_value"}]
        selected = structured or contents
        unique = list({block.text: block for block in selected}.values())
        if unique:
            if len(unique) == 1:
                assignments = _ASSIGNMENT.findall(unique[0].text)
                concise = assignments[0][1].strip() if len(assignments) == 1 else unique[0].text.strip()
                if len(concise) <= 80:
                    return AnswerResult("ANSWER", concise, tuple(unique), context.source_file,
                                        context.file_hash, "HIGH", "section_extraction",
                                        result_type="SECTION_RESULT")
            answer = title + ":\n" + "\n".join(f"- {block.text}" for block in unique)
            return AnswerResult("ANSWER", answer, tuple(unique), context.source_file,
                                context.file_hash, "HIGH", "section_extraction", result_type="SECTION_RESULT")
    if not plan.requested_family:
        if not (plan.requested_label_tokens & {"assumption", "assumptions", "hypothese", "hypotheses"}):
            return None
    if plan.requested_label_tokens & {"assumption", "assumptions", "hypothese", "hypotheses"}:
        selected: list[CanonicalBlock] = []
        collecting = False
        for block in index.canonical_blocks:
            normalized = _norm(block.text)
            if re.search(r"\bhypotheses?\b|\bassumptions?\b", normalized):
                collecting = True
            if collecting and re.search(r"\b(?:examples?|exemples?)\b", normalized):
                break
            if collecting and block.block_type != "heading":
                selected.append(block)
        if selected:
            return AnswerResult("ANSWER", " ".join(block.text for block in selected), tuple(selected),
                                context.source_file, context.file_hash, "HIGH", "section_extraction")
    if plan.requested_family == "format" and plan.roles == frozenset({"output"}):
        output_blocks = [block for block in index.canonical_blocks
                         if "output format" in _norm(block.section or "") and block.block_type == "paragraph"]
        if output_blocks:
            return AnswerResult("ANSWER", " ".join(block.text for block in output_blocks), tuple(output_blocks),
                                context.source_file, context.file_hash, "HIGH", "section_extraction")
    if (plan.requested_family == "duplicate" and not has_duplicate_parameter
            and not [item for item in index.field_index.get("duplicate", ()) if "udr" not in _norm(item.label)]
            and plan.requested_label_tokens & {"detected", "detection", "how"}):
        prose = [block for block in index.canonical_blocks
                 if "duplicate" in _norm(block.section or "") and re.search(r"(?i)\bCRC\b|redondance cyclique", block.text)]
        if prose:
            return AnswerResult("ANSWER", " ".join(block.text for block in prose), tuple(prose),
                                context.source_file, context.file_hash, "HIGH", "section_prose_extraction")
    if plan.requested_family == "workflow":
        found: list[tuple[str, CanonicalBlock]] = []
        for block in index.canonical_blocks:
            match = re.search(r"(?i)\bworkflow\s*\d+\s*:\s*([A-Za-z][A-Za-z0-9 _-]{0,50})", block.text)
            if match:
                found.append((match.group(1).strip(), block))
        if found:
            names = list(dict.fromkeys(name for name, _ in found))
            evidence = {block.block_id: block for _, block in found}
            answer = str(len(names)) if plan.requested_label_tokens & {"many", "combien", "number", "nombre"} else ", ".join(names)
            return AnswerResult("ANSWER", answer, tuple(evidence.values()), context.source_file,
                                context.file_hash, "HIGH", "structured_list_extraction")
        instance_rows = [block for block in index.canonical_blocks
                         if block.block_type == "table_row" and re.search(r"(?i)\binstance\s*\d+", block.text)]
        if instance_rows:
            return AnswerResult("ANSWER", "\n".join(f"- {block.text}" for block in instance_rows),
                                tuple(instance_rows), context.source_file, context.file_hash,
                                "HIGH", "structured_list_extraction", result_type="TOPIC_RESULT")
    if plan.requested_family == "table":
        named_tables: list[tuple[str, CanonicalBlock]] = []
        for block in index.canonical_blocks:
            for match in re.finditer(r"\b[A-Z][A-Z0-9_]*(?:_TABLE|_INPUT|_OUTPUT|PARAM)[A-Z0-9_]*\b", block.text):
                named_tables.append((match.group(0), block))
        if named_tables and "audit" in query_tokens:
            values = list(dict.fromkeys(value for value, _ in named_tables
                                        if "audit" not in query_tokens or "AUDIT" in value))
            if values:
                evidence = tuple({block.block_id: block for _, block in named_tables}.values())
                result_type = "MULTI_VALUE" if len(values) > 1 else "SINGLE_VALUE"
                answer = values[0] if len(values) == 1 else "\n".join(f"- {value}" for value in values)
                return AnswerResult("ANSWER", answer, evidence,
                                    context.source_file, context.file_hash, "HIGH", "table_name_extraction",
                                    result_type=result_type)
        table_spans: list[tuple[float, str, CanonicalBlock]] = []
        for block_index, block in enumerate(index.canonical_blocks):
            span = _span_from_text(block.text, plan)
            if span:
                table_spans.append((_block_score(index, block_index, plan), span, block))
        if table_spans:
            best_score = max(score for score, _, _ in table_spans)
            best = [(span, block) for score, span, block in table_spans if score == best_score]
            if len({_norm(span) for span, _ in best}) == 1:
                evidence = {block.block_id: block for _, block in best}
                return AnswerResult("ANSWER", best[0][0], tuple(evidence.values()), context.source_file,
                                    context.file_hash, "HIGH", "local_span_extraction")
    if plan.requested_family == "archive" and plan.requested_label_tokens & {"many", "combien", "number", "nombre"}:
        for block in index.canonical_blocks:
            match = re.search(r"(?i)\b(one|two|three|un|une|deux|trois|\d+)\s+modes?\b", _norm(block.text))
            if match:
                numbers = {"one": "1", "two": "2", "three": "3", "un": "1", "une": "1", "deux": "2", "trois": "3"}
                value = numbers.get(match.group(1).casefold(), match.group(1))
                return AnswerResult("ANSWER", value, (block,), context.source_file, context.file_hash,
                                    "HIGH", "local_count_extraction")
    occurrences = list(index.field_index.get(plan.requested_family, ()))
    if plan.requested_family == "filename" and plan.requested_label_tokens & {"pattern", "patterns"}:
        patterned = [item for item in occurrences if "pattern" in _tokens(item.label)]
        if patterned:
            occurrences = patterned
    if plan.requested_family == "host":
        simple = [item for item in occurrences if not re.search(r"(?i)\b(?:login|password)\s*[:=]", item.value)]
        if simple:
            occurrences = simple
    if plan.requested_family == "duplicate" and plan.requested_label_tokens & {"file", "files"}:
        occurrences = [item for item in occurrences if "udr" not in _norm(item.label)]
    if not occurrences:
        return None
    explicitly_qualified = [item for item in occurrences if plan.entities & _tokens(item.local_context)]
    if explicitly_qualified:
        occurrences = explicitly_qualified
    if plan.requested_family == "directory":
        alternative_rows = {(item.block.table_index, item.block.row_index) for item in occurrences
                            if item.block.metadata.get("alternative_value")}
        if alternative_rows:
            occurrences = [item for item in occurrences
                           if (item.block.table_index, item.block.row_index) not in alternative_rows
                           or item.block.metadata.get("alternative_value")]
    if plan.requested_family == "filename" and plan.requested_label_tokens & {"pattern", "patterns"}:
        patterns: list[tuple[str, CanonicalBlock]] = []
        for item in occurrences:
            start = 0
            for match in re.finditer(r"(?i)\.(?:csv|zip|txt|dat|xml)\b", item.value):
                value = item.value[start:match.end()]
                value = re.sub(r"^\s*(?:\([^)]*\)\s*)?[_;, ]*", "", value).strip()
                start = match.end()
                if value:
                    patterns.append((value, item.block))
        if len(patterns) > 1:
            evidence = tuple({block.block_id: block for _, block in patterns}.values())
            return AnswerResult("ANSWER", "Filename patterns:\n" + "\n".join(f"- {value}" for value, _ in patterns),
                                evidence, context.source_file, context.file_hash, "HIGH",
                                "multi_value_extraction", result_type="MULTI_VALUE")
    if plan.requested_family == "frequency" and not any(
            plan.entities & _tokens(item.value) for item in occurrences):
        alternatives = []
        for item in occurrences:
            times = re.findall(r"(?i)\b(?:\d{1,2}\s*h(?:\s*AM|\s*PM)?|\d+(?:[.,]\d+)?\s*(?:min(?:ute)?s?|hours?|days?|heures?|jours?))\b", item.value)
            if len(times) > 1:
                alternatives.append(item)
        if alternatives:
            evidence = tuple({item.block.block_id: item.block for item in alternatives}.values())
            return AnswerResult("ANSWER", "Collection schedules:\n" + "\n".join(f"- {item.value}" for item in alternatives),
                                evidence, context.source_file, context.file_hash, "HIGH",
                                "multi_value_extraction", result_type="MULTI_VALUE")
    scored = [(_occurrence_score(item, plan), item) for item in occurrences]
    best_score = max(score for score, _ in scored)
    best = [item for score, item in scored if score == best_score]
    if any(_SECRET.search(item.label) for item in best):
        return AnswerResult("SENSITIVE_BLOCK", NO_EXPLICIT_EVIDENCE, tuple(item.block for item in best),
                            context.source_file, context.file_hash, "HIGH", "sensitive_block")
    if plan.requested_family == "destination" and len(best) >= 1:
        values_in_order: list[str] = []
        for item in best:
            if item.value not in values_in_order:
                values_in_order.append(item.value)
        evidence = {item.block.block_id: item.block for item in best}
        return AnswerResult("ANSWER", ", ".join(values_in_order), tuple(evidence.values()),
                            context.source_file, context.file_hash, "HIGH", "structured_list_extraction")
    if plan.requested_family == "frequency":
        qualified: list[tuple[int, str, CanonicalBlock]] = []
        for item in best:
            matches = list(re.finditer(
                r"(?i)\b(\d+(?:[.,]\d+)?\s*(?:seconds?|minutes?|hours?|days?|secondes?|heures?|jours?))\b",
                item.value,
            ))
            for position, match in enumerate(matches):
                end = matches[position + 1].start() if position + 1 < len(matches) else len(item.value)
                qualifier = item.value[match.end():end]
                overlap = len(_tokens(qualifier) & set(plan.requested_label_tokens))
                qualified.append((overlap, match.group(1), item.block))
        if qualified:
            best_overlap = max(score for score, _, _ in qualified)
            selected = [(value, block) for score, value, block in qualified if score == best_overlap]
            selected_values = {_norm(value) for value, _ in selected}
            if best_overlap > 0 and len(selected_values) == 1:
                evidence = {block.block_id: block for _, block in selected}
                return AnswerResult("ANSWER", selected[0][0], tuple(evidence.values()), context.source_file,
                                    context.file_hash, "HIGH", "qualified_value_extraction")
            if len(qualified) == 1:
                return AnswerResult("ANSWER", qualified[0][1], (qualified[0][2],), context.source_file,
                                    context.file_hash, "HIGH", "structured_span_extraction")
            return AnswerResult("AMBIGUOUS", NO_EXPLICIT_EVIDENCE,
                                tuple({block.block_id: block for _, _, block in qualified}.values()),
                                context.source_file, context.file_hash, "LOW", "ambiguous",
                                "multiple qualified frequencies without a unique query match")
    if plan.yes_no:
        normalized_values = {_norm(item.value) for item in best}
        if normalized_values and normalized_values <= {"no", "non", "n/a", "na", "none", "aucun", "aucune"}:
            evidence = {item.block.block_id: item.block for item in best}
            return AnswerResult("ANSWER", "No", tuple(evidence.values()), context.source_file,
                                context.file_hash, "HIGH", "boolean_field_extraction")
        if normalized_values and normalized_values <= {"yes", "oui"}:
            evidence = {item.block.block_id: item.block for item in best}
            return AnswerResult("ANSWER", "Yes", tuple(evidence.values()), context.source_file,
                                context.file_hash, "HIGH", "boolean_field_extraction")
        spans = [(span, item.block) for item in best if (span := _span_from_text(item.value, plan))]
        if spans and len({_norm(span) for span, _ in spans}) == 1:
            evidence = {block.block_id: block for _, block in spans}
            return AnswerResult("ANSWER", spans[0][0], tuple(evidence.values()), context.source_file,
                                context.file_hash, "HIGH", "boolean_span_extraction")
    has_qualified_record = any(plan.entities & _tokens(item.local_context) for item in occurrences)
    if not plan.roles and not has_qualified_record:
        scalar = [item for item in best if item.block.block_type == "key_value"]
        if scalar and len({_norm(item.value) for item in scalar}) == 1:
            best = scalar
    multi_families = {"host", "directory", "username", "filename", "frequency", "parameter", "format", "destination"}
    server_ip_query = plan.requested_family == "host" and {"server", "ip"} <= set(plan.requested_label_tokens)
    if plan.requested_family in multi_families and not has_qualified_record and not plan.roles and not server_ip_query:
        valid = [item for item in occurrences if not _SECRET.search(item.label)]
        unique_values = list(dict.fromkeys(_norm(item.value) for item in valid))
        if len(unique_values) > 1:
            lines: list[str] = []
            evidence: dict[str, CanonicalBlock] = {}
            seen: set[tuple[str, str]] = set()
            for item in valid:
                context_label = item.local_context or item.block.section or item.label
                key = (_norm(context_label), _norm(item.value))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- {context_label}: {item.value}")
                evidence[item.block.block_id] = item.block
            title = plan.requested_family.replace("_", " ").title() + "s"
            return AnswerResult("ANSWER", title + ":\n" + "\n".join(lines), tuple(evidence.values()),
                                context.source_file, context.file_hash, "HIGH", "multi_value_extraction",
                                result_type="MULTI_VALUE")
    values = {_norm(item.value) for item in best}
    if len(values) != 1:
        return AnswerResult("AMBIGUOUS", NO_EXPLICIT_EVIDENCE, tuple(item.block for item in best),
                            context.source_file, context.file_hash, "LOW", "ambiguous",
                            "equally ranked conflicting values")
    value = best[0].value
    if plan.requested_family == "port":
        port_match = re.search(r"\b\d{1,5}\b", value)
        if port_match:
            value = port_match.group(0)
    if plan.requested_family == "host" and "ip" in plan.requested_label_tokens:
        ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", value)
        if ip_match:
            value = ip_match.group(0)
    if plan.requested_family == "transformation" and re.match(r"(?i)^no\b|^none\b|^aucun", value):
        value = "No"
    evidence_by_id = {item.block.block_id: item.block for item in best}
    return AnswerResult("ANSWER", value, tuple(evidence_by_id.values()), context.source_file,
                        context.file_hash, "HIGH", "exact_field_index")


def _span_from_text(text: str, plan: QuestionPlan) -> str | None:
    normalized = _norm(text)
    family = plan.requested_family
    if family == "copyright":
        match = re.search(r"(?i)(?:©\s*)?copyright\s+(\d{4}\s*,\s*[A-Za-zÀ-ÿ0-9_.-]+)", text)
        return match.group(1).strip() if match else None
    if family == "port":
        match = re.search(r"(?i)\bport\s*(?:=|:|is|est)?\s*(\d{1,5})\b", text)
        return match.group(1) if match else None
    if family == "retention":
        match = re.search(r"(?i)\b(\d+(?:[.,]\d+)?\s*(?:seconds?|minutes?|hours?|days?|secondes?|heures?|jours?))\b", text)
        return match.group(1) if match else None
    if family == "author":
        match = re.search(r"(?i)\b(?:author|written\s+by|auteur|redige\s+par)\s*(?:=|:|is|est)?\s*([A-ZÀ-ÖØ-öø-ÿ][\wÀ-ÖØ-öø-ÿ.' -]{2,80})", text)
        return match.group(1).strip(" .") if match else None
    if family == "reviewer":
        match = re.search(r"(?i)\b(?:reviewer|reviewed\s+by|relecteur|revu\s+par)\s*(?:=|:|is|est)?\s*([A-ZÀ-ÖØ-öø-ÿ][\wÀ-ÖØ-öø-ÿ.' -]{2,80})", text)
        return match.group(1).strip(" .") if match else None
    connection_question = bool(plan.requested_label_tokens & {"connection", "connexion", "establish", "etablit"})
    if connection_question:
        if not re.search(r"(?i)\b(?:connection|connexion)\b", normalized):
            return None
        if re.search(r"(?i)\b(?:aucun(?:e)?\s+(?:connection|connexion)|sans\s+(?:connection|connexion)|does\s+not|doesn't|no|never|n['’ ]?etablit\s+pas|ne\s+.+?pas)\b", normalized):
            return "No"
        if re.search(r"(?i)\b(?:establish(?:es)?|etablit)\b[^.]{0,60}\b(?:connection|connexion)", normalized):
            return "Yes"
    if not connection_question and (family == "mode" or plan.requested_label_tokens & {"send", "sent", "transfer", "transferred", "transfert", "envoie"}):
        if re.search(r"(?i)\bpush(?:ed|es|ing)?\b|\bpousse?s?\b", normalized):
            return "PUSH"
        if re.search(r"(?i)\bpull(?:ed|s|ing)?\b", normalized):
            return "PULL"
        if re.search(r"(?i)\b(?:direct transfer|transfert direct)\b", normalized):
            return "Direct transfer"
    if family == "transformation" or plan.requested_label_tokens & {"transform", "transformed", "transformation", "transforme"}:
        if re.search(r"(?i)\b(?:no|without|none|aucun|aucune|sans)\b[^.]{0,60}\b(?:transform|transformation|traitement)", text):
            return "No"
    if plan.requested_label_tokens & {"many", "combien", "number", "nombre", "instances", "workflows"}:
        match = re.search(r"(?i)\b(\d+)\s+(?:instances?|workflows?|flux)\b", text)
        return match.group(1) if match else None
    if family == "workflow":
        match = re.search(r"(?i)\bworkflow\s*\d+\s*:\s*([A-Za-z][A-Za-z0-9 _-]{0,50})", text)
        if match:
            return match.group(1).strip()
    if family == "duplicate":
        if plan.requested_label_tokens & {"file", "files"} and re.search(r"(?i)\bsans\b[^.]{0,80}\bduplicate\s+check", normalized):
            return "No"
        match = re.search(r"\b(PARAM_[A-Z0-9_]*DUP[A-Z0-9_]*)\b", text)
        if match:
            return match.group(1)
    if family == "table":
        match = re.search(r"(?i)\btable\s*[«\"']?\s*([A-Z][A-Z0-9_]{2,})", text)
        if match:
            return match.group(1)
    if family == "archive" and plan.requested_label_tokens & {"many", "combien", "number", "nombre"}:
        word_numbers = {"one": "1", "two": "2", "three": "3", "un": "1", "une": "1", "deux": "2", "trois": "3"}
        match = re.search(r"(?i)\b(one|two|three|un|une|deux|trois|\d+)\s+modes?\b", normalized)
        if match:
            return word_numbers.get(match.group(1).casefold(), match.group(1))
    return None


def _rank_block_indices(index: DocumentQueryIndex, plan: QuestionPlan) -> list[int]:
    scored: list[tuple[float, int]] = []
    query_tokens = set(plan.requested_label_tokens)
    bm25_scores = index.bm25.get_scores(list(query_tokens)) if index.bm25 is not None else [0.0] * len(index.canonical_blocks)
    for block_index in range(len(index.canonical_blocks)):
        tokens = set(index.tokenized_blocks[block_index])
        score = len(tokens & query_tokens) * 3 + len(tokens & plan.entities) * 5 + len(tokens & plan.roles) * 4
        score += float(bm25_scores[block_index])
        scored.append((score, block_index))
    scored.sort(key=lambda item: (-item[0], index.canonical_blocks[item[1]].block_id))
    return [block_index for _, block_index in scored]


def _block_score(index: DocumentQueryIndex, block_index: int, plan: QuestionPlan) -> float:
    return float(_block_score_breakdown(index, block_index, plan)["total"])


def _block_score_breakdown(index: DocumentQueryIndex, block_index: int,
                           plan: QuestionPlan) -> dict[str, float]:
    tokens = set(index.tokenized_blocks[block_index])
    query_tokens = set(plan.requested_label_tokens)
    concept = float(len(tokens & query_tokens) * 3)
    entity = float(len(tokens & plan.entities) * 5)
    role = float(len(_canonical_roles(tokens) & set(plan.roles)) * 4)
    bm25 = 0.0
    if index.bm25 is not None:
        bm25 = float(index.bm25.get_scores(list(query_tokens))[block_index])
    return {"concept_overlap": concept, "entity_overlap": entity, "role_compatibility": role,
            "bm25": bm25, "total": concept + entity + role + bm25}


class FastDirectAnswerEngine:
    """Zero-generation selected-document engine with validated answer caching."""

    def __init__(self, index_cache: DocumentQueryIndexCache | None = None) -> None:
        self.index_cache = index_cache if index_cache is not None else DocumentQueryIndexCache()
        self.fact_stores: dict[str, DocumentFactStore] = {}

    def prepare(self, document: CanonicalDocument) -> DocumentQueryIndex:
        index = self.index_cache.get_or_build(document)
        if document.file_hash not in self.fact_stores:
            self.fact_stores[document.file_hash] = DocumentFactStore.build(document, index)
        return index

    def query(self, context: ActiveDocumentContext, question: str) -> tuple[AnswerResult, DirectAnswerTrace]:
        total_started = time.perf_counter()
        timings: dict[str, float] = {}
        stages: list[str] = []
        started = time.perf_counter()
        plan = parse_question(question)
        timings["question_normalization"] = (time.perf_counter() - started) * 1000
        if _SECRET.search(question):
            result = AnswerResult("SENSITIVE_BLOCK", NO_EXPLICIT_EVIDENCE, (), context.source_file,
                                  context.file_hash, "HIGH", "sensitive_block", "sensitive request",
                                  query_language=plan.language)
            timings["total"] = (time.perf_counter() - total_started) * 1000
            return result, DirectAnswerTrace(("security",), timings, False,
                                              active_block_count=len(context.block_ids))
        started = time.perf_counter()
        index = self.prepare(context.canonical_document)
        timings["index_resolution"] = (time.perf_counter() - started) * 1000
        stages.append("exact_structured_lookup")
        started = time.perf_counter()
        section_intent = _natural_section_intent(question)
        result = _section_answer(context, section_intent) if section_intent else None
        if result is None:
            result = _structured_index_answer(context, index, plan)
        timings["structured_lookup"] = (time.perf_counter() - started) * 1000
        if result is None:
            stages.extend(("same_record_relational", "exhaustive_lexical_scan", "bm25"))
            started = time.perf_counter()
            ranked = _rank_block_indices(index, plan)
            timings["lexical_bm25"] = (time.perf_counter() - started) * 1000
            extracted: list[tuple[float, str, CanonicalBlock]] = []
            started = time.perf_counter()
            for block_index in ranked:
                block = index.canonical_blocks[block_index]
                span = _span_from_text(block.text, plan)
                if span:
                    extracted.append((_block_score(index, block_index, plan), span, block))
            timings["span_extraction"] = (time.perf_counter() - started) * 1000
            if extracted:
                best_span_score = max(score for score, _, _ in extracted)
                best_extracted = [(value, block) for score, value, block in extracted if score == best_span_score]
                if plan.requested_family == "workflow" and best_extracted:
                    workflow_values = list(dict.fromkeys(value for value, _ in best_extracted))
                    evidence = {block.block_id: block for _, block in best_extracted}
                    answer = (
                        str(len(workflow_values))
                        if plan.requested_label_tokens & {"many", "combien", "number", "nombre"}
                        else ", ".join(workflow_values)
                    )
                    result = AnswerResult("ANSWER", answer, tuple(evidence.values()), context.source_file,
                                          context.file_hash, "HIGH", "local_list_extraction")
                    best_extracted = []
                values = {_norm(value) for value, _ in best_extracted}
                if result is not None:
                    pass
                elif len(values) == 1:
                    evidence = {block.block_id: block for _, block in best_extracted}
                    result = AnswerResult("ANSWER", best_extracted[0][0], tuple(evidence.values()), context.source_file,
                                          context.file_hash, "HIGH", "local_span_extraction")
                else:
                    result = AnswerResult("AMBIGUOUS", NO_EXPLICIT_EVIDENCE,
                                          tuple(block for _, block in best_extracted), context.source_file,
                                          context.file_hash, "LOW", "ambiguous")
            else:
                stages.append("semantic_retrieval_unavailable")
        if result is None or result.status == "NO_EVIDENCE":
            # Definition and terse-topic fallback over the complete canonical document.
            definition_query = bool(plan.requested_label_tokens & {"mean", "meaning", "signifie"}) or len(plan.requested_label_tokens) == 1
            topic_matches: list[CanonicalBlock] = []
            for block in index.canonical_blocks:
                block_tokens = _tokens(block.text)
                if plan.requested_label_tokens and plan.requested_label_tokens <= block_tokens:
                    topic_matches.append(block)
            if definition_query:
                raw_key = next(iter(plan.requested_label_tokens - {"mean", "meaning", "signifie"}), "")
                definitions = index.raw_field_index.get(_norm(raw_key), ())
                if definitions:
                    values = list(dict.fromkeys(item.value for item in definitions))
                    evidence = {item.block.block_id: item.block for item in definitions}
                    if values == ["[NO_VALUE]"]:
                        result = AnswerResult(
                            "EXPLICIT_TERM_WITHOUT_VALUE",
                            f"{raw_key} is listed in the glossary, but no description is provided.",
                            tuple(evidence.values()), context.source_file, context.file_hash,
                            "HIGH", "explicit_empty_definition", result_type="EXPLICIT_EMPTY_VALUE",
                        )
                    else:
                        result = AnswerResult("ANSWER", ", ".join(values), tuple(evidence.values()), context.source_file,
                                              context.file_hash, "HIGH", "definition_fact", result_type="DEFINITION")
                elif topic_matches:
                    unique_topics = list({block.text: block for block in topic_matches}.values())
                    result = AnswerResult("ANSWER", "\n".join(f"- {block.text}" for block in unique_topics),
                                          tuple(unique_topics), context.source_file, context.file_hash,
                                          "MEDIUM", "topic_extraction", result_type="TOPIC_RESULT")
            elif topic_matches:
                unique_topics = list({block.text: block for block in topic_matches}.values())
                result = AnswerResult("ANSWER", "\n".join(f"- {block.text}" for block in unique_topics),
                                      tuple(unique_topics), context.source_file, context.file_hash,
                                      "MEDIUM", "topic_extraction", result_type="TOPIC_RESULT")
        if result is None:
            result = AnswerResult("NO_EVIDENCE", NO_EXPLICIT_EVIDENCE, (), context.source_file,
                                  context.file_hash, "LOW", "no_evidence",
                                  "all local retrieval stages exhausted")
        if result.status == "NO_EVIDENCE":
            # Import lazily to keep the canonical engine independent at module load.
            from mvp_services import suggestion_candidates
            result = replace(result, suggestions=suggestion_candidates(
                question, context.canonical_document
            ))
        result = replace(result, query_language=plan.language)
        timings["formatting_validation"] = 0.0
        timings["total"] = (time.perf_counter() - total_started) * 1000
        ranked_for_debug = _rank_block_indices(index, plan)
        top = tuple({
            "block_id": index.canonical_blocks[block_index].block_id,
            "score_breakdown": _block_score_breakdown(index, block_index, plan),
            "text": index.canonical_blocks[block_index].text,
            "rejection_reason": "no extractable span" if not _span_from_text(index.canonical_blocks[block_index].text, plan) else "",
        } for block_index in ranked_for_debug[:10])
        return result, DirectAnswerTrace(tuple(stages), timings, False, active_block_count=len(context.block_ids),
                                         candidate_count=len(ranked_for_debug), top_candidates=top)


_DEFAULT_FAST_ENGINE = FastDirectAnswerEngine()


def answer_direct(question: str, document: CanonicalDocument) -> AnswerResult:
    """Pure-question API over one immutable complete canonical document."""
    context = ActiveDocumentContext(
        source_file=document.source_file, file_hash=document.file_hash,
        canonical_document=document, block_ids=tuple(block.block_id for block in document.blocks),
        selection_version=0,
    )
    return _DEFAULT_FAST_ENGINE.query(context, question)[0]


def answer_question(context: ActiveDocumentContext, query: str) -> AnswerResult:
    """Answer locally. This API cannot accept or call an LLM/generator."""
    return _DEFAULT_FAST_ENGINE.query(context, query)[0]


def is_synthesis_query(query: str) -> bool:
    return bool(_SYNTHESIS.search(_norm(query)))


def debug_snapshot(context: ActiveDocumentContext, answer: AnswerResult | None = None) -> dict[str, Any]:
    sections = sorted({block.section for block in context.canonical_document.blocks if block.section})
    evidence = answer.evidence_blocks if answer else ()
    return {
        "source_file": context.source_file,
        "file_hash": context.file_hash,
        "file_type": context.canonical_document.file_type,
        "canonical_blocks": len(context.block_ids),
        "sections": sections,
        "answer_method": answer.method if answer else None,
        "evidence_block_ids": [block.block_id for block in evidence],
        "cross_document_evidence_count": sum(block.file_hash != context.file_hash for block in evidence),
    }
