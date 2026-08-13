from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
import unicodedata
from typing import Any, Callable, Iterable

from document_normalizer import CanonicalBlock, CanonicalDocument
from mvp_services import detect_query_language


INTENTS = {
    "SINGLE_FACT", "MULTI_FACT", "EXHAUSTIVE_LIST", "SECTION_SUMMARY",
    "TABLE_QUERY", "EXPLANATION", "COMPARISON", "FOLLOW_UP",
}

ALIASES = {
    "directory": {"directory", "repertoire", "folder", "path", "chemin"},
    "collection": {"collection", "collecte", "input", "entree"},
    "distribution": {"distribution", "delivery", "livraison", "output", "sortie"},
    "frequency": {"frequency", "frequence", "periodicity", "periodicite"},
    "format": {"format", "forme"},
    "host": {"host", "hostname", "serveur", "server", "ip"},
    "protocol": {"protocol", "protocole"},
    "sheet": {"sheet", "sheets", "worksheet", "worksheets", "feuille", "feuilles", "onglet", "onglets"},
}


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9_.:/-]+", value))


def classify_intent(query: str) -> str:
    q = _norm(query)
    if re.search(r"\b(and|et|also|aussi|its?|son|sa|ses|leur|what about|et pour)\b", q) and len(q.split()) <= 8:
        return "FOLLOW_UP"
    if re.search(r"\b(all|every|everything|complete|entire|tous|toutes|tout|chaque|liste|list)\b", q):
        return "EXHAUSTIVE_LIST"
    if any(word in q for word in ("sheet", "feuille", "onglet", "table", "row", "ligne")):
        return "TABLE_QUERY"
    if any(word in q for word in ("compare", "comparison", "difference", "comparer", "comparaison")):
        return "COMPARISON"
    if any(word in q for word in ("explain", "overview", "describe", "resume", "résume", "explique")):
        return "EXPLANATION"
    if any(word in q for word in ("section", "chapter", "chapitre")):
        return "SECTION_SUMMARY"
    if re.search(r"\b(and|et|plus|ainsi que)\b", q):
        return "MULTI_FACT"
    return "SINGLE_FACT"


def rewrite_follow_up(query: str, history: list[dict[str, str]], document_hash: str) -> str:
    """Resolve short references, but never import context from another document."""
    if classify_intent(query) != "FOLLOW_UP":
        return query
    relevant = [item for item in history[-8:] if item.get("document_hash") == document_hash]
    if not relevant:
        return query
    prior = next((item.get("content", "") for item in reversed(relevant)
                  if item.get("role") == "user" and item.get("content", "").strip()), "")
    if not prior:
        return query
    entities = [token for token in re.findall(r"\b[A-Z][A-Z0-9_-]{1,}\b", prior) if token not in {"WHAT", "THE"}]
    if not entities:
        return query
    entity = entities[-1]
    return re.sub(r"\b(its?|son|sa|ses|leur)\b", entity, query, flags=re.I) + f" ({entity})"


def _expanded_tokens(query: str) -> set[str]:
    tokens = set(_norm(query).split())
    for values in ALIASES.values():
        if tokens & values:
            tokens.update(values)
    return tokens


def _score(block: CanonicalBlock, tokens: set[str]) -> float:
    haystack = _norm(" ".join(filter(None, [block.text, block.section, block.sheet])))
    words = set(haystack.split())
    exact = len(tokens & words)
    partial = sum(0.35 for token in tokens if token not in words and any(
        SequenceMatcher(None, token, word).ratio() >= .84 for word in words
    ))
    return exact + partial


def retrieve_evidence(document: CanonicalDocument, query: str, intent: str) -> list[CanonicalBlock]:
    """Retrieve canonical blocks; exhaustive intents saturate over the selected document."""
    tokens = _expanded_tokens(query)
    q = _norm(query)
    if document.file_type == "xlsx" and tokens & ALIASES["sheet"]:
        seen: set[str] = set()
        result: list[CanonicalBlock] = []
        for block in document.blocks:
            if block.sheet and block.sheet not in seen:
                seen.add(block.sheet)
                result.append(block)
        return result

    scored = [(_score(block, tokens), order, block)
              for order, block in enumerate(document.blocks)]
    relevant = [(score, order, block) for score, order, block in scored if score > 0]
    relevant.sort(key=lambda item: (-item[0], item[1]))

    if intent in {"EXHAUSTIVE_LIST", "TABLE_QUERY", "SECTION_SUMMARY"}:
        section_names = {_norm(block.section or "") for score, _, block in relevant[:8] if score >= 1}
        table_ids = {block.table_index for score, _, block in relevant[:8]
                     if score >= 1 and block.table_index is not None}
        saturated = [block for block in document.blocks if (
            _score(block, tokens) > 0
            or (block.section and _norm(block.section) in section_names)
            or (block.table_index is not None and block.table_index in table_ids)
        )]
        return saturated[:200]
    return [block for _, _, block in relevant[:12]]


def evidence_catalog(blocks: Iterable[CanonicalBlock]) -> str:
    return "\n\n".join(f"<evidence id={json.dumps(block.block_id)} location={json.dumps(_location(block))}>\n{block.text}\n</evidence>"
                         for block in blocks)


def _location(block: CanonicalBlock) -> str:
    parts = [block.source_file]
    if block.page is not None:
        parts.append(f"Page {block.page}")
    if block.sheet:
        parts.append(f"Sheet {block.sheet}")
    if block.row_index is not None:
        parts.append(f"Row {block.row_index}")
    if block.section:
        parts.append(f"Section {block.section}")
    return " · ".join(parts)


def build_grounded_prompt(query: str, language: str, intent: str,
                          blocks: list[CanonicalBlock]) -> str:
    detail = "include every explicit item" if intent == "EXHAUSTIVE_LIST" else "match detail to the question"
    return f"""You are Corporate Brain, a grounded enterprise assistant.
Return ONLY valid JSON with this schema:
{{"answer":"natural Markdown answer","claims":[{{"text":"factual claim","evidence_ids":["exact-id"]}}]}}

Rules:
- Answer in {language}. Be direct; omit filler and closing offers.
- Synthesize and paraphrase naturally. Do not copy long source passages.
- Use only the evidence below. Preserve proper nouns and technical values.
- {detail}. Use a Markdown table when records have several comparable fields.
- Every factual claim needs one or more exact evidence IDs. Never invent an ID.
- Do not put [SOURCE N] markers in the answer.

Intent: {intent}
Question: {query}

Evidence:
{evidence_catalog(blocks)}"""


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    evidence_ids: tuple[str, ...]
    claims: tuple[dict[str, Any], ...]
    repaired: bool = False


def _parse_generation(raw: str, allowed: set[str]) -> GroundedAnswer:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError("malformed model output")
    payload = json.loads(match.group(0))
    answer = str(payload.get("answer", "")).strip()
    claims = payload.get("claims", [])
    if not answer or not isinstance(claims, list):
        raise ValueError("missing answer or claims")
    used: list[str] = []
    for claim in claims:
        ids = claim.get("evidence_ids", []) if isinstance(claim, dict) else []
        if not ids or any(item not in allowed for item in ids):
            raise ValueError("invalid evidence citation")
        used.extend(ids)
    return GroundedAnswer(answer, tuple(dict.fromkeys(used)), tuple(claims))


def generate_grounded(query: str, document: CanonicalDocument, history: list[dict[str, str]],
                       generate: Callable[[str], str]) -> tuple[GroundedAnswer, list[CanonicalBlock], str, str]:
    language = detect_query_language(query)
    rewritten = rewrite_follow_up(query, history, document.file_hash)
    intent = classify_intent(query)
    blocks = retrieve_evidence(document, rewritten, intent)
    if not blocks:
        message = ("Je n’ai trouvé aucune preuve explicite dans le document sélectionné."
                   if language == "French" else
                   "I found no explicit evidence in the selected document.")
        return GroundedAnswer(message, (), ()), [], language, intent
    prompt = build_grounded_prompt(rewritten, language, intent, blocks)
    allowed = {block.block_id for block in blocks}
    expected = ([block.sheet for block in blocks if block.sheet]
                if document.file_type == "xlsx" and _expanded_tokens(query) & ALIASES["sheet"] else [])

    def validate(raw: str) -> GroundedAnswer:
        parsed = _parse_generation(raw, allowed)
        missing = [item for item in dict.fromkeys(expected) if item and _norm(item) not in _norm(parsed.answer)]
        if missing:
            raise ValueError("incomplete exhaustive answer; missing: " + ", ".join(missing))
        return parsed
    try:
        result = validate(generate(prompt))
    except (ValueError, json.JSONDecodeError) as exc:
        repair = prompt + f"\n\nYour previous output was rejected: {exc}. Return corrected JSON only."
        result = validate(generate(repair))
        result = GroundedAnswer(result.answer, result.evidence_ids, result.claims, repaired=True)
    by_id = {block.block_id: block for block in blocks}
    cited = [by_id[item] for item in result.evidence_ids]
    if intent == "EXHAUSTIVE_LIST" and not cited:
        raise ValueError("exhaustive answer did not cite evidence")
    return result, cited, language, intent
