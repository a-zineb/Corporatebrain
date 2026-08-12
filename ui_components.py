"""Reusable Corporate Brain Streamlit presentation components."""

from __future__ import annotations

import html
import io
import os
import re
import time
from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

import canonical_rag
from document_normalizer import CanonicalBlock, CanonicalDocument
from mvp_services import GlobalSearchHit, SourceTarget


DESIGN_CSS = """
<style>
:root { --cb-primary:#3157d5; --cb-ink:#172033; --cb-muted:#687386; --cb-card:#fff; }
.stApp { font-family: Inter, "Segoe UI", system-ui, sans-serif; color:var(--cb-ink); }
h1 { font-size:30px !important; letter-spacing:-.02em; }
h2, h3 { font-size:22px !important; }
[data-testid="stChatMessage"] { border:1px solid #e7eaf0; border-radius:16px; padding:.35rem .8rem; background:#fff; box-shadow:0 4px 18px rgba(23,32,51,.045); }
[data-testid="stChatMessage"] p { font-size:16px; line-height:1.55; }
.cb-source { border:1px solid #e1e6ef; border-radius:12px; padding:.75rem; margin:.4rem 0; background:#f8fafc; font-size:14px; }
.cb-meta { color:var(--cb-muted); font-size:12px; }
.cb-status { display:inline-block; border:1px solid #ccd5e3; border-radius:999px; padding:.16rem .55rem; font-size:12px; font-weight:650; }
.cb-highlight { background:#fff1a8; border-left:4px solid #e7ae21; padding:.7rem; border-radius:6px; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
</style>
"""


LABELS = {
    "English": {"source": "Source passage", "open": "Open source", "more": "See more",
                "multiple": "I found several matching values.", "suggest": "Did you mean:",
                "none": "I found no explicit evidence answering this question in the selected document."},
    "French": {"source": "Passage source", "open": "Ouvrir la source", "more": "Voir plus",
               "multiple": "J’ai trouvé plusieurs valeurs correspondantes.", "suggest": "Voulez-vous dire :",
               "none": "Je n’ai trouvé aucune preuve explicite répondant à cette question dans le document sélectionné."},
}


def inject_design() -> None:
    st.markdown(DESIGN_CSS, unsafe_allow_html=True)


def render_document_status(state: str) -> None:
    readable = {"READY": "Ready", "READY_WITH_WARNINGS": "Ready with warnings", "FAILED": "Failed"}.get(state, state)
    st.markdown(f'<span class="cb-status">{html.escape(readable)}</span>', unsafe_allow_html=True)


def source_target(block: CanonicalBlock, document: CanonicalDocument | None = None) -> SourceTarget:
    return SourceTarget.from_block(block, document.file_type if document else "")


def _pairs(text: str) -> list[tuple[str, str]]:
    return [(left.strip(), right.strip()) for left, right in re.findall(
        r"(?:^|\|)\s*([^=|:]+?)\s*(?:=|:)\s*([^|]+)", text
    )]


def structured_rows(blocks: Sequence[CanonicalBlock]) -> list[dict[str, str]]:
    rows = []
    for block in blocks:
        pairs = _pairs(block.text)
        if pairs:
            rows.append(dict(pairs))
    return rows


def render_table_answer(blocks: Sequence[CanonicalBlock]) -> None:
    rows = structured_rows(blocks)
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        st.dataframe(pd.DataFrame(rows, columns=columns), hide_index=True, use_container_width=True)


def render_multi_value_answer(result: canonical_rag.AnswerResult) -> None:
    st.caption(LABELS[result.query_language]["multiple"])
    values = [line.removeprefix("- ").strip() for line in result.answer.splitlines()
              if line.strip() and not line.rstrip().endswith(":")]
    for value in values:
        st.markdown(f"- {value}")


def render_section_answer(result: canonical_rag.AnswerResult) -> None:
    rows = structured_rows(result.evidence_blocks)
    if rows:
        render_table_answer(result.evidence_blocks)
    else:
        st.markdown(result.answer)


def render_source_card(target: SourceTarget, key: str, language: str = "English") -> None:
    labels = LABELS[language]
    st.markdown(
        f'<div class="cb-source"><strong>{html.escape(labels["source"])}</strong><br>'
        f'<span class="cb-meta">{html.escape(target.source_file)} · {html.escape(target.location_label)}</span><br>'
        f'{html.escape(target.evidence_text[:320])}</div>', unsafe_allow_html=True,
    )
    if st.button(labels["open"], key=key):
        st.session_state.pending_source_target = target
        st.session_state.active_overlay = "SOURCE_VIEWER"
        st.rerun()


def render_source_viewer(document: CanonicalDocument, target: SourceTarget,
                         source_path: str | None = None) -> None:
    top_left, top_right = st.columns([1, 5])
    with top_left:
        if st.button("← Back", use_container_width=True):
            st.session_state.active_overlay = st.session_state.get("source_return_overlay", "NONE")
            st.session_state.pending_source_target = None
            st.rerun()
    with top_right:
        st.subheader(document.source_file)
    st.caption(target.location_label)
    if (document.file_type == "pdf" and source_path and os.path.isfile(source_path)
            and target.page is not None):
        import fitz
        from PIL import Image, ImageDraw

        pdf = fitz.open(source_path)
        try:
            page = pdf.load_page(max(0, target.page - 1))
            scale = 1.6
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            rendered = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            if target.bbox:
                draw = ImageDraw.Draw(rendered)
                draw.rectangle(tuple(value * scale for value in target.bbox), outline="#e6a700", width=5)
            st.image(rendered, use_container_width=True)
        finally:
            pdf.close()
    blocks = list(document.blocks)
    selected_index = next((index for index, block in enumerate(blocks) if block.block_id == target.block_id), 0)
    start, end = max(0, selected_index - 2), min(len(blocks), selected_index + 3)
    for block in blocks[start:end]:
        css = "cb-highlight" if block.block_id == target.block_id else "cb-source"
        st.markdown(f'<div class="{css}">{html.escape(block.text)}</div>', unsafe_allow_html=True)
    previous_col, next_col = st.columns(2)
    with previous_col:
        if st.button("← Previous", disabled=selected_index == 0, use_container_width=True):
            st.session_state.pending_source_target = SourceTarget.from_block(blocks[selected_index - 1], document.file_type)
            st.rerun()
    with next_col:
        if st.button("Next →", disabled=selected_index >= len(blocks) - 1, use_container_width=True):
            st.session_state.pending_source_target = SourceTarget.from_block(blocks[selected_index + 1], document.file_type)
            st.rerun()


def render_suggestions(suggestions: Sequence[str], language: str, key_prefix: str) -> str | None:
    if not suggestions:
        return None
    st.caption(LABELS[language]["suggest"])
    selected = None
    columns = st.columns(min(len(suggestions), 3))
    for index, suggestion in enumerate(suggestions):
        with columns[index % len(columns)]:
            if st.button(suggestion, key=f"{key_prefix}_{index}", use_container_width=True):
                selected = suggestion
    return selected


def render_see_more(result: canonical_rag.AnswerResult, latency_ms: float = 0.0) -> None:
    with st.expander(LABELS[result.query_language]["more"]):
        st.caption(f"Document: {result.source_file}")
        st.caption(f"Hash: {result.file_hash[:12]}…")
        st.caption(f"Method: {result.method} · Type: {result.result_type}")
        st.caption(f"Latency: {latency_ms:.1f} ms · Language: {result.query_language}")
        if result.reason:
            st.caption(f"Warning: {result.reason}")


def render_answer(result: canonical_rag.AnswerResult, *, latency_ms: float = 0.0,
                  document: CanonicalDocument | None = None, key_prefix: str = "answer") -> str | None:
    labels = LABELS[result.query_language]
    if result.status == "SENSITIVE_BLOCK":
        st.warning("This sensitive value cannot be displayed." if result.query_language == "English"
                   else "Cette valeur sensible ne peut pas être affichée.")
    elif result.status == "NO_EVIDENCE":
        st.markdown(labels["none"])
    elif result.result_type in {"TABLE_RESULT"}:
        render_table_answer(result.evidence_blocks)
    elif result.result_type == "SECTION_RESULT":
        render_section_answer(result)
    elif result.result_type in {"MULTI_VALUE", "MULTI_MENTION"}:
        render_multi_value_answer(result)
    elif result.answer:
        st.markdown(result.answer)
    for index, block in enumerate(result.evidence_blocks):
        render_source_card(source_target(block, document), f"{key_prefix}_source_{index}", result.query_language)
    selected = render_suggestions(result.suggestions, result.query_language, f"{key_prefix}_suggest")
    render_see_more(result, latency_ms)
    return selected


def render_global_search_result_card(hit: GlobalSearchHit, result_index: int,
                                     language: str = "English") -> None:
    target = hit.target
    st.markdown(
        '<div class="cb-source">'
        f'<span class="cb-status">{html.escape(hit.document.file_type.upper())}</span> '
        f'<strong>{html.escape(hit.document.source_file)}</strong><br>'
        f'<span class="cb-meta">{html.escape(target.location_label)}</span><br>'
        f'<strong>{html.escape(hit.display_title or hit.matched_topic)}</strong><br>'
        f'{html.escape(hit.relation)}: {html.escape(hit.display_value)}'
        '</div>', unsafe_allow_html=True,
    )
    with st.expander("Show passage" if language == "English" else "Afficher le passage"):
        st.caption(hit.preview)
    key = f"open_source_{hit.document.file_hash}_{target.block_id}_{result_index}"
    if st.button(LABELS[language]["open"], key=key):
        st.session_state.pending_source_target = target
        st.session_state.source_return_overlay = "FIND_ME"
        st.session_state.active_overlay = "SOURCE_VIEWER"
        st.rerun()


@st.dialog("Find me: where", width="large")
def render_find_me(registry, language: str = "English") -> None:
    query = st.text_input(
        "Search", placeholder="Search across all documents..." if language == "English" else "Rechercher dans tous les documents…",
        label_visibility="collapsed", value=st.session_state.get("find_me_query", ""),
    )
    if not query:
        st.caption("All prepared documents are searched. Your active document will not change.")
        return
    if query != st.session_state.get("find_me_query"):
        st.session_state.find_me_query = query
        st.session_state.find_me_results = registry.global_search(query, limit=200)
        st.session_state.find_me_page = 0
    hits: Sequence[GlobalSearchHit] = st.session_state.get("find_me_results", ())
    if not hits:
        st.info(LABELS[language]["none"])
        return
    locations = {(hit.document.file_hash, hit.target.sheet or hit.target.section) for hit in hits}
    st.markdown(f"**{len(hits)} relevant matches across {len(locations)} document sections.**")
    counts: dict[str, int] = {}
    for hit in hits:
        value = hit.display_value.strip()
        if value:
            counts[value] = counts.get(value, 0) + 1
    common = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))[:4]
    if common:
        st.caption("Most common: " + " · ".join(f"{value} ({count})" for value, count in common))
    page_size = 10
    pages = max(1, (len(hits) + page_size - 1) // page_size)
    page = min(int(st.session_state.get("find_me_page", 0)), pages - 1)
    start = page * page_size
    for index, hit in enumerate(hits[start:start + page_size], start=start):
        render_global_search_result_card(hit, index, language)
    st.caption(f"{start + 1}–{min(start + page_size, len(hits))} of {len(hits)}")
    previous, next_page = st.columns(2)
    with previous:
        if st.button("Previous", disabled=page == 0, use_container_width=True):
            st.session_state.find_me_page = page - 1
            st.rerun()
    with next_page:
        if st.button("Next", disabled=page >= pages - 1, use_container_width=True):
            st.session_state.find_me_page = page + 1
            st.rerun()


def render_typing_indicator(mode: str, language: str) -> None:
    if mode == "AI answer":
        text = "Generating answer..." if language == "English" else "Génération de la réponse…"
    else:
        text = "Corporate Brain is searching..." if language == "English" else "Corporate Brain recherche…"
    st.caption(text)
