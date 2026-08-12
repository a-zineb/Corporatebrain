# Corporate Brain — Complete frontend makeover report

Date: 2026-08-11

## 1–6. Architecture and backend compatibility

The previous React frontend used one left navigation rail and a wide single
content column. Chat selected documents through a native select box; Documents
and Search were separate pages. The new architecture is a three-column AI
workspace: collapsible navigation, centered readable chat, and a collapsible
real-document panel. Streamlit remains available as fallback.

No answer or retrieval algorithm was rewritten. The frontend reuses the
FastAPI adapter over `canonical_rag.answer_direct`, canonical preparation,
global search, hybrid RAG, Ollama generation and citation selection. Only two
small file adapters were added for original-file open/download and first-source
inspection.

Files added include `DocumentsPanel.tsx`, `AnswerContent.tsx` and this report.
Modified frontend files cover Sidebar, ChatPage, HistoryPage, SearchPage,
ChatComposer, API client and design tokens. Backend changes are limited to
safe document file/source endpoints and ZIP reload support.

## 7–10. Sidebars and documents

- Left sidebar: brand, real global-search input, New Chat, History, Settings,
  appearance entry and smooth persisted collapse.
- Right panel: actual prepared-document count/cards, persisted collapse, mobile
  drawer behavior, PDF/Word/Excel/CSV/ZIP filter chips and independent scroll.
- Card body selects a document; eye opens the cached internal source viewer;
  download retrieves the exact stored original with attachment disposition.
- Composer and panel uploads call the real ingestion endpoint.

## 11–16. Chat, composer and visual design

- Compact 860px chat, user right/assistant left, source groups collapsed by
  default, genuine structured tables and multi-value lists.
- Segmented Direct/Catalog/AI and Document/All controls.
- Composer attachment workflow: upload → prepare → select → archive previous
  conversation → start new document chat.
- Dark translucent panels use restrained green/teal glass; ambient bottom glow,
  slow orb motion and composer glow respect reduced-motion preferences.
- Light/dark/system theme tokens remain persisted through Settings.

## 17–21. Search, history and answer sources

- Top-left search calls `/api/search` directly and never enters chat routing.
- Search provides real result counts, common-value summary, 12-card pagination
  and exact source opening without changing active chat document.
- New Chat archives actual messages in session storage while preserving the
  selected document. History cards restore those messages and document.
- Source navigation continues to use the cached `(file_hash, block_id)` map.
- Table/section responses use a horizontally scrollable semantic table when
  structured source rows exist; multi-values use compact list cards.

## 22–24. Responsive behavior, animation and accessibility

The desktop layout is three columns. Medium width turns Documents into an
overlay drawer/rail. Mobile hides document chrome to protect chat width and
uses the existing sidebar drawer. Animations use opacity/transform/width with
reduced-motion overrides. Controls have labels/titles, visible selected states,
semantic buttons, form submission and keyboard-friendly chat entry.

## 25–26. Validation

- Strict TypeScript and Vite production build: pass, 1,817 modules.
- Focused API/UI/backend regression: 33 passed.
- FastAPI adapter including original open/download: 5 passed.
- Full Python regression: 321 passed, two known environment-contract failures.
- Known failures are unchanged: active Chroma has 3,165 chunks while the
  committed certified manifest expects 1,072.
- `./run` health checks previously validated API and UI HTTP 200; the build run
  by that launcher remains enabled by default.

Manual browser acceptance still needs a human visual pass for exact reference
similarity and one live Ollama/Qwen AI Answer session. Automated checks cannot
certify subjective appearance or a locally running external model.

## 27–28. Limitations and replacement decision

- UI chrome is English; backend response language still follows each message.
- History is session-scoped because no durable conversation backend exists.
- The Chroma benchmark manifest mismatch remains unrelated to this makeover.
- Authentication, favorites and fake collections were intentionally not added.

The React frontend is ready for visual/user acceptance testing and normal
Direct/Search/Documents use. Keep Streamlit installed until a live AI Answer,
manual responsive pass and all acceptance checklist interactions are signed off.

