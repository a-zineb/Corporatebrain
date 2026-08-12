# Corporate Brain — React/FastAPI migration report

Date: 2026-08-11

## Summary

The Streamlit MVP remains operational. A thin FastAPI adapter and a strict
React/TypeScript frontend were added alongside it. Existing normalization,
Direct Answer, hybrid retrieval, prompt, Ollama and citation functions were
preserved rather than reimplemented.

The preceding UI stability hotfix also centralizes Streamlit overlay state,
keeps only one dialog, moves source inspection into an inline panel, caches
Find Me state and limits initial result rendering to ten humanized cards.

## Files created

- `backend/main.py`, `backend/schemas.py`, `backend/services/runtime.py`
- `frontend/` Vite React application, API client, components, pages and styles
- `tests/test_fastapi_adapter.py`
- `MIGRATION_REPORT.md`

## Files modified

- `app.py`, `mvp_services.py`, `ui_components.py`
- `requirements.txt`, `README.md`
- `tests/test_mvp_productization.py`

No files were removed. Streamlit remains a runtime dependency.

## Backend functions preserved

- `canonical_rag.answer_direct`
- `rag_pipeline.hybrid_search`
- `rag_pipeline.build_source_list`
- `rag_pipeline.build_production_prompt`
- `rag_pipeline.stream_generate`
- `rag_pipeline.select_display_sources`
- `document_normalizer.normalize_document` through `PreparedDocumentRegistry`

## API endpoints

- `GET /api/health`
- `GET /api/filters`
- `GET /api/documents`
- `POST /api/documents/upload`
- `DELETE /api/documents/{file_hash}`
- `POST /api/search`
- `GET /api/sources/{file_hash}/{block_id}`
- `POST /api/chat` (`direct` or `ai`)

## React components and pages

- Animated/collapsible `Sidebar` and `AppLayout`
- `ChatComposer`, `SourceCard`, `SourcePanel`
- `DocumentCard`
- Chat, Documents, Search, History and Settings pages
- Dark/light/system themes persisted in local storage
- Responsive/mobile layout and reduced-motion accessibility

## Verified features

- FastAPI imports and schemas compile.
- Health, filters, real document listing, global search and source round-trip.
- Direct chat requires and respects one selected document.
- Streamlit Find Me searches globally without using chat routing.
- Only one Streamlit dialog remains.
- Source opening uses cached `(file_hash, block_id)` navigation.
- Full Python suite: 320 passed; two pre-existing Chroma-manifest failures.

## Not yet fully migrated or certified

- React production build: Node.js/npm is not installed on this workstation.
- Live browser checks, console audit and responsive visual acceptance.
- Live Ollama/Qwen response through `/api/chat` in `ai` mode.
- Conversation persistence: the current backend has no durable conversation store.
- Admin analytics, authentication, favorites and collections were not invented.
- ZIP upload through FastAPI is not advertised until API parity is tested; the
  Streamlit ingestion flow continues to support ZIP.
- Backend metadata-filter application needs browser/API parity validation;
  Streamlit remains the authoritative filtered experience meanwhile.

## Commands

Backend:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

Frontend, after installing Node.js LTS:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Legacy Streamlit:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

## Architecture

```text
React/TypeScript ──HTTP──> FastAPI thin adapter
                              │
Streamlit fallback ───────────┤
                              ▼
               Existing canonical/RAG services
                              │
          ChromaDB + BM25/RRF + MiniLM + Ollama
```

## Streamlit uninstall decision

No. Streamlit cannot safely be uninstalled yet. Remove it only after Node-based
build validation and the complete manual parity checklist pass against real
Direct Answer, AI Answer, filters, citations, uploads and source navigation.

