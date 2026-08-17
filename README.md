# Corporate Brain

Corporate Brain is an enterprise RAG application for deterministic document
answers, local hybrid retrieval and Gemini API synthesis with cited sources.

## One-command startup

From PowerShell at the project root, run:

```powershell
./run
```

That single command:

1. creates the Python venv if it is missing;
2. synchronizes `requirements.txt` inside the venv;
3. installs Node.js LTS through `winget` if necessary;
4. creates `frontend/.env.local` from the example;
5. runs `npm install`;
6. executes the strict TypeScript/Vite production build;
7. starts FastAPI on port 8000 and React on port 5173;
8. checks both services, then opens the application in the browser;
9. stops both servers when you press `Ctrl+C`.

For a faster development restart after a previously successful build:

```powershell
./run -SkipBuild
```

To avoid opening a browser automatically, add `-NoBrowser`.

## Architecture

```text
React + TypeScript (frontend/)
          ↓ HTTP JSON
FastAPI adapter (backend/)
          ↓ direct function calls
canonical_rag.py / rag_pipeline.py / document_normalizer.py
          ↓
ChromaDB + BM25/RRF + MiniLM (local retrieval)
          ↓ selected evidence only
Gemini API through the official Google Gen AI SDK (AI Answer generation)

Streamlit app.py remains available during migration and uses the same services.
```

FastAPI is an adapter over the canonical document engine. Direct Answer remains
deterministic. AI Answer classifies the query intent, resolves same-document
follow-ups, saturates structured/exhaustive retrieval over the selected
document, and asks the configured Gemini model to synthesize the application-selected evidence. Gemini returns conversational Markdown;
source cards remain application-controlled. AI Answer searches all prepared
documents by default and does not require a selected file. Direct Answer remains
strictly selected-document only, and Knowledge Catalog makes no Gemini call.

## Features

- PDF, DOCX, DOC, XLSX, CSV and ZIP ingestion in the existing Streamlit flow
- Deterministic selected-document Direct Answer without an API call
- Hybrid vector/BM25 retrieval with RRF and API-backed generation
- Global prepared-document search and exact canonical source navigation
- OCM/OEG/OJO/OCI and MZ/KPSA metadata values preserved
- React dark/light/system themes and collapsible responsive sidebar
- Clerk authentication and user-scoped persistent conversation history
- Streamlit fallback retained until React parity is manually certified

## Backend development

From PowerShell, activate the existing environment before installing anything:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

Health check: `http://localhost:8000/api/health`

Optional environment variables:

- `CORPORATE_BRAIN_STORAGE_DIR`
- `CORPORATE_BRAIN_CHROMA_PATH`
- `CORPORATE_BRAIN_COLLECTION`
- `AI_PROVIDER`
- `GEMINI_MODEL`
- `GEMINI_TIMEOUT_SECONDS`
- `GEMINI_TEMPERATURE`
- `GEMINI_TOP_P`
- `GEMINI_MAX_OUTPUT_TOKENS`
- `GEMINI_ENABLE_STREAMING`

## Gemini API setup

1. Create a Gemini API key in Google AI Studio.
2. Copy `.env.example` to a root `.env` and set `GEMINI_API_KEY=...`.
3. Keep `AI_PROVIDER=gemini`; the default model is `gemini-3.6-flash`.
4. Never place this secret in `frontend/.env.local`, a `VITE_` variable, source
   code, or browser storage.
5. Restart Corporate Brain with `./run` and verify AI Answer.

If the key is missing, rejected, rate-limited, or the network times out, AI
Answer reports a safe provider error. Direct Answer, catalog/search, graph, and
source navigation remain available. The backend never returns the key; Settings
shows only whether the provider is configured.

## Frontend development

Install a current Node.js LTS release, then:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

The frontend reads `VITE_API_BASE_URL`; no backend secrets are sent to React.

Production build:

```powershell
cd frontend
npm run build
```

## Legacy Streamlit application

The working application remains available during migration:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Do not uninstall Streamlit until chat, AI generation, citations, filters,
uploads, deletion and source navigation pass the browser acceptance checklist.

## API endpoints

- `GET /api/health`
- `GET /api/filters`
- `GET /api/documents`
- `POST /api/documents/upload`
- `POST /api/documents/upload-async`
- `GET /api/ingestion/jobs`
- `POST /api/ingestion/jobs/{job_id}/retry`
- `POST /api/documents/{file_hash}/reindex`
- `DELETE /api/documents/{file_hash}`
- `GET /api/documents/{file_hash}/content` (registered files only; no client path)
- `GET /api/documents/{file_hash}/preview` (cached rendered DOCX PDF)
- `GET /api/documents/{file_hash}/preview-info?block_id=...`
- `GET /api/documents/{file_hash}/table?sheet=...`
- `POST /api/search`
- `GET /api/sources/{file_hash}/{block_id}`
- `POST /api/chat` with `mode: direct | ai`
- `GET|PUT|PATCH|DELETE /api/conversations/{conversation_id}`

Spreadsheet citations expose the exact sheet and cell range. CSV citations
expose the original row number. PDF citations open the registered original at
the one-based PDF page using `#page=N`; arbitrary filesystem paths are never
accepted by the API.

DOCX evidence previews are generated lazily with Microsoft Word on Windows and
cached under `.run/previews` by document checksum. The original DOCX remains
immutable and is served separately. If Word automation is unavailable, the API
returns a controlled 503 response and the UI keeps the original-file action.

Asynchronous ingestion reports real pipeline stages (`uploading`, `extracting`,
`normalizing`, `chunking`, `embedding`, `indexing`, then `ready`, `warning`, or
`failed`). Retry reuses the retained upload payload. Re-index deletes only the
selected document's old Chroma records by checksum/source identity before
adding stable block IDs and rebuilding BM25.

## Graph View

Graph View uses bounded 18–46 px nodes, collision-aware force layout, a
significance threshold, and at most three strongest edges per document. It
supports approximate search, file-type filtering, Global/Focused/Community
modes, zoom/reset/fullscreen controls, tooltips, relation explanations, and a
document detail panel. Run its deterministic tests with:

```powershell
cd frontend
npm test
```

## Tests

```powershell
.\venv\Scripts\Activate.ps1
python -m pytest -q
```

Document and Chroma data remain local and must not be committed.

### Chroma benchmark and re-indexing

The committed `corporatebrain.v1` benchmark is certified against its immutable
1,072-chunk corpus. The development upload folder can legitimately produce a
different count and must not be made to pass by editing that manifest.

Before rebuilding a development index, back up `doc_storage_v2` and
`chroma_db_local_v2`, remove unintended duplicate uploads through the
application, and start a fresh collection name:

```powershell
$env:CORPORATE_BRAIN_COLLECTION="documents-schema-v3"
./run.ps1
```

Keep the old collection until representative searches and citations have been
verified. A parser/schema change requires a new collection name or an explicit
full re-index; mixing chunks produced by different parser versions is not a
certified state.

## Authentication with Clerk

Corporate Brain uses Clerk in the React/Vite frontend and verifies Clerk session
JWTs in FastAPI. The frontend receives only a publishable key. The backend uses
the instance's public JWKS endpoint; this implementation does not require or
expose `CLERK_SECRET_KEY`.

### 1. Create the Clerk application

Create an account and application in the Clerk Dashboard. On the **API keys**
page, copy the Publishable Key and the Frontend API URL. The current frontend
dependency is `@clerk/react`; backend verification uses `PyJWT[crypto]`.

### 2. Install dependencies

Keep the Python virtual environment active for Python packages:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 3. Configure the frontend

Copy `frontend/.env.example` to `frontend/.env.local`, then set:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_CLERK_PUBLISHABLE_KEY=pk_test_your_publishable_key
```

Restart Vite after changing this file. An empty key intentionally displays an
authentication-configuration screen instead of a fake user profile.

### 4. Configure the backend

Copy `.env.example` values into the server environment (PowerShell example):

```powershell
$env:CLERK_JWKS_URL="https://your-frontend-api/.well-known/jwks.json"
$env:CLERK_AUTHORIZED_PARTIES="http://localhost:5173,http://127.0.0.1:5173"
```

`CLERK_JWKS_URL` is the Clerk Frontend API URL followed by
`/.well-known/jwks.json`. `CLERK_AUTHORIZED_PARTIES` is a comma-separated allow
list checked against the token's `azp` claim. If `CLERK_JWKS_URL` is absent, the
backend retains the documented local-development boundary for compatibility;
production deployments must set it.

### 5. Configure the Clerk Dashboard

Allow `http://localhost:5173` and `http://127.0.0.1:5173` during development.
Add the real HTTPS production origin before deployment. This app uses Clerk's
modal sign-in/sign-up flows and returns to `/`, so no custom callback route is
required. Configure production sign-in/sign-up URLs only if replacing those
modals with hosted pages.

### 6. Start the app

From the repository root:

```powershell
./run
```

For troubleshooting, start FastAPI with
`python -m uvicorn backend.main:app --reload --port 8000` and Vite with
`cd frontend; npm run dev` in separate terminals.

### 7. Test authentication

Open the frontend. Chat, Direct Answer and AI Answer work while signed out.
Optionally create an account or sign in, open **Profile** from the
bottom-left card, start a document chat, verify it appears under **Recent** and
**History**, reopen it, then sign out with Clerk's user button. A different
Clerk account must not see the first account's conversations.

### 8. Common errors

- **Missing publishable key:** set `VITE_CLERK_PUBLISHABLE_KEY` and restart Vite.
- **Wrong origin:** add the exact localhost or production origin in Clerk.
- **401 from FastAPI:** confirm the browser sends `Authorization: Bearer ...`,
  the JWKS URL belongs to the same Clerk instance, and `azp` is allowed.
- **Invalid/expired session:** sign out/in and verify the system clock.
- **“Invalid or expired session”:** restart Vite after `.env.local` changes,
  verify the publishable key and backend JWKS belong to the same Clerk
  application, and confirm requests can obtain a current bearer token. A stale
  or invalid token now falls back to signed-out mode and cannot block RAG.
- **`.env.local` not loaded:** keep it inside `frontend/` and restart Vite.
- **Backend environment ignored:** set variables in the terminal that launches
  FastAPI (or load them through your deployment platform).

### 9. Security

Never commit `.env.local`, never expose `CLERK_SECRET_KEY` to Vite, and never
place server secrets in variables prefixed with `VITE_`. Publishable keys are
safe for frontend use. Revoke and rotate any credential that is accidentally
leaked. Conversation records are scoped by the verified Clerk `sub` user ID.
