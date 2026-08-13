# Corporate Brain

Corporate Brain is a local enterprise RAG application for deterministic document
answers, hybrid retrieval and Ollama/Qwen synthesis with cited sources.

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
ChromaDB + BM25/RRF + MiniLM + Ollama/Qwen

Streamlit app.py remains available during migration and uses the same services.
```

FastAPI is an adapter over the canonical document engine. Direct Answer remains
deterministic. AI Answer classifies the query intent, resolves same-document
follow-ups, saturates structured/exhaustive retrieval over the selected
document, and asks Ollama for validated JSON claims tied to canonical block IDs.

## Features

- PDF, DOCX, DOC, XLSX, CSV and ZIP ingestion in the existing Streamlit flow
- Deterministic selected-document Direct Answer without Ollama
- Hybrid vector/BM25 retrieval with RRF and local Ollama generation
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
- `CORPORATE_BRAIN_OLLAMA_MODEL`

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
- `DELETE /api/documents/{file_hash}`
- `POST /api/search`
- `GET /api/sources/{file_hash}/{block_id}`
- `POST /api/chat` with `mode: direct | ai`

## Tests

```powershell
.\venv\Scripts\Activate.ps1
python -m pytest -q
```

Document and Chroma data remain local and must not be committed.

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

Open the frontend, create an account or sign in, open **Profile** from the
bottom-left card, start a document chat, verify it appears under **Recent** and
**History**, reopen it, then sign out with Clerk's user button. A different
Clerk account must not see the first account's conversations.

### 8. Common errors

- **Missing publishable key:** set `VITE_CLERK_PUBLISHABLE_KEY` and restart Vite.
- **Wrong origin:** add the exact localhost or production origin in Clerk.
- **401 from FastAPI:** confirm the browser sends `Authorization: Bearer ...`,
  the JWKS URL belongs to the same Clerk instance, and `azp` is allowed.
- **Invalid/expired session:** sign out/in and verify the system clock.
- **`.env.local` not loaded:** keep it inside `frontend/` and restart Vite.
- **Backend environment ignored:** set variables in the terminal that launches
  FastAPI (or load them through your deployment platform).

### 9. Security

Never commit `.env.local`, never expose `CLERK_SECRET_KEY` to Vite, and never
place server secrets in variables prefixed with `VITE_`. Publishable keys are
safe for frontend use. Revoke and rotate any credential that is accidentally
leaked. Conversation records are scoped by the verified Clerk `sub` user ID.
