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

FastAPI is an adapter, not a second RAG implementation. It calls the existing
canonical Direct Answer engine and the existing hybrid search, prompt,
generation and citation functions.

## Features

- PDF, DOCX, DOC, XLSX, CSV and ZIP ingestion in the existing Streamlit flow
- Deterministic selected-document Direct Answer without Ollama
- Hybrid vector/BM25 retrieval with RRF and local Ollama generation
- Global prepared-document search and exact canonical source navigation
- OCM/OEG/OJO/OCI and MZ/KPSA metadata values preserved
- React dark/light/system themes and collapsible responsive sidebar
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
