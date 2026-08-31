# 🧠 Corporate Brain

> **Enterprise RAG · Intelligent Document Search · Grounded AI Answers**

Corporate Brain is an enterprise-grade **Retrieval-Augmented Generation (RAG)** platform designed to transform corporate documents into a searchable, explainable knowledge base.

It combines **deterministic retrieval**, **hybrid semantic + lexical search**, **document-aware navigation**, and **Gemini-powered answer synthesis** while keeping source references under application control.

---

## ✨ Highlights

| Capability                    | Description                                                          |
| ----------------------------- | -------------------------------------------------------------------- |
| 🧠 **Hybrid RAG**             | Vector + BM25 retrieval with Reciprocal Rank Fusion                  |
| 🎯 **Deterministic Answers**  | Selected-document answers without an external AI call                |
| ✨ **AI Answers**              | Gemini-powered synthesis grounded only in retrieved evidence         |
| 📚 **Multi-format ingestion** | PDF, DOCX, DOC, XLSX, CSV and ZIP                                    |
| 🔎 **Global Search**          | Search across the complete prepared document catalog                 |
| 📌 **Source Navigation**      | Jump directly to the originating document, page, sheet or cell range |
| 📊 **Spreadsheet Citations**  | Exact sheet, row and cell-range references                           |
| 🕸️ **Knowledge Graph**       | Explore document relationships visually                              |
| ⚡ **Lazy Loading**            | Parse and hydrate documents only when required                       |
| 🔐 **Authentication**         | Clerk-based authentication with user-scoped conversations            |
| 🌗 **Modern UI**              | React interface with dark, light and system themes                   |
| 🧪 **Tested**                 | Python + TypeScript test suites                                      |
| 🖥️ **Legacy Fallback**       | Streamlit application retained during migration                      |

---

# 🏗️ Architecture

```text
                         ┌─────────────────────────┐
                         │       React + TS        │
                         │       Frontend          │
                         └────────────┬────────────┘
                                      │
                                  HTTP / JSON
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        FastAPI          │
                         │        API Layer        │
                         └────────────┬────────────┘
                                      │
                             Direct function calls
                                      │
                                      ▼
              ┌───────────────────────────────────────────┐
              │          Canonical Document Engine         │
              │                                           │
              │  canonical_rag.py                         │
              │  rag_pipeline.py                          │
              │  document_normalizer.py                   │
              └─────────────────────┬─────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
              ┌──────────────┐             ┌──────────────┐
              │   ChromaDB   │             │     BM25     │
              │ Vector Search│             │ Lexical Search│
              └──────┬───────┘             └──────┬───────┘
                     │                             │
                     └─────────────┬───────────────┘
                                   │
                                   ▼
                           ┌───────────────┐
                           │   RRF Fusion  │
                           └───────┬───────┘
                                   │
                            Selected Evidence
                                   │
                                   ▼
                         ┌────────────────────┐
                         │    Gemini API      │
                         │  Answer Synthesis  │
                         └────────────────────┘
```

### Request modes

**Direct Answer**

```text
Selected Document
       ↓
Deterministic Retrieval
       ↓
Evidence
       ↓
Answer
```

No external AI provider is required.

**AI Answer**

```text
User Query
    ↓
Intent Classification
    ↓
Global Document Discovery
    ↓
Relevant Document Hydration
    ↓
Hybrid Retrieval
    ↓
Evidence Selection
    ↓
Gemini Synthesis
    ↓
Grounded Markdown Answer
```

Gemini receives only the application-selected evidence. Source cards remain controlled by Corporate Brain.

---

# 🚀 Quick Start

## One-command startup

From **PowerShell at the repository root**:

```powershell
./run
```

The launcher automatically:

1. Creates the Python virtual environment if necessary
2. Synchronizes `requirements.txt`
3. Installs Node.js LTS through `winget` when required
4. Creates `frontend/.env.local`
5. Runs `npm install`
6. Executes the strict TypeScript/Vite production build
7. Starts FastAPI on `:8000`
8. Starts React on `:5173`
9. Checks both services
10. Opens the application in your browser
11. Shuts both servers down with `Ctrl+C`

### ⚡ Faster development restart

After a successful build:

```powershell
./run -SkipBuild
```

Prevent automatic browser launch:

```powershell
./run -NoBrowser
```

---

# 📦 Project Structure

```text
Corporate-Brain/
│
├── backend/
│   └── main.py
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── .env.example
│   └── package.json
│
├── canonical_rag.py
├── rag_pipeline.py
├── document_normalizer.py
├── app.py
│
├── requirements.txt
├── .env.example
├── run
├── run.ps1
│
├── chroma_db_local_v2/
├── doc_storage_v2/
└── .run/
    ├── prepared/
    └── previews/
```

---

# 📚 Supported Documents

Corporate Brain currently supports:

* 📄 PDF
* 📝 DOCX
* 📝 DOC
* 📊 XLSX
* 📋 CSV
* 📦 ZIP

The ingestion pipeline normalizes documents into a canonical representation before chunking, embedding and indexing.

---

# 🔍 Retrieval Engine

Corporate Brain uses a **hybrid retrieval architecture** combining semantic and lexical search.

```text
                    User Query
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
        Vector Search           BM25 Search
          ChromaDB               Keywords
             │                     │
             └──────────┬──────────┘
                        ▼
                  RRF Fusion
                        │
                        ▼
                Ranked Evidence
```

This approach combines semantic similarity with exact keyword matching.

The system preserves canonical metadata including:

```text
OCM
OEG
OJO
OCI
MZ
KPSA
```

---

# ⚡ Lazy Document Loading

Corporate Brain does **not** eagerly parse every document at startup.

Instead, it maintains a lightweight catalog containing:

* Checksums
* Filenames
* File types
* File sizes
* Business metadata
* Discovery terms

Full parsing happens through:

```python
hydrate_document(file_hash)
```

### Loading strategy

| Component           | Behavior                                                       |
| ------------------- | -------------------------------------------------------------- |
| Direct Answer       | Hydrates only the selected document                            |
| AI Answer           | Searches document profiles first, then hydrates relevant files |
| Source Navigation   | Reuses hydrated documents                                      |
| Upload              | Creates catalog entry before full parsing                      |
| Cache               | Bounded in-memory LRU + persistent cache                       |
| Concurrent requests | Share the same in-flight hydration                             |
| File changes        | New checksum invalidates the relevant cache                    |

### Configuration

```env
LAZY_HYDRATION_ENABLED=true
HYDRATED_DOCUMENT_CACHE_SIZE=8
MAX_CONCURRENT_HYDRATIONS=3
AI_DISCOVERY_MAX_DOCUMENTS=6
```

### Performance

On the current **22-document corpus**, the measured startup time decreased from approximately:

```text
59,686 ms  →  486 ms
```

with zero documents hydrated during startup.

Additional measured values:

```text
Document-card serialization     ≈ 0.05 ms
Persistent PDF/DOCX hydration   ≈ 8 ms
First uncached XLSX hydration   ≈ 431 ms
In-memory cache hit             ≈ 0.02 ms
```

---

# 🕸️ Knowledge Graph

The Graph View provides a visual representation of relationships between documents.

### Capabilities

* Approximate search
* File-type filtering
* Global / Focused / Community modes
* Zoom and reset controls
* Fullscreen mode
* Tooltips
* Relationship explanations
* Document detail panel
* Collision-aware force layout
* Significance filtering
* Maximum three strongest edges per document

Run frontend tests with:

```powershell
cd frontend
npm test
```

---

# 🔐 Authentication

Corporate Brain uses **Clerk** for authentication.

```text
React / Clerk
      │
      │ JWT
      ▼
   FastAPI
      │
      ▼
JWT verification
      │
      ▼
User-scoped conversations
```

The frontend receives only the **publishable key**.

The backend verifies Clerk JWTs using the instance's public JWKS endpoint.

> ⚠️ `CLERK_SECRET_KEY` is never required by the frontend and must never be exposed through Vite.

---

# 🤖 Gemini Configuration

## Setup

1. Create a Gemini API key in Google AI Studio.
2. Copy `.env.example` to `.env`.
3. Configure:

```env
GEMINI_API_KEY=your_key_here
AI_PROVIDER=gemini
```

The default model is:

```text
gemini-3.6-flash
```

### Available configuration

```env
AI_PROVIDER
GEMINI_MODEL
GEMINI_TIMEOUT_SECONDS
GEMINI_TEMPERATURE
GEMINI_TOP_P
GEMINI_MAX_OUTPUT_TOKENS
GEMINI_ENABLE_STREAMING
```

### 🔒 Security

Never place the Gemini API key in:

```text
frontend/.env.local
VITE_* variables
source code
browser storage
```

If the provider is unavailable, Direct Answer and document search remain functional.

---

# 🖥️ Frontend Development

Install a current Node.js LTS release.

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Configure:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Backend secrets are never sent to React.

### Production build

```powershell
cd frontend
npm run build
```

---

# ⚙️ Backend Development

Activate the Python environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start FastAPI:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Health endpoint:

```text
http://localhost:8000/api/health
```

---

# 🧩 Legacy Streamlit Application

The original Streamlit application remains available during the React migration.

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Do not remove Streamlit until the following have passed browser acceptance testing:

* Chat
* AI generation
* Citations
* Filters
* Uploads
* Deletion
* Source navigation

---

# 🔌 API Reference

## Health & Discovery

```http
GET /api/health
GET /api/filters
GET /api/documents
```

## Documents

```http
POST   /api/documents/upload
POST   /api/documents/upload-async
DELETE /api/documents/{file_hash}
POST   /api/documents/{file_hash}/reindex
GET    /api/documents/{file_hash}/content
GET    /api/documents/{file_hash}/preview
GET    /api/documents/{file_hash}/preview-info
GET    /api/documents/{file_hash}/table
```

## Ingestion

```http
GET  /api/ingestion/jobs
POST /api/ingestion/jobs/{job_id}/retry
```

## Search & Sources

```http
POST /api/search
GET  /api/sources/{file_hash}/{block_id}
```

## Chat

```http
POST /api/chat
```

Supported modes:

```text
direct
ai
```

## Conversations

```http
GET    /api/conversations/{conversation_id}
PUT    /api/conversations/{conversation_id}
PATCH  /api/conversations/{conversation_id}
DELETE /api/conversations/{conversation_id}
```

---

# 📌 Citation System

Corporate Brain maintains precise source references across different file types.

| File type | Citation                    |
| --------- | --------------------------- |
| 📊 XLSX   | Exact sheet + cell range    |
| 📋 CSV    | Original row number         |
| 📄 PDF    | Original one-based PDF page |
| 📝 DOCX   | Cached rendered preview     |

PDF citations use:

```text
#page=N
```

Arbitrary filesystem paths are never accepted by the API.

---

# 🔄 Ingestion Pipeline

Asynchronous ingestion exposes real processing stages:

```text
uploading
    ↓
extracting
    ↓
normalizing
    ↓
chunking
    ↓
embedding
    ↓
indexing
    ↓
ready
```

Alternative terminal states:

```text
warning
failed
```

Retry operations reuse the retained upload payload.

Re-indexing removes only the selected document's previous Chroma records before rebuilding its index.

---

# 🧪 Testing

## Backend

```powershell
.\venv\Scripts\Activate.ps1
python -m pytest -q
```

## Frontend

```powershell
cd frontend
npm test
```

---

# 🗄️ ChromaDB & Re-indexing

The certified benchmark:

```text
Collection: corporatebrain.v1
Corpus: 1,072 chunks
```

The development upload directory may legitimately contain a different number of chunks.

Before rebuilding a development index:

1. Back up `doc_storage_v2`
2. Back up `chroma_db_local_v2`
3. Remove unintended duplicate uploads through the application
4. Start a fresh collection

Example:

```powershell
$env:CORPORATE_BRAIN_COLLECTION="documents-schema-v3"
./run.ps1
```

Keep the previous collection until representative searches and citations have been verified.

> ⚠️ Parser or schema changes require a new collection name or an explicit full re-index. Mixing chunks from different parser versions is not a certified state.

---

# 🔧 Environment Variables

## Storage

```env
CORPORATE_BRAIN_STORAGE_DIR
CORPORATE_BRAIN_CHROMA_PATH
CORPORATE_BRAIN_COLLECTION
```

## AI

```env
AI_PROVIDER
GEMINI_MODEL
GEMINI_TIMEOUT_SECONDS
GEMINI_TEMPERATURE
GEMINI_TOP_P
GEMINI_MAX_OUTPUT_TOKENS
GEMINI_ENABLE_STREAMING
```

## Lazy Hydration

```env
LAZY_HYDRATION_ENABLED=true
HYDRATED_DOCUMENT_CACHE_SIZE=8
MAX_CONCURRENT_HYDRATIONS=3
AI_DISCOVERY_MAX_DOCUMENTS=6
```

## Frontend

```env
VITE_API_BASE_URL
VITE_CLERK_PUBLISHABLE_KEY
```

## Authentication

```env
CLERK_JWKS_URL
CLERK_AUTHORIZED_PARTIES
```

---

# 🛡️ Security

Corporate Brain follows a strict separation between frontend configuration and server-side secrets.

### Never commit

```text
.env
.env.local
API keys
server secrets
document storage
Chroma data
```

### Never expose

```text
CLERK_SECRET_KEY
GEMINI_API_KEY
server-side credentials
```

### User isolation

Conversation records are scoped using the verified Clerk `sub` user identifier.

If a credential is accidentally exposed:

> **Revoke it and rotate it immediately.**

---

# 📈 Current Performance

### Startup

```text
Before lazy hydration    59,686 ms
After lazy hydration        486 ms

Improvement              ~99%
```

### Corpus

```text
Documents                  22
Certified benchmark chunks 1,072
```

### Hydration

```text
PDF/DOCX persistent       ~8 ms
XLSX uncached           ~431 ms
Memory cache             ~0.02 ms
```

---

# 🧭 Development Philosophy

Corporate Brain is built around several core principles:

### 1. Grounded answers

AI answers should be generated from application-selected evidence rather than unrestricted model knowledge.

### 2. Deterministic retrieval

Direct Answer remains deterministic and does not require an AI provider.

### 3. Source transparency

Users should be able to navigate from an answer back to the originating document.

### 4. Efficient document processing

Documents are hydrated only when necessary.

### 5. Separation of concerns

```text
Frontend
   ↓
API
   ↓
Canonical document services
   ↓
Retrieval
   ↓
Evidence
   ↓
AI synthesis
```

### 6. Graceful degradation

If Gemini is unavailable, core search, Direct Answer, catalog, graph and source navigation remain available.

---

# 🚦 Development Status

```text
┌─────────────────────────────────────────┐
│         CORPORATE BRAIN                 │
│                                         │
│  🟢 RAG Engine          Available       │
│  🟢 Hybrid Retrieval    Available       │
│  🟢 React Frontend      Available       │
│  🟢 FastAPI Backend     Available       │
│  🟢 Gemini Integration  Available       │
│  🟢 Clerk Auth          Available       │
│  🟢 Knowledge Graph     Available       │
│  🟢 Lazy Hydration      Available       │
│  🟡 Streamlit           Migration mode  │
└─────────────────────────────────────────┘
```

---

# 🧠 Corporate Brain

**Turn corporate documents into an intelligent, searchable and explainable knowledge base.**

```text
Documents
    ↓
Normalize
    ↓
Index
    ↓
Retrieve
    ↓
Ground
    ↓
Answer
    ↓
Cite
```

> **Search less. Understand more.**
