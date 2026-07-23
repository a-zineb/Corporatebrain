# Corporate Brain — Enterprise RAG Assistant

> A Retrieval-Augmented Generation (RAG) application.
> Allows employees to ask natural language questions about internal company documents and receive AI-generated answers with cited sources.

---

## Features

- Document Upload: PDF, DOCX, DOC (Word 97), XLSX, CSV, ZIP
- Hybrid Search: Vectorial (Sentence Transformers) + BM25 (keyword) with RRF fusion
- Answer Generation: via local Ollama LLM (Qwen 2.5, Llama 3, Mistral...)
- Filters: by Geographical Zone (OCM, OEG, OJO, OCI) and by Application (MZ, KPSA)
- Cited Sources: each answer displays the source files with a direct open button
- Anti-Hallucination: the AI refuses to answer if the information is not in the documents
- Conversational Memory: reformulates questions based on chat history

---

## Tech Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| PDF Extraction | PyMuPDF (fitz) |
| DOCX Extraction | python-docx |
| DOC (Word 97) Extraction | olefile |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector Database | ChromaDB |
| Keyword Search | BM25 (rank_bm25) |
| LLM | Ollama (local) |

---

## Project Structure

```
Corporatebrain/
├── app_V2.py           # Main application (current version)
├── app.py              # Initial version (reference)
├── requirements.txt    # Python dependencies
├── .streamlit/
│   └── config.toml     # Streamlit configuration (theme)
├── doc_storage_v2/     # Document folder (ignored by git)
└── chroma_db_final_v3/ # Vector database (auto-generated, ignored by git)
```

> Note: Enterprise documents and the ChromaDB database are excluded from the repository for confidentiality reasons.

---

## Versions

| Version | File | Description |
|---|---|---|
| v1 | app.py | Initial version: manual upload, basic ChromaDB |
| v2 | app_V2.py | Hybrid RRF search, metadata filters, DOC/XLSX support, cited sources |

---

## Author

Developed by Zineb — 2026
