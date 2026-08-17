from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.auth import current_user_id
from backend.schemas import (ChatRequest, ChatResponse, ConversationRename, ConversationSave,
                             SearchRequest, SourceResponse)
from backend.services.runtime import get_runtime
from backend.services.conversations import ConversationStore
from backend.services.runtime import ROOT
from backend.llm import ProviderError, get_generation_provider


conversation_store = ConversationStore(ROOT / ".run" / "conversations.sqlite3")


app = FastAPI(title="Corporate Brain API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    # A liveness probe must stay instant; document preparation remains lazy and
    # happens on the first endpoint that actually needs the corpus.
    # Provider configuration is cheap to inspect and must not initialize the
    # document registry, Chroma, embeddings, or any model connection.
    provider = get_generation_provider()
    return {"status": "ok", "service": "corporate-brain",
            "ai_provider_configured": provider.configured}


@app.get("/api/filters")
def filters():
    return get_runtime().filters()


@app.get("/api/documents")
def documents():
    return get_runtime().documents()


@app.post("/api/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    try:
        return get_runtime().upload(file.filename or "upload", await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/documents/upload-async", status_code=202)
async def upload_document_async(file: UploadFile = File(...)):
    return get_runtime().start_upload(file.filename or "upload", await file.read())


@app.get("/api/ingestion/jobs")
def ingestion_jobs():
    return get_runtime().jobs()


@app.post("/api/ingestion/jobs/{job_id}/retry", status_code=202)
def retry_ingestion(job_id: str):
    try:
        return get_runtime().retry(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Ingestion job not found") from exc


@app.post("/api/documents/{file_hash}/reindex")
def reindex_document(file_hash: str):
    try:
        return get_runtime().reindex(file_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@app.delete("/api/documents/{file_hash}", status_code=204)
def delete_document(file_hash: str):
    try:
        get_runtime().delete(file_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@app.get("/api/documents/{file_hash}/content")
def document_content(file_hash: str, download: bool = False):
    try:
        path = get_runtime().document_path(file_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    disposition = "attachment" if download else "inline"
    return FileResponse(path, filename=path.name, content_disposition_type=disposition)


@app.get("/api/documents/{file_hash}/preview")
def document_preview(file_hash: str):
    try:
        path = get_runtime().preview_path(file_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document preview not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")


@app.get("/api/documents/{file_hash}/preview-info")
def document_preview_info(file_hash: str, block_id: str):
    try:
        return get_runtime().preview_info(file_hash, block_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Preview evidence mapping not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/documents/{file_hash}/table")
def document_table(file_hash: str, sheet: str | None = None):
    try:
        return get_runtime().tabular_evidence(file_hash, sheet)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/documents/{file_hash}/source", response_model=SourceResponse)
def document_source(file_hash: str):
    try:
        return get_runtime().first_source(file_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document source not found") from exc


@app.post("/api/search")
def search(request: SearchRequest):
    return {"query": request.query, "results": get_runtime().search(request.query, request.limit)}


@app.get("/api/sources/{file_hash}/{block_id}", response_model=SourceResponse)
def source(file_hash: str, block_id: str):
    try:
        return get_runtime().source(file_hash, block_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Source target not found") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, user_id: str = Depends(current_user_id)):
    try:
        if request.mode == "ai":
            return get_runtime().chat_ai(request.message, request.document_hash,
                                         request.conversation_id, request.history)
        return get_runtime().chat_direct(request.message, request.document_hash, request.conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Selected document not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Corporate Brain could not process this request.") from exc


@app.get("/api/conversations")
def conversations(user_id: str = Depends(current_user_id)):
    return conversation_store.list(user_id)


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: str, user_id: str = Depends(current_user_id)):
    item = conversation_store.get(conversation_id, user_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return item


@app.put("/api/conversations/{conversation_id}", status_code=204)
def save_conversation(conversation_id: str, payload: ConversationSave,
                      user_id: str = Depends(current_user_id)):
    conversation_store.upsert(conversation_id, user_id, payload.title, payload.document_hash,
                              payload.document_name, payload.messages)


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user_id: str = Depends(current_user_id)):
    if not conversation_store.delete(conversation_id, user_id):
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.patch("/api/conversations/{conversation_id}", status_code=204)
def rename_conversation(conversation_id: str, payload: ConversationRename,
                        user_id: str = Depends(current_user_id)):
    if not conversation_store.rename(conversation_id, user_id, payload.title.strip()):
        raise HTTPException(status_code=404, detail="Conversation not found")
