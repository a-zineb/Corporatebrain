from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import ChatRequest, ChatResponse, SearchRequest, SourceResponse
from backend.services.runtime import get_runtime


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
    return {"status": "ok", "service": "corporate-brain"}


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


@app.delete("/api/documents/{file_hash}", status_code=204)
def delete_document(file_hash: str):
    try:
        get_runtime().delete(file_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


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
def chat(request: ChatRequest):
    try:
        if request.mode == "ai":
            return get_runtime().chat_ai(request.message, request.document_hash,
                                         request.conversation_id, request.history)
        return get_runtime().chat_direct(request.message, request.document_hash, request.conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Selected document not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Corporate Brain could not process this request.") from exc
