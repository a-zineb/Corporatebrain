from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.auth import current_user_id
from backend.schemas import ChatRequest, ChatResponse, ConversationSave, SearchRequest, SourceResponse
from backend.services.runtime import get_runtime
from backend.services.conversations import ConversationStore
from backend.services.runtime import ROOT


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


@app.get("/api/documents/{file_hash}/content")
def document_content(file_hash: str, download: bool = False):
    try:
        path = get_runtime().document_path(file_hash)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    disposition = "attachment" if download else "inline"
    return FileResponse(path, filename=path.name, content_disposition_type=disposition)


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
