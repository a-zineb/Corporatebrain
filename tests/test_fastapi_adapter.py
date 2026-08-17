from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.runtime import get_runtime


client = TestClient(app)


def test_health_filters_and_documents_use_real_runtime():
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert client.get("/api/filters").json() == {
        "zones": ["OCM", "OEG", "OJO", "OCI"],
        "applications": ["MZ", "KPSA"],
    }
    assert isinstance(client.get("/api/documents").json(), list)


def test_search_and_source_round_trip_without_chat_routing():
    response = client.post("/api/search", json={"query": "protocol", "limit": 3})
    assert response.status_code == 200
    results = response.json()["results"]
    if results:
        source = results[0]["source"]
        resolved = client.get(f"/api/sources/{source['file_hash']}/{source['block_id']}")
        assert resolved.status_code == 200
        assert resolved.json()["block_id"] == source["block_id"]


def test_direct_chat_requires_and_respects_selected_document():
    documents = get_runtime().documents()
    if not documents:
        return
    selected = documents[0]
    response = client.post("/api/chat", json={
        "message": "host?", "mode": "direct", "document_hash": selected["id"],
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"]
    assert all(source["file_hash"] == selected["id"] for source in payload["sources"])


def test_direct_chat_without_document_is_actionable():
    response = client.post("/api/chat", json={"message": "host?", "mode": "direct"})
    assert response.status_code == 422
    assert "selected document" in response.json()["detail"].casefold()


def test_invalid_clerk_token_never_blocks_direct_answer(monkeypatch):
    import backend.auth
    class BrokenJwks:
        def get_signing_key_from_jwt(self, token):
            raise __import__("jwt").InvalidTokenError("synthetic invalid token")
    monkeypatch.setattr(backend.auth, "_jwks_client", lambda: BrokenJwks())
    documents = get_runtime().documents()
    if not documents:
        return
    response = client.post("/api/chat", headers={"Authorization": "Bearer invalid"}, json={
        "message": "host?", "mode": "direct", "document_hash": documents[0]["id"],
    })
    assert response.status_code == 200
    assert response.json()["status"] != "TOKEN_INVALID"


def test_missing_token_can_use_history_in_anonymous_scope():
    conversation_id = "anonymous-auth-regression-test"
    saved = client.put(f"/api/conversations/{conversation_id}", json={
        "title": "Anonymous chat", "messages": [{"role": "user", "text": "hello"}],
    })
    assert saved.status_code == 204
    loaded = client.get(f"/api/conversations/{conversation_id}")
    assert loaded.status_code == 200 and loaded.json()["title"] == "Anonymous chat"


def test_conversation_can_be_deleted_in_its_user_scope():
    conversation_id = "anonymous-delete-regression-test"
    assert client.put(f"/api/conversations/{conversation_id}", json={
        "title": "Delete me", "messages": [],
    }).status_code == 204
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_conversation_can_be_renamed_without_replacing_messages():
    conversation_id = "anonymous-rename-regression-test"
    payload = {"title": "Old title", "messages": [{"role": "user", "text": "hello"}]}
    assert client.put(f"/api/conversations/{conversation_id}", json=payload).status_code == 204
    assert client.patch(f"/api/conversations/{conversation_id}", json={"title": "New title"}).status_code == 204
    loaded = client.get(f"/api/conversations/{conversation_id}")
    assert loaded.status_code == 200
    assert loaded.json()["title"] == "New title"
    assert loaded.json()["messages"] == payload["messages"]


def test_invalid_token_still_reaches_grounded_ai_generation(monkeypatch):
    import backend.auth

    class BrokenJwks:
        def get_signing_key_from_jwt(self, token):
            raise __import__("jwt").InvalidTokenError("synthetic invalid token")

    def generate(prompt):
        return "Grounded AI response."

    monkeypatch.setattr(backend.auth, "_jwks_client", lambda: BrokenJwks())
    monkeypatch.setattr(get_runtime().generation_provider, "generate", generate)
    documents = get_runtime().documents()
    if not documents:
        return
    response = client.post("/api/chat", headers={"Authorization": "Bearer invalid"}, json={
        "message": "Explain the main section", "mode": "ai", "document_hash": documents[0]["id"],
    })
    assert response.status_code == 200
    assert response.json()["answer"] == "Grounded AI response."
    assert response.json()["sources"]


def test_original_document_can_be_opened_or_downloaded():
    documents = get_runtime().documents()
    if not documents:
        return
    file_hash = documents[0]["id"]
    inline = client.get(f"/api/documents/{file_hash}/content")
    download = client.get(f"/api/documents/{file_hash}/content?download=true")
    assert inline.status_code == 200 and inline.content
    assert download.status_code == 200 and download.content == inline.content
    assert "inline" in inline.headers["content-disposition"]
    assert "attachment" in download.headers["content-disposition"]
