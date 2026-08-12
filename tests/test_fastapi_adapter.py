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
