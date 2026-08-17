from __future__ import annotations

import pytest

from backend.llm import GeminiProvider, GenerationProvider, ProviderError
from backend.services.runtime import CorporateBrainRuntime
from document_normalizer import CanonicalBlock, CanonicalDocument


class RecordingProvider(GenerationProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The BI host is `bi.internal`."


def runtime_with_document(provider: GenerationProvider, monkeypatch, tmp_path):
    import backend.services.runtime as runtime_module
    monkeypatch.setattr(runtime_module, "STORAGE_DIR", tmp_path / "documents")
    block = CanonicalBlock(
        block_id="ev_bi", text="System = BI | Host = bi.internal", block_type="table_row",
        source_file="systems.csv", file_hash="doc-1", row_index=2,
    )
    document = CanonicalDocument(
        document_id="doc-1", file_hash="doc-1", source_file="systems.csv",
        file_type="csv", blocks=(block,),
    )
    runtime = CorporateBrainRuntime(provider)
    runtime.registry.add(document)
    return runtime, block


def test_direct_answer_and_catalog_never_invoke_generation_provider(monkeypatch, tmp_path):
    provider = RecordingProvider()
    runtime, _ = runtime_with_document(provider, monkeypatch, tmp_path)
    runtime.chat_direct("What is the BI host?", "doc-1", None)
    runtime.documents()
    runtime.search("host")
    assert provider.prompts == []


def test_ai_answer_invokes_provider_with_evidence_and_preserves_source_text(monkeypatch, tmp_path):
    provider = RecordingProvider()
    runtime, block = runtime_with_document(provider, monkeypatch, tmp_path)
    result = runtime.chat_ai("What is the BI host?", None, None, [])
    assert len(provider.prompts) == 1
    assert block.text in provider.prompts[0]
    assert result["sources"][0]["text"] == block.text


def test_missing_gemini_key_fails_with_actionable_safe_message():
    provider = GeminiProvider(api_key="")
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        provider.generate("prompt")


def test_gemini_settings_are_forwarded_to_official_sdk_config():
    class Response:
        text = "A natural Markdown answer."

    class Models:
        def __init__(self):
            self.call = None

        def generate_content(self, **kwargs):
            self.call = kwargs
            return Response()

    class Client:
        def __init__(self):
            self.models = Models()

    client = Client()
    provider = GeminiProvider(
        api_key="test-only", model="gemini-3.6-flash", timeout=60,
        temperature=0.2, top_p=0.95, max_output_tokens=2048,
        enable_streaming=False, client=client,
    )
    assert provider.generate("grounded prompt") == "A natural Markdown answer."
    call = client.models.call
    assert call["model"] == "gemini-3.6-flash"
    assert call["contents"] == "grounded prompt"
    assert call["config"].temperature == 0.2
    assert call["config"].top_p == 0.95
    assert call["config"].max_output_tokens == 2048
    assert str(call["config"].thinking_config.thinking_level).casefold().endswith("low")
    assert call["config"].response_mime_type is None
    assert call["config"].response_schema is None
    assert list(provider.stream("grounded prompt")) == ["A natural Markdown answer."]


def test_transient_gemini_server_error_is_retried(monkeypatch):
    class Response:
        text = "ok"

    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("503 UNAVAILABLE: temporarily overloaded")
            return Response()

    class Client:
        def __init__(self):
            self.models = Models()

    client = Client()
    monkeypatch.setattr("backend.llm.gemini_provider.time.sleep", lambda _: None)
    provider = GeminiProvider(api_key="test-only", client=client)
    assert provider.generate("prompt") == "ok"
    assert client.models.calls == 3


def test_ai_answer_does_not_require_or_use_selected_document_scope(monkeypatch, tmp_path):
    provider = RecordingProvider()
    runtime, _ = runtime_with_document(provider, monkeypatch, tmp_path)
    result = runtime.chat_ai("What is the BI host?", None, None, [])
    assert result["answer"] == "The BI host is `bi.internal`."
    assert result["sources"][0]["file_hash"] == "doc-1"
