from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import time

import canonical_rag
from backend.services.lazy_documents import DocumentHydrator, LightweightCatalog
from backend.services.runtime import CorporateBrainRuntime
from backend.llm import GenerationProvider
from document_normalizer import CanonicalBlock, CanonicalDocument


def make_document(data: bytes, name: str) -> CanonicalDocument:
    digest = sha256(data).hexdigest()
    block = CanonicalBlock("block", "Evidence", "paragraph", name, digest)
    return CanonicalDocument(digest, digest, name, Path(name).suffix.lstrip("."), (block,))


def catalog_with_files(tmp_path: Path, count: int = 1) -> LightweightCatalog:
    storage = tmp_path / "documents"
    storage.mkdir()
    for index in range(count):
        (storage / f"document-{index}.csv").write_text(f"Name,Value\nA,{index}\n", encoding="utf-8")
    return LightweightCatalog(storage, tmp_path / "catalog.json")


def test_catalog_scans_files_without_canonical_parsing(tmp_path, monkeypatch):
    storage = tmp_path / "documents"
    storage.mkdir()
    for index in range(20):
        (storage / f"file-{index}.csv").write_text(f"A,B\n1,{index}\n", encoding="utf-8")
    monkeypatch.setattr(canonical_rag, "normalize_with_gate",
                        lambda *_: (_ for _ in ()).throw(AssertionError("eager parse")))
    catalog = LightweightCatalog(storage, tmp_path / "catalog.json")
    assert len(catalog.entries) == 20
    assert all(entry.status == "AVAILABLE" for entry in catalog.entries.values())


def test_hydration_is_deduplicated_for_concurrent_requests(tmp_path, monkeypatch):
    catalog = catalog_with_files(tmp_path)
    entry = next(iter(catalog.entries.values()))
    calls = 0

    def normalize(data, name):
        nonlocal calls
        calls += 1
        time.sleep(.05)
        return type("Outcome", (), {"document": make_document(data, name)})()

    monkeypatch.setattr(canonical_rag, "normalize_with_gate", normalize)
    hydrator = DocumentHydrator(catalog, tmp_path / "prepared", memory_size=2, max_workers=2)
    with ThreadPoolExecutor(max_workers=4) as executor:
        documents = list(executor.map(lambda _: hydrator.hydrate(entry.file_hash), range(4)))
    assert calls == 1
    assert len({id(document) for document in documents}) == 1


def test_persistent_cache_survives_restart_without_reparse(tmp_path, monkeypatch):
    catalog = catalog_with_files(tmp_path)
    entry = next(iter(catalog.entries.values()))
    calls = 0

    def normalize(data, name):
        nonlocal calls
        calls += 1
        return type("Outcome", (), {"document": make_document(data, name)})()

    monkeypatch.setattr(canonical_rag, "normalize_with_gate", normalize)
    first = DocumentHydrator(catalog, tmp_path / "prepared").hydrate(entry.file_hash)
    second_hydrator = DocumentHydrator(catalog, tmp_path / "prepared")
    second = second_hydrator.hydrate(entry.file_hash)
    assert first == second
    assert calls == 1
    assert second_hydrator.metrics["persistent_hits"] == 1


def test_lru_evicts_memory_only_and_keeps_persistent_cache(tmp_path, monkeypatch):
    catalog = catalog_with_files(tmp_path, 3)
    monkeypatch.setattr(canonical_rag, "normalize_with_gate",
                        lambda data, name: type("Outcome", (), {"document": make_document(data, name)})())
    hydrator = DocumentHydrator(catalog, tmp_path / "prepared", memory_size=2)
    hashes = list(catalog.entries)
    for file_hash in hashes:
        hydrator.hydrate(file_hash)
    assert hydrator.memory_hashes == tuple(hashes[-2:])
    assert all(hydrator._cache_path(file_hash).is_file() for file_hash in hashes)


def test_runtime_direct_is_strict_and_ai_hydrates_only_discovered_document(tmp_path, monkeypatch):
    import backend.services.runtime as runtime_module
    storage = tmp_path / "documents"
    storage.mkdir()
    (storage / "EKYC_test_cases.csv").write_text("Case,Expected\nTC01,Accepted\n", encoding="utf-8")
    (storage / "Tango_protocol.csv").write_text("System,Protocol\nTango,SFTP\n", encoding="utf-8")
    (storage / "CRBT_hosts.csv").write_text("System,Host\nCRBT,10.0.0.1\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "STORAGE_DIR", storage)
    monkeypatch.setattr(runtime_module, "CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(runtime_module, "PREPARED_CACHE_DIR", tmp_path / "prepared")
    monkeypatch.setattr(runtime_module, "PREVIEW_DIR", tmp_path / "previews")

    class Provider(GenerationProvider):
        def generate(self, prompt: str) -> str:
            return "TC01 is accepted."

    runtime = CorporateBrainRuntime(Provider())
    assert runtime.hydrator.memory_hashes == ()
    result = runtime.chat_ai("Give me the EKYC test cases", None, None, [])
    assert result["status"] == "ANSWER"
    assert len(runtime.hydrator.memory_hashes) == 1
    ekyc_hash = next(item["id"] for item in runtime.documents() if "EKYC" in item["name"])
    assert runtime.hydrator.memory_hashes == (ekyc_hash,)

    tango_hash = next(item["id"] for item in runtime.documents() if "Tango" in item["name"])
    direct = runtime.chat_direct("What is the protocol?", tango_hash, None)
    assert "SFTP" in direct["answer"]
    assert all(source["file_hash"] == tango_hash for source in direct["sources"])
