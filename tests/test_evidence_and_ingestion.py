from __future__ import annotations

import time
from pathlib import Path

import fitz

from backend.services.evidence import PreviewService, read_tabular_evidence
from backend.services.runtime import CorporateBrainRuntime
from document_normalizer import CanonicalBlock, CanonicalDocument


def test_docx_preview_maps_blocks_to_rendered_pdf_pages(tmp_path, monkeypatch):
    original = tmp_path / "document with spaces (v1).docx"
    original.write_bytes(b"synthetic")
    document = CanonicalDocument("doc", "hash", original.name, "docx", (
        CanonicalBlock("b1", "First rendered paragraph", "paragraph", original.name, "hash"),
        CanonicalBlock("b2", "Second rendered requirement", "requirement", original.name, "hash"),
    ))

    def render(_original: Path, destination: Path):
        pdf = fitz.open()
        pdf.new_page().insert_text((72, 72), "First rendered paragraph")
        pdf.new_page().insert_text((72, 72), "Second rendered requirement")
        pdf.save(destination)
        pdf.close()

    monkeypatch.setattr(PreviewService, "_convert_with_word", staticmethod(render))
    preview, mapping = PreviewService(tmp_path / "previews").ensure(original, document)
    assert preview.is_file()
    assert mapping == {"b1": 1, "b2": 2}


def test_csv_table_viewer_preserves_multiline_and_rows(tmp_path):
    path = tmp_path / "données (locales).csv"
    path.write_text('Name,Details\nA,"line one\nline two"\n', encoding="utf-8")
    payload = read_tabular_evidence(path)
    assert payload["kind"] == "csv"
    assert payload["rows"][1][1].splitlines() == ["line one", "line two"]
    assert payload["max_row"] == 2


def test_reindex_deletes_only_target_document_before_stable_add():
    block = CanonicalBlock("block", "Evidence", "paragraph", "target.pdf", "hash", page=2)
    document = CanonicalDocument("doc", "hash", "target.pdf", "pdf", (block,))

    class Collection:
        def __init__(self): self.deletes=[];self.added=None
        def delete(self, **kwargs): self.deletes.append(kwargs)
        def add(self, **kwargs): self.added=kwargs
        def count(self): return 1
        def get(self, **kwargs): return {"documents":["Evidence"],"metadatas":[{"file_hash":"hash"}]}
    class Model:
        def encode(self, texts): return [[0.1, 0.2] for _ in texts]

    runtime = object.__new__(CorporateBrainRuntime)
    runtime.registry = type("Registry", (), {"documents": [document], "remove": lambda *_: None})()
    runtime.hydrator = type("Hydrator", (), {"invalidate": lambda *_: None})()
    runtime.hydrate_document = lambda _hash: document
    collection = Collection()
    runtime._load_rag = lambda: (collection, Model(), None)
    runtime._bm25_payload = None
    result = runtime.reindex("hash")
    assert collection.deletes == [{"where": {"file_hash": "hash"}}, {"where": {"source_file": "target.pdf"}}]
    assert collection.added["ids"] == ["hash:block"]
    assert result["chunks"] == 1


def test_async_ingestion_reports_real_stage_sequence(monkeypatch):
    block = CanonicalBlock("b", "Evidence", "paragraph", "sample.csv", "hash")
    document = CanonicalDocument("doc", "hash", "sample.csv", "csv", (block,))
    runtime = object.__new__(CorporateBrainRuntime)
    runtime._lock = __import__("threading").RLock()
    runtime._jobs = {};runtime._job_payloads={};runtime._paths={}
    runtime._executor = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
    runtime.upload = lambda *_: {"id": "hash", "name": "sample.csv"}
    from backend.services import runtime as runtime_module
    monkeypatch.setattr(runtime_module, "STORAGE_DIR", Path(__file__).parent / ".tmp-ingestion")
    runtime_module.STORAGE_DIR.mkdir(exist_ok=True)
    job = runtime.start_upload("sample.csv", b"a,b\n1,2\n")
    for _ in range(100):
        current = runtime.jobs()[0]
        if current["status"] != "running": break
        time.sleep(.01)
    assert current["stage"] == "ready"
    assert current["completed_stages"] == current["total_stages"] == 2
    (runtime_module.STORAGE_DIR / "sample.csv").unlink(missing_ok=True)
    runtime_module.STORAGE_DIR.rmdir()
