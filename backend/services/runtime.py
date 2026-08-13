from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import RLock
import os
import time
import uuid

import canonical_rag
import rag_pipeline
from backend.services.ai_quality import generate_grounded
from mvp_services import PreparedDocumentRegistry, detect_query_language


ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR = Path(os.getenv("CORPORATE_BRAIN_STORAGE_DIR", ROOT / "doc_storage_v2"))
CHROMA_PATH = str(Path(os.getenv("CORPORATE_BRAIN_CHROMA_PATH", ROOT / "chroma_db_local_v2")))
COLLECTION_NAME = os.getenv("CORPORATE_BRAIN_COLLECTION", "documents")
OLLAMA_MODEL = os.getenv("CORPORATE_BRAIN_OLLAMA_MODEL", "qwen3:8b")


class CorporateBrainRuntime:
    """Thin reusable facade over the existing canonical and RAG functions."""

    def __init__(self) -> None:
        self.registry = PreparedDocumentRegistry()
        self._paths: dict[str, Path] = {}
        self._lock = RLock()
        self._chroma = None
        self._embedding_model = None
        self._bm25_payload = None
        self.refresh_documents()

    def refresh_documents(self) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        for path in sorted(STORAGE_DIR.iterdir()):
            if not path.is_file() or path.suffix.casefold() not in {".pdf", ".docx", ".doc", ".xlsx", ".csv", ".zip"}:
                continue
            try:
                result = self.registry.prepare(path.read_bytes(), path.name)
            except OSError:
                continue
            if result.document is not None:
                self._paths[result.document.file_hash] = path

    def documents(self) -> list[dict[str, object]]:
        return [{
            "id": document.file_hash,
            "name": document.source_file,
            "type": document.file_type,
            "status": canonical_rag.ingestion_diagnostics(document).status,
            "blocks": len(document.blocks),
            "warnings": list(document.warnings),
            "filiale": next((zone for zone in ("OCM", "OEG", "OJO", "OCI")
                              if zone in document.source_file.upper()), None),
            "application": next((app for app in ("MZ", "KPSA")
                                  if app in document.source_file.upper()), None),
        } for document in self.registry.documents]

    def document_path(self, file_hash: str) -> Path:
        path = self._paths.get(file_hash)
        if path is None or not path.is_file() or path.parent.resolve() != STORAGE_DIR.resolve():
            raise KeyError(file_hash)
        return path

    def filters(self) -> dict[str, list[str]]:
        return {"zones": ["OCM", "OEG", "OJO", "OCI"], "applications": ["MZ", "KPSA"]}

    def upload(self, name: str, data: bytes) -> dict[str, object]:
        with self._lock:
            result = self.registry.prepare(data, name)
            if result.document is None or result.state == "FAILED":
                raise ValueError("The document could not be prepared reliably.")
            safe_name = Path(name).name
            destination = STORAGE_DIR / safe_name
            if destination.exists() and destination.read_bytes() != data:
                destination = STORAGE_DIR / f"{destination.stem}_{result.document.file_hash[:8]}{destination.suffix}"
            destination.write_bytes(data)
            self._paths[result.document.file_hash] = destination
            return next(item for item in self.documents() if item["id"] == result.document.file_hash)

    def delete(self, file_hash: str) -> None:
        path = self._paths.get(file_hash)
        if path is None or not path.is_file() or path.parent.resolve() != STORAGE_DIR.resolve():
            raise KeyError(file_hash)
        path.unlink()
        # Rebuild the small registry so deleted content cannot remain searchable.
        self.registry = PreparedDocumentRegistry()
        self._paths.clear()
        self.refresh_documents()

    def search(self, query: str, limit: int = 50) -> list[dict[str, object]]:
        return [{
            "document_hash": hit.document.file_hash,
            "document_name": hit.document.source_file,
            "file_type": hit.document.file_type,
            "title": hit.display_title,
            "relation": hit.relation,
            "entity": hit.entity,
            "value": hit.display_value,
            "preview": hit.preview,
            "score": hit.score,
            "source": self._source_dict(hit.target),
        } for hit in self.registry.global_search(query, limit)]

    def source(self, file_hash: str, block_id: str) -> dict[str, object]:
        target = self.registry.source_target(file_hash, block_id)
        if target is None:
            raise KeyError((file_hash, block_id))
        return self._source_dict(target)

    def first_source(self, file_hash: str) -> dict[str, object]:
        document = next((item for item in self.registry.documents if item.file_hash == file_hash), None)
        if document is None or not document.blocks:
            raise KeyError(file_hash)
        target = self.registry.source_target(file_hash, document.blocks[0].block_id)
        if target is None:
            raise KeyError(file_hash)
        return self._source_dict(target)

    @staticmethod
    def _source_dict(target) -> dict[str, object]:
        return {
            "document": target.source_file, "file_hash": target.file_hash,
            "file_type": target.file_type, "block_id": target.block_id,
            "location": target.location_label, "page": target.page, "sheet": target.sheet,
            "row": target.row_index, "section": target.section, "text": target.evidence_text,
            "metadata": {"bbox": target.bbox, "table_index": target.table_index,
                         "paragraph_index": target.paragraph_index},
        }

    def chat_direct(self, message: str, document_hash: str | None, conversation_id: str | None) -> dict[str, object]:
        if not document_hash:
            raise ValueError("Direct Answer requires a selected document.")
        document = next((item for item in self.registry.documents if item.file_hash == document_hash), None)
        if document is None:
            raise KeyError(document_hash)
        started = time.perf_counter()
        result = canonical_rag.answer_direct(message, document)
        return {
            "answer": result.answer, "status": result.status, "result_type": result.result_type,
            "language": result.query_language, "method": result.method,
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "sources": [self._source_dict(self.registry.source_target(document.file_hash, block.block_id))
                        for block in result.evidence_blocks],
            "suggestions": list(result.suggestions),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    def _load_rag(self):
        if self._chroma is None:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            self._chroma = client.get_collection(COLLECTION_NAME)
            config = rag_pipeline.RAGConfig(chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME)
            self._embedding_model = rag_pipeline.load_embedding_model_offline(config)
            self._bm25_payload = rag_pipeline.build_bm25_index(self._chroma, self._chroma.count())
        return self._chroma, self._embedding_model, self._bm25_payload

    def chat_ai(self, message: str, document_hash: str | None, conversation_id: str | None,
                history: list[dict[str, str]]) -> dict[str, object]:
        import ollama
        selected = next((item for item in self.registry.documents if item.file_hash == document_hash), None)
        if selected is None:
            raise ValueError("AI Answer requires a selected document.")

        def generate(prompt: str) -> str:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.2},
            )
            if isinstance(response, dict):
                return str(response.get("message", {}).get("content", ""))
            return str(response.message.content)

        started = time.perf_counter()
        generation, evidence, language, intent = generate_grounded(
            message, selected, history, generate,
        )
        source_payload = [self._source_dict(
            self.registry.source_target(selected.file_hash, block.block_id)
        ) for block in evidence]
        return {
            "answer": generation.answer, "status": "ANSWER" if evidence else "NO_EVIDENCE",
            "result_type": intent, "language": language,
            "method": "canonical_grounded_generation_repair" if generation.repaired else "canonical_grounded_generation",
            "conversation_id": conversation_id or str(uuid.uuid4()), "sources": source_payload,
            "suggestions": [], "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }


@lru_cache(maxsize=1)
def get_runtime() -> CorporateBrainRuntime:
    return CorporateBrainRuntime()
