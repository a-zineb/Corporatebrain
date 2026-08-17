from __future__ import annotations

from functools import lru_cache
from difflib import get_close_matches
from pathlib import Path
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
import os
import time
import uuid
import re
import fitz

from dotenv import load_dotenv

import canonical_rag
import rag_pipeline
from backend.services.ai_quality import (build_conversational_prompt, classify_intent,
                                         retrieve_evidence, rewrite_follow_up)
from backend.llm import GenerationProvider, get_generation_provider
from backend.services.evidence import PreviewService, read_tabular_evidence
from mvp_services import PreparedDocumentRegistry, detect_query_language


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
STORAGE_DIR = Path(os.getenv("CORPORATE_BRAIN_STORAGE_DIR", ROOT / "doc_storage_v2"))
CHROMA_PATH = str(Path(os.getenv("CORPORATE_BRAIN_CHROMA_PATH", ROOT / "chroma_db_local_v2")))
COLLECTION_NAME = os.getenv("CORPORATE_BRAIN_COLLECTION", "documents")
PREVIEW_DIR = ROOT / ".run" / "previews"


class CorporateBrainRuntime:
    """Thin reusable facade over the existing canonical and RAG functions."""

    def __init__(self, generation_provider: GenerationProvider | None = None) -> None:
        self.registry = PreparedDocumentRegistry()
        self._paths: dict[str, Path] = {}
        self._lock = RLock()
        self._chroma = None
        self._embedding_model = None
        self._bm25_payload = None
        self.generation_provider = generation_provider or get_generation_provider()
        self.previews = PreviewService(PREVIEW_DIR)
        self._jobs: dict[str, dict[str, object]] = {}
        self._job_payloads: dict[str, tuple[str, bytes]] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="corporate-ingest")
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

    def preview_path(self, file_hash: str) -> Path:
        path = self.document_path(file_hash)
        document = next((item for item in self.registry.documents if item.file_hash == file_hash), None)
        if document is None or document.file_type not in {"docx", "doc"}:
            raise KeyError(file_hash)
        preview, _ = self.previews.ensure(path, document)
        return preview

    def preview_info(self, file_hash: str, block_id: str) -> dict[str, object]:
        path = self.document_path(file_hash)
        document = next((item for item in self.registry.documents if item.file_hash == file_hash), None)
        if document is None or document.file_type not in {"docx", "doc"}:
            raise KeyError(file_hash)
        preview, mapping = self.previews.ensure(path, document)
        page = mapping.get(block_id)
        if page is None:
            raise KeyError((file_hash, block_id))
        return {"page": page, "pages": len(fitz.open(preview))}

    def tabular_evidence(self, file_hash: str, sheet: str | None = None) -> dict[str, object]:
        return read_tabular_evidence(self.document_path(file_hash), sheet)

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

    def start_upload(self, name: str, data: bytes) -> dict[str, object]:
        job_id = str(uuid.uuid4())
        safe_name = Path(name).name
        self._job_payloads[job_id] = (safe_name, data)
        self._jobs[job_id] = {"id": job_id, "name": safe_name, "stage": "uploading",
                              "completed_stages": 1, "total_stages": 7, "status": "running",
                              "error": None, "document_id": None}
        self._executor.submit(self._process_upload, job_id)
        return dict(self._jobs[job_id])

    def _set_job(self, job_id: str, stage: str, completed: int, **values: object) -> None:
        with self._lock:
            self._jobs[job_id].update(stage=stage, completed_stages=completed, **values)

    def _process_upload(self, job_id: str) -> None:
        name, data = self._job_payloads[job_id]
        try:
            self._set_job(job_id, "extracting", 2)
            result = self.registry.prepare(data, name)
            if result.document is None or result.state == "FAILED":
                raise ValueError("The document could not be prepared reliably.")
            self._set_job(job_id, "normalizing", 3)
            self._set_job(job_id, "chunking", 4, units_total=len(result.document.blocks),
                          units_completed=len(result.document.blocks))
            with self._lock:
                safe_name = Path(name).name
                destination = STORAGE_DIR / safe_name
                if destination.exists() and destination.read_bytes() != data:
                    destination = STORAGE_DIR / f"{destination.stem}_{result.document.file_hash[:8]}{destination.suffix}"
                if not destination.exists():
                    destination.write_bytes(data)
                self._paths[result.document.file_hash] = destination
            self._set_job(job_id, "embedding", 5)
            self.reindex(result.document.file_hash, stage_callback=lambda stage, done: self._set_job(job_id, stage, done))
            terminal = "warning" if result.document.warnings else "ready"
            self._set_job(job_id, terminal, 7, status="complete",
                          document_id=result.document.file_hash, warnings=list(result.document.warnings))
        except Exception as exc:
            self._set_job(job_id, "failed", self._jobs[job_id].get("completed_stages", 0),
                          status="failed", error=str(exc))

    def jobs(self) -> list[dict[str, object]]:
        return [dict(job) for job in self._jobs.values()]

    def retry(self, job_id: str) -> dict[str, object]:
        if job_id not in self._job_payloads:
            raise KeyError(job_id)
        if self._jobs[job_id]["status"] == "running":
            return dict(self._jobs[job_id])
        self._jobs[job_id].update(stage="uploading", completed_stages=1, status="running", error=None)
        self._executor.submit(self._process_upload, job_id)
        return dict(self._jobs[job_id])

    def reindex(self, file_hash: str, stage_callback=None) -> dict[str, object]:
        document = next((item for item in self.registry.documents if item.file_hash == file_hash), None)
        if document is None:
            raise KeyError(file_hash)
        collection, model, _ = self._load_rag()
        if stage_callback:
            stage_callback("embedding", 5)
        texts = [block.text for block in document.blocks]
        encoded = model.encode(texts) if texts else []
        embeddings = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        if stage_callback:
            stage_callback("indexing", 6)
        # Both selectors are scoped to this document. Older collections may use
        # source_file only; newer entries always carry file_hash as well.
        collection.delete(where={"file_hash": file_hash})
        collection.delete(where={"source_file": document.source_file})
        if texts:
            collection.add(
                ids=[f"{file_hash}:{block.block_id}" for block in document.blocks],
                documents=texts,
                embeddings=embeddings,
                metadatas=[{"file_hash": file_hash, "source_file": document.source_file,
                            "file_type": document.file_type, "block_id": block.block_id,
                            "block_type": block.block_type, "section": block.section or "",
                            "page": block.page or 0, "sheet": block.sheet or "",
                            "row_index": block.row_index or 0, "parser_version": 3,
                            "index_version": 1} for block in document.blocks],
            )
        self._bm25_payload = rag_pipeline.build_bm25_index(collection, collection.count())
        return {"document_id": file_hash, "chunks": len(texts), "status": "ready"}

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

    def _source_dict(self, target) -> dict[str, object]:
        page = target.page
        return {
            "document": target.source_file, "file_hash": target.file_hash,
            "file_type": target.file_type, "block_id": target.block_id,
            "location": target.location_label, "page": page, "sheet": target.sheet,
            "row": target.row_index, "row_end": target.row_end, "cell_range": target.cell_range,
            "section": target.section, "text": target.evidence_text,
            "metadata": {"bbox": target.bbox, "table_index": target.table_index,
                         "paragraph_index": target.paragraph_index,
                         "preview_available": target.file_type in {"docx", "doc"}},
        }

    def chat_direct(self, message: str, document_hash: str | None, conversation_id: str | None) -> dict[str, object]:
        if not document_hash:
            raise ValueError("Direct Answer requires a selected document.")
        document = next((item for item in self.registry.documents if item.file_hash == document_hash), None)
        if document is None:
            raise KeyError(document_hash)
        started = time.perf_counter()
        result = canonical_rag.answer_direct(message, document)
        answer = result.answer
        if answer == canonical_rag.NO_EXPLICIT_EVIDENCE:
            answer = ("Je n’ai pas trouvé cette information explicitement dans le document sélectionné."
                      if result.query_language == "French" else
                      "I could not find this information explicitly in the selected document.")
        return {
            "answer": answer, "status": result.status, "result_type": result.result_type,
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
        started = time.perf_counter()
        language = detect_query_language(message)
        intent = classify_intent(message)
        topic_history = [item for item in history[-8:] if item.get("role") == "user"]
        prior = topic_history[-1].get("content", "") if topic_history else ""
        retrieval_query = message
        if intent == "FOLLOW_UP" and prior:
            retrieval_query = f"{prior} {message}"

        # Correct high-confidence corpus vocabulary typos before discovery.
        vocabulary: set[str] = set()
        for document in self.registry.documents:
            vocabulary.update(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", document.source_file))
            for block in document.blocks:
                vocabulary.update(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", " ".join(filter(None, (block.section, block.sheet)))))
        corrected = []
        for token in retrieval_query.split():
            bare = re.sub(r"[^A-Za-z0-9_-]", "", token)
            match = get_close_matches(bare, vocabulary, n=1, cutoff=.86) if len(bare) >= 4 else []
            corrected.append(match[0] if match and match[0].casefold() != bare.casefold() else token)
        retrieval_query = " ".join(corrected)

        discovery = self.registry.global_search(retrieval_query, 100)
        candidate_hashes = list(dict.fromkeys(hit.document.file_hash for hit in discovery))
        candidate_documents = [document for document in self.registry.documents
                               if document.file_hash in candidate_hashes]
        generic_terms = {
            "give", "show", "list", "all", "every", "what", "which", "where", "when",
            "with", "from", "about", "please", "test", "tests", "case", "cases", "document",
            "documents", "system", "systems", "the", "and", "for", "les", "des", "tous", "toutes",
        }
        query_terms = {token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", retrieval_query)}
        anchor_terms = query_terms - generic_terms
        if anchor_terms:
            anchored = []
            for document in candidate_documents:
                searchable = " ".join([document.source_file, *(block.text for block in document.blocks)]).casefold()
                if all(term in searchable for term in anchor_terms):
                    anchored.append(document)
            if anchored:
                candidate_documents = anchored

        if not anchor_terms and query_terms & {"test", "tests", "case", "cases"} and len(candidate_documents) > 1:
            names = ", ".join(document.source_file for document in candidate_documents[:4])
            answer = (f"J’ai trouvé des cas de test dans plusieurs documents ({names}). Lequel souhaitez-vous explorer ?"
                      if language == "French" else
                      f"I found test cases in several documents ({names}). Which one do you want to explore?")
            return {
                "answer": answer, "status": "CLARIFICATION", "result_type": "CLARIFICATION",
                "language": language, "method": "corpus_clarification",
                "conversation_id": conversation_id or str(uuid.uuid4()), "sources": [],
                "suggestions": [], "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }

        evidence = []
        seen: set[tuple[str, str]] = set()
        per_document_limit = 200 if intent in {"EXHAUSTIVE_LIST", "TABLE_QUERY"} else 8
        for document in candidate_documents:
            for block in retrieve_evidence(document, retrieval_query, intent)[:per_document_limit]:
                identity = (block.file_hash, block.text.casefold())
                if identity not in seen:
                    seen.add(identity)
                    evidence.append(block)

        if not evidence:
            answer = ("Je n’ai trouvé aucune preuve pertinente dans les documents importés. "
                      "Pouvez-vous préciser le système, le processus ou le type d’information recherché ?"
                      if language == "French" else
                      "I couldn't find relevant evidence in the uploaded documents. Could you clarify the system, process, or type of information you mean?")
        else:
            limit = 240 if intent in {"EXHAUSTIVE_LIST", "TABLE_QUERY"} else 30
            evidence = evidence[:limit]
            prompt = build_conversational_prompt(message, language, intent, evidence, history)
            answer = self.generation_provider.generate(prompt)
        source_payload = [self._source_dict(
            self.registry.source_target(block.file_hash, block.block_id)
        ) for block in evidence]
        return {
            "answer": answer, "status": "ANSWER" if evidence else "CLARIFICATION",
            "result_type": intent, "language": language,
            "method": "gemini_global_grounded_generation",
            "conversation_id": conversation_id or str(uuid.uuid4()), "sources": source_payload,
            "suggestions": [], "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }


@lru_cache(maxsize=1)
def get_runtime() -> CorporateBrainRuntime:
    return CorporateBrainRuntime()
