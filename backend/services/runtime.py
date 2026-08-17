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
from backend.services.lazy_documents import DocumentHydrator, LightweightCatalog
from mvp_services import PreparedDocumentRegistry, detect_query_language


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
STORAGE_DIR = Path(os.getenv("CORPORATE_BRAIN_STORAGE_DIR", ROOT / "doc_storage_v2"))
CHROMA_PATH = str(Path(os.getenv("CORPORATE_BRAIN_CHROMA_PATH", ROOT / "chroma_db_local_v2")))
COLLECTION_NAME = os.getenv("CORPORATE_BRAIN_COLLECTION", "documents")
PREVIEW_DIR = ROOT / ".run" / "previews"
CATALOG_PATH = ROOT / ".run" / "document_catalog.json"
PREPARED_CACHE_DIR = ROOT / ".run" / "prepared"


class CorporateBrainRuntime:
    """Thin reusable facade over the existing canonical and RAG functions."""

    def __init__(self, generation_provider: GenerationProvider | None = None) -> None:
        started = time.perf_counter()
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
        hydration_workers = int(os.getenv("MAX_CONCURRENT_HYDRATIONS", "3"))
        self._executor = ThreadPoolExecutor(max_workers=max(1, hydration_workers), thread_name_prefix="corporate-work")
        self.catalog = LightweightCatalog(STORAGE_DIR, CATALOG_PATH)
        self.hydrator = DocumentHydrator(
            self.catalog, PREPARED_CACHE_DIR,
            int(os.getenv("HYDRATED_DOCUMENT_CACHE_SIZE", "8")),
            hydration_workers,
            self.registry.remove,
        )
        self.lazy_enabled = os.getenv("LAZY_HYDRATION_ENABLED", "true").casefold() in {"1", "true", "yes", "on"}
        self._paths = {item.file_hash: Path(item.path) for item in self.catalog.entries.values()}
        if not self.lazy_enabled:
            for file_hash in self.catalog.entries:
                try:
                    self.hydrate_document(file_hash)
                except Exception:
                    continue
        self.startup_ms = (time.perf_counter() - started) * 1000

    def refresh_documents(self) -> None:
        self.catalog.refresh()
        self._paths = {item.file_hash: Path(item.path) for item in self.catalog.entries.values()}

    def hydrate_document(self, file_hash: str):
        existing = next((item for item in self.registry.documents if item.file_hash == file_hash), None)
        if existing is not None:
            return existing
        document = self.hydrator.hydrate(file_hash)
        self.registry.add(document)
        return document

    def prefetch(self, file_hash: str) -> None:
        self.hydrator.prefetch(file_hash)

    def documents(self) -> list[dict[str, object]]:
        return [{
            "id": item.file_hash, "name": item.filename, "type": item.file_type,
            "status": "FAILED" if item.status == "FAILED" else
                      "WARNING" if item.status == "READY_WITH_WARNINGS" else
                      "PREPARING" if item.status == "HYDRATING" else "READY",
            "lifecycle_state": item.status, "blocks": item.blocks,
            "warnings": item.warnings or [], "filiale": item.filiale,
            "application": item.application, "size": item.file_size,
        } for item in sorted(self.catalog.entries.values(), key=lambda value: value.filename.casefold())]

    def document_path(self, file_hash: str) -> Path:
        path = self._paths.get(file_hash)
        if path is None or not path.is_file() or path.parent.resolve() != STORAGE_DIR.resolve():
            raise KeyError(file_hash)
        return path

    def preview_path(self, file_hash: str) -> Path:
        path = self.document_path(file_hash)
        document = self.hydrate_document(file_hash)
        if document is None or document.file_type not in {"docx", "doc"}:
            raise KeyError(file_hash)
        preview, _ = self.previews.ensure(path, document)
        return preview

    def preview_info(self, file_hash: str, block_id: str) -> dict[str, object]:
        path = self.document_path(file_hash)
        document = self.hydrate_document(file_hash)
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
            safe_name = Path(name).name
            file_hash = __import__("hashlib").sha256(data).hexdigest()
            destination = STORAGE_DIR / safe_name
            if destination.exists() and destination.read_bytes() != data:
                destination = STORAGE_DIR / f"{destination.stem}_{file_hash[:8]}{destination.suffix}"
            destination.write_bytes(data)
            self.refresh_documents()
            return next(item for item in self.documents() if item["id"] == file_hash)

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
            item = self.upload(name, data)
            self._set_job(job_id, "ready", 2, total_stages=2, status="complete",
                          document_id=item["id"], warnings=[])
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
        self.hydrator.invalidate(file_hash)
        self.registry.remove(file_hash)
        document = self.hydrate_document(file_hash)
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
        try:
            import chromadb
            collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION_NAME)
        except Exception:
            collection = None
        if collection is not None:
            collection.delete(where={"file_hash": file_hash})
            collection.delete(where={"source_file": path.name})
            self._bm25_payload = None
        path.unlink()
        self.hydrator.invalidate(file_hash)
        self.registry.remove(file_hash)
        (PREVIEW_DIR / f"{file_hash}.pdf").unlink(missing_ok=True)
        (PREVIEW_DIR / f"{file_hash}.pages.json").unlink(missing_ok=True)
        self.refresh_documents()

    def search(self, query: str, limit: int = 50) -> list[dict[str, object]]:
        terms = {token.casefold() for token in re.findall(r"[\w-]{2,}", query)}
        ranked = sorted(
            ((len(terms & set(entry.search_text().split())), entry) for entry in self.catalog.entries.values()),
            key=lambda pair: (-pair[0], pair[1].filename.casefold()),
        )
        for score, entry in ranked[:min(6, limit)]:
            if score:
                try:
                    self.hydrate_document(entry.file_hash)
                except Exception:
                    continue
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
        self.hydrate_document(file_hash)
        target = self.registry.source_target(file_hash, block_id)
        if target is None:
            raise KeyError((file_hash, block_id))
        return self._source_dict(target)

    def first_source(self, file_hash: str) -> dict[str, object]:
        document = self.hydrate_document(file_hash)
        if not document.blocks:
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
        document = self.hydrate_document(document_hash)
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

        # Correct high-confidence lightweight-catalog vocabulary typos before discovery.
        vocabulary: set[str] = set()
        for entry in self.catalog.entries.values():
            vocabulary.update(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", entry.search_text()))
        corrected = []
        for token in retrieval_query.split():
            bare = re.sub(r"[^A-Za-z0-9_-]", "", token)
            match = get_close_matches(bare, vocabulary, n=1, cutoff=.86) if len(bare) >= 4 else []
            corrected.append(match[0] if match and match[0].casefold() != bare.casefold() else token)
        retrieval_query = " ".join(corrected)

        query_tokens = {token.casefold() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", retrieval_query)}
        ranked_entries = []
        for entry in self.catalog.entries.values():
            profile_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", entry.search_text()))
            overlap = query_tokens & profile_tokens
            if overlap:
                ranked_entries.append((len(overlap) / max(len(query_tokens), 1), entry))
        ranked_entries.sort(key=lambda pair: (-pair[0], pair[1].filename.casefold()))
        max_documents = int(os.getenv("AI_DISCOVERY_MAX_DOCUMENTS", "6"))
        if intent in {"EXHAUSTIVE_LIST", "TABLE_QUERY", "COMPARISON", "CORPUS_OVERVIEW"}:
            max_documents = max(max_documents, 12)
        candidate_entries = [entry for score, entry in ranked_entries if score >= .12][:max_documents]
        if not candidate_entries:
            candidate_entries = [entry for _, entry in ranked_entries[:max_documents]]
        if not candidate_entries:
            try:
                _, _, bm25_payload = self._load_rag()
                indexed_documents = bm25_payload[1]
                indexed_metadatas = bm25_payload[2]
                source_scores: dict[str, int] = {}
                for text, metadata in zip(indexed_documents, indexed_metadatas):
                    overlap = len(query_tokens & set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.casefold())))
                    source_file = str(metadata.get("source_file", ""))
                    if overlap and source_file:
                        source_scores[source_file] = max(source_scores.get(source_file, 0), overlap)
                by_name = {entry.filename: entry for entry in self.catalog.entries.values()}
                candidate_entries = [by_name[name] for name, _ in sorted(
                    source_scores.items(), key=lambda pair: (-pair[1], pair[0].casefold())
                ) if name in by_name][:max_documents]
            except Exception:
                candidate_entries = []
        if not candidate_entries and self.registry.documents:
            candidate_documents = list(self.registry.documents)[:max_documents]
        else:
            candidate_documents = []
        if not candidate_entries and intent in {"COMPARISON", "CORPUS_OVERVIEW"}:
            candidate_entries = list(self.catalog.entries.values())[:max_documents]
        generic_terms = {
            "give", "show", "list", "all", "every", "what", "which", "where", "when",
            "with", "from", "about", "please", "test", "tests", "case", "cases", "document",
            "documents", "system", "systems", "the", "and", "for", "les", "des", "tous", "toutes",
        }
        query_terms = query_tokens
        anchor_terms = query_terms - generic_terms

        if not anchor_terms and query_terms & {"test", "tests", "case", "cases"} and len(candidate_entries) > 1:
            names = ", ".join(entry.filename for entry in candidate_entries[:4])
            answer = (f"J’ai trouvé des cas de test dans plusieurs documents ({names}). Lequel souhaitez-vous explorer ?"
                      if language == "French" else
                      f"I found test cases in several documents ({names}). Which one do you want to explore?")
            return {
                "answer": answer, "status": "CLARIFICATION", "result_type": "CLARIFICATION",
                "language": language, "method": "corpus_clarification",
                "conversation_id": conversation_id or str(uuid.uuid4()), "sources": [],
                "suggestions": [], "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }

        futures = [self._executor.submit(self.hydrate_document, entry.file_hash)
                   for entry in candidate_entries]
        for future in futures:
            try:
                candidate_documents.append(future.result())
            except Exception:
                continue
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

    def debug_metrics(self) -> dict[str, object]:
        return {"startup_ms": round(self.startup_ms, 3),
                "catalog_load_ms": round(self.catalog.load_time_ms, 3),
                "catalog_entries": len(self.catalog.entries),
                "lazy_hydration_enabled": self.lazy_enabled,
                "hydrated_in_memory": len(self.hydrator.memory_hashes),
                **self.hydrator.metrics}


@lru_cache(maxsize=1)
def get_runtime() -> CorporateBrainRuntime:
    return CorporateBrainRuntime()
