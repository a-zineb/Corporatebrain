from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
from threading import RLock
import time

import canonical_rag
from document_normalizer import CANONICAL_SCHEMA_VERSION, CanonicalDocument, SUPPORTED_EXTENSIONS


LIGHTWEIGHT_INDEX_VERSION = 1


@dataclass
class CatalogEntry:
    file_hash: str
    filename: str
    file_type: str
    file_size: int
    modified_ns: int
    path: str
    status: str = "AVAILABLE"
    title: str = ""
    version: str = ""
    application: str | None = None
    filiale: str | None = None
    terms: list[str] | None = None
    sheet_names: list[str] | None = None
    blocks: int = 0
    warnings: list[str] | None = None
    lightweight_index_version: int = LIGHTWEIGHT_INDEX_VERSION
    canonical_schema_version: int = CANONICAL_SCHEMA_VERSION

    def search_text(self) -> str:
        return " ".join([self.filename, self.title, self.version, self.application or "",
                         self.filiale or "", *(self.terms or []), *(self.sheet_names or [])]).casefold()


class LightweightCatalog:
    def __init__(self, storage: Path, cache_path: Path) -> None:
        self.storage = storage
        self.cache_path = cache_path
        self.entries: dict[str, CatalogEntry] = {}
        self.load_time_ms = 0.0
        self._lock = RLock()
        self.refresh()

    def refresh(self) -> None:
        started = time.perf_counter()
        self.storage.mkdir(parents=True, exist_ok=True)
        cached_by_path: dict[str, dict] = {}
        if self.cache_path.is_file():
            try:
                cached_by_path = {item["path"]: item for item in json.loads(
                    self.cache_path.read_text(encoding="utf-8"))}
            except (OSError, ValueError, KeyError):
                cached_by_path = {}
        entries: dict[str, CatalogEntry] = {}
        for path in sorted(self.storage.iterdir()):
            if not path.is_file() or path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                continue
            stat = path.stat()
            cached = cached_by_path.get(str(path.resolve()))
            if cached and cached.get("file_size") == stat.st_size and cached.get("modified_ns") == stat.st_mtime_ns:
                entry = CatalogEntry(**cached)
            else:
                digest = sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                upper = path.name.upper()
                entry = CatalogEntry(
                    digest.hexdigest(), path.name, path.suffix.casefold().lstrip("."), stat.st_size,
                    stat.st_mtime_ns, str(path.resolve()), title=path.stem,
                    application=next((value for value in ("MZ", "KPSA") if value in upper), None),
                    filiale=next((value for value in ("OCM", "OEG", "OJO", "OCI") if value in upper), None),
                    terms=[token for token in path.stem.replace("_", " ").replace("-", " ").split() if len(token) > 2],
                    warnings=[], sheet_names=[],
                )
            entries[entry.file_hash] = entry
        with self._lock:
            self.entries = entries
            self._persist()
        self.load_time_ms = (time.perf_counter() - started) * 1000

    def _persist(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps([asdict(item) for item in self.entries.values()],
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.cache_path)

    def update_hydrated(self, document: CanonicalDocument, state: str) -> None:
        with self._lock:
            entry = self.entries[document.file_hash]
            entry.status = "READY_WITH_WARNINGS" if document.warnings else state
            entry.blocks = len(document.blocks)
            entry.warnings = list(document.warnings)
            entry.sheet_names = sorted({block.sheet for block in document.blocks if block.sheet})
            entry.terms = sorted({token.casefold() for block in document.blocks[:200]
                                  for token in (block.section or "").split() if len(token) > 2})[:300]
            self._persist()

    def mark_failed(self, file_hash: str, message: str) -> None:
        with self._lock:
            entry = self.entries[file_hash]
            entry.status = "FAILED"
            entry.warnings = [message]
            self._persist()


class DocumentHydrator:
    def __init__(self, catalog: LightweightCatalog, cache_dir: Path,
                 memory_size: int = 8, max_workers: int = 3, on_evict=None) -> None:
        self.catalog = catalog
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_size = max(1, memory_size)
        self.executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="hydrate")
        self._memory: OrderedDict[str, CanonicalDocument] = OrderedDict()
        self._inflight: dict[str, Future[CanonicalDocument]] = {}
        self._lock = RLock()
        self.on_evict = on_evict
        self.metrics = {"hydrations": 0, "memory_hits": 0, "persistent_hits": 0,
                        "misses": 0, "failures": 0, "hydration_ms": []}

    def _cache_path(self, file_hash: str) -> Path:
        return self.cache_dir / f"{file_hash}.schema-{CANONICAL_SCHEMA_VERSION}.pickle"

    def hydrate(self, file_hash: str) -> CanonicalDocument:
        with self._lock:
            cached = self._memory.pop(file_hash, None)
            if cached is not None:
                self._memory[file_hash] = cached
                self.metrics["memory_hits"] += 1
                return cached
            future = self._inflight.get(file_hash)
            if future is None:
                if file_hash not in self.catalog.entries:
                    raise KeyError(file_hash)
                future = self.executor.submit(self._hydrate_uncached, file_hash)
                self._inflight[file_hash] = future
        try:
            document = future.result()
        finally:
            with self._lock:
                if self._inflight.get(file_hash) is future:
                    self._inflight.pop(file_hash, None)
        with self._lock:
            self._remember(file_hash, document)
        return document

    def _remember(self, file_hash: str, document: CanonicalDocument) -> None:
        self._memory[file_hash] = document
        while len(self._memory) > self.memory_size:
            evicted_hash, _ = self._memory.popitem(last=False)
            if self.on_evict:
                self.on_evict(evicted_hash)

    def prefetch(self, file_hash: str) -> None:
        with self._lock:
            if file_hash in self._memory or file_hash in self._inflight:
                return
            if file_hash not in self.catalog.entries:
                raise KeyError(file_hash)
            future = self.executor.submit(self._hydrate_uncached, file_hash)
            self._inflight[file_hash] = future
        def finish(completed: Future[CanonicalDocument]) -> None:
            try:
                document = completed.result()
                with self._lock:
                    self._remember(file_hash, document)
            finally:
                with self._lock:
                    self._inflight.pop(file_hash, None)
        future.add_done_callback(finish)

    def _hydrate_uncached(self, file_hash: str) -> CanonicalDocument:
        started = time.perf_counter()
        entry = self.catalog.entries[file_hash]
        entry.status = "HYDRATING"
        cache_path = self._cache_path(file_hash)
        try:
            if cache_path.is_file():
                with cache_path.open("rb") as stream:
                    document = pickle.load(stream)
                if isinstance(document, CanonicalDocument) and document.file_hash == file_hash:
                    self.metrics["persistent_hits"] += 1
                    self.catalog.update_hydrated(document, "HYDRATED")
                    return document
            self.metrics["misses"] += 1
            data = Path(entry.path).read_bytes()
            outcome = canonical_rag.normalize_with_gate(data, entry.filename)
            if outcome.document is None:
                raise ValueError("The document could not be hydrated reliably.")
            document = outcome.document
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(document, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(cache_path)
            self.metrics["hydrations"] += 1
            self.catalog.update_hydrated(document, "HYDRATED")
            return document
        except Exception as exc:
            self.catalog.mark_failed(file_hash, str(exc))
            self.metrics["failures"] += 1
            raise
        finally:
            self.metrics["hydration_ms"].append((time.perf_counter() - started) * 1000)

    def invalidate(self, file_hash: str, persistent: bool = True) -> None:
        with self._lock:
            self._memory.pop(file_hash, None)
        if persistent:
            self._cache_path(file_hash).unlink(missing_ok=True)

    @property
    def memory_hashes(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._memory)
