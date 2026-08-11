"""Versioned, ingestion-stable corpus fingerprinting.

Version 1 remains implemented by the historical benchmark contract.  This
module provides an intentionally separate version 2 contract that excludes
ingestion-derived identifiers and timestamps.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

FINGERPRINT_SCHEMA_VERSION = 2
_CHUNK_SUFFIX = re.compile(r"_chunk_(\d+)$")
_STABLE_METADATA_FIELDS = frozenset(
    {
        "source_file",
        "file_hash",
        "application",
        "geographical_entity",
        "location",
        "page",
        "sheet",
        "document_identifier",
    }
)


class FingerprintSchemaError(ValueError):
    """Raised when a fingerprint uses an unsupported schema version."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordinal(record: Mapping[str, Any]) -> int | None:
    metadata = record.get("metadata") or {}
    value = metadata.get("chunk_ordinal")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    record_id = str(record.get("chunk_id", record.get("id", "")))
    match = _CHUNK_SUFFIX.search(record_id)
    return int(match.group(1)) if match else None


def _stable_metadata(metadata: Mapping[str, Any], source_file: str, file_hash: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(_STABLE_METADATA_FIELDS):
        value = metadata.get(key)
        if key == "source_file":
            value = value or source_file
        elif key == "file_hash":
            value = value or file_hash
        if value is not None:
            result[key] = value
    return result


def canonical_records_v2(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic records using only stable corpus identity fields."""

    prepared: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        text = str(record.get("document", record.get("text", "")))
        source_file = str(metadata.get("source_file") or record.get("source_file") or "")
        file_hash = str(metadata.get("file_hash") or record.get("file_hash") or "")
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        prepared.append(
            {
                "source_file": source_file,
                "file_hash": file_hash,
                "chunk_ordinal": _ordinal(record),
                "content_sha256": content_sha256,
                "stable_metadata": _stable_metadata(metadata, source_file, file_hash),
            }
        )

    prepared.sort(
        key=lambda row: (
            row["source_file"],
            row["file_hash"],
            row["chunk_ordinal"] is None,
            row["chunk_ordinal"] if row["chunk_ordinal"] is not None else -1,
            row["content_sha256"],
        )
    )
    # Records without an ordinal receive a deterministic ordinal after sorting;
    # this makes insertion order irrelevant while retaining chunk-boundary changes.
    counters: dict[tuple[str, str], int] = {}
    for row in prepared:
        if row["chunk_ordinal"] is None:
            key = (row["source_file"], row["file_hash"])
            row["chunk_ordinal"] = counters.get(key, 0)
            counters[key] = row["chunk_ordinal"] + 1
    return prepared


def fingerprint_v2(records: Iterable[Mapping[str, Any]], *, runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compute a schema-v2 corpus, metadata, and optional runtime fingerprint."""

    canonical_records = canonical_records_v2(records)
    corpus_payload = "\n".join(_canonical(row) for row in canonical_records)
    metadata_payload = "\n".join(
        _canonical(
            {
                "source_file": row["source_file"],
                "file_hash": row["file_hash"],
                "chunk_ordinal": row["chunk_ordinal"],
                "content_sha256": row["content_sha256"],
                "stable_metadata": row["stable_metadata"],
            }
        )
        for row in canonical_records
    )
    result: dict[str, Any] = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "chunk_count": len(canonical_records),
        "corpus_sha256": hashlib.sha256(corpus_payload.encode("utf-8")).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_payload.encode("utf-8")).hexdigest(),
        "records": canonical_records,
    }
    if runtime is not None:
        runtime_payload = {"fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION, **dict(runtime)}
        runtime_payload["corpus_sha256"] = result["corpus_sha256"]
        runtime_payload["metadata_sha256"] = result["metadata_sha256"]
        result["runtime_fingerprint_sha256"] = hashlib.sha256(_canonical(runtime_payload).encode("utf-8")).hexdigest()
    return result


def compare_fingerprints(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare fingerprints, refusing to compare v1 and v2 implicitly."""

    left_version = left.get("fingerprint_schema_version", 1)
    right_version = right.get("fingerprint_schema_version", 1)
    for version in (left_version, right_version):
        if version not in (1, FINGERPRINT_SCHEMA_VERSION):
            raise FingerprintSchemaError(f"Unsupported fingerprint schema version: {version}")
    if left_version != right_version:
        return {"status": "NOT_COMPARABLE", "reason": "fingerprint_schema_version_mismatch"}
    return {
        "status": "MATCH" if left == right else "MISMATCH",
        "fingerprint_schema_version": left_version,
    }
