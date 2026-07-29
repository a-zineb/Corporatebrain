"""Read-only validation for the versioned Corporate Brain benchmark contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import chromadb

from rag_pipeline import RAGConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"


def canonical(value: object) -> str:
    """Serialize a fingerprint input deterministically."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_cases() -> list[dict[str, object]]:
    """Load the versioned JSONL seed without changing it."""

    return [
        json.loads(line)
        for line in (BENCHMARK_DIR / "corporatebrain.v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def current_fingerprint() -> dict[str, object]:
    """Calculate the documented read-only collection fingerprint."""

    config = RAGConfig()
    collection = chromadb.PersistentClient(path=config.chroma_path).get_collection(config.collection_name)
    data = collection.get(include=["documents", "metadatas"])
    rows = [
        {
            "chunk_id": chunk_id,
            "content_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
            "metadata": metadata,
        }
        for chunk_id, document, metadata in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    rows.sort(key=lambda row: row["chunk_id"])
    corpus_sha256 = hashlib.sha256("\n".join(canonical(row) for row in rows).encode("utf-8")).hexdigest()
    metadata_sha256 = hashlib.sha256(
        "\n".join(canonical(row["metadata"]) for row in rows).encode("utf-8")
    ).hexdigest()
    retrieval_config = {
        "vector_candidate_count": config.vector_candidate_count,
        "bm25_candidate_count": config.bm25_candidate_count,
        "rrf_k": config.rrf_k,
        "default_top_k": config.default_top_k,
        "production_top_k": config.production_top_k,
        "min_results_before_relax": config.min_results_before_relax,
    }
    runtime = {
        "collection_metadata": collection.metadata,
        "collection_count": collection.count(),
        "corpus_sha256": corpus_sha256,
        "metadata_sha256": metadata_sha256,
        "retrieval_config": {
            "chroma_path": config.chroma_path,
            "collection_name": config.collection_name,
            "embedding_model_name": config.embedding_model_name,
            **retrieval_config,
        },
    }
    return {
        "rows": rows,
        "collection_metadata": collection.metadata,
        "chunk_count": collection.count(),
        "embedding_model_name": config.embedding_model_name,
        "retrieval_config": retrieval_config,
        "corpus_sha256": corpus_sha256,
        "metadata_sha256": metadata_sha256,
        "runtime_fingerprint_sha256": hashlib.sha256(canonical(runtime).encode("utf-8")).hexdigest(),
    }


class BenchmarkContractTests(unittest.TestCase):
    """Validate the benchmark contract against the active corpus, read-only."""

    def setUp(self) -> None:
        self.schema = json.loads((BENCHMARK_DIR / "schema.json").read_text(encoding="utf-8"))
        self.manifest = json.loads(
            (BENCHMARK_DIR / "corporatebrain.v1.manifest.json").read_text(encoding="utf-8")
        )
        self.cases = load_cases()

    def test_schema_declares_explicit_behavior_and_anchor_contract(self) -> None:
        required = set(self.schema["required"])
        self.assertIn("expected_behavior", required)
        self.assertIn("chunk_id", self.schema["$defs"]["anchor"]["required"])
        self.assertIn("content_sha256", self.schema["$defs"]["anchor"]["required"])
        self.assertNotIn("claim_coverage", self.schema["properties"])

    def test_seed_has_24_unique_cases_and_required_coverage(self) -> None:
        self.assertEqual(len(self.cases), 24)
        identifiers = [case["id"] for case in self.cases]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        categories = {case["category"] for case in self.cases}
        self.assertEqual(
            categories,
            {
                "direct", "metadata_filtered", "acronym", "synonym", "typo", "multilingual",
                "conversational_follow_up", "unanswerable", "ambiguous",
            },
        )

    def test_answerability_and_expected_behavior_rules(self) -> None:
        for case in self.cases:
            mode = case["expected_behavior"]["mode"]
            if case["answerability"] == "answerable":
                self.assertEqual(mode, "answer")
                self.assertTrue(case.get("expected_answer"))
                self.assertTrue(case["relevance"])
                self.assertTrue(case["acceptable_citations"])
            elif case["answerability"] == "unanswerable":
                self.assertEqual(mode, "refuse_no_coverage")
                self.assertEqual(case["expected_behavior"].get("source_display"), "none")
                self.assertEqual(case["relevance"], [])
                self.assertEqual(case["acceptable_citations"], [])
            else:
                self.assertEqual(mode, "request_clarification")

    def test_manifest_and_every_chunk_annotation_match_active_collection(self) -> None:
        fingerprint = current_fingerprint()
        self.assertEqual(fingerprint["chunk_count"], self.manifest["collection_identity"]["chunk_count"])
        self.assertEqual(fingerprint["collection_metadata"], self.manifest["collection_identity"]["collection_metadata"])
        for name in ("embedding_model_name", "retrieval_config", "corpus_sha256", "metadata_sha256", "runtime_fingerprint_sha256"):
            self.assertEqual(fingerprint[name], self.manifest[name])

        by_id = {row["chunk_id"]: row for row in fingerprint["rows"]}
        metadata_values = {
            key: {row["metadata"].get(key) for row in fingerprint["rows"]}
            for key in {key for row in fingerprint["rows"] for key in row["metadata"]}
        }
        for case in self.cases:
            for key, value in case["metadata_filter"].items():
                self.assertIn(key, metadata_values)
                self.assertIn(value, metadata_values[key])
            anchors = {anchor["chunk_id"]: anchor for anchor in case["relevance"]}
            for anchor in anchors.values():
                record = by_id[anchor["chunk_id"]]
                self.assertEqual(anchor["anchor_type"], "chunk_id")
                self.assertEqual(anchor["content_sha256"], record["content_sha256"])
                self.assertEqual(anchor["source_file"], record["metadata"]["source_file"])
                self.assertEqual(anchor["location"], record["metadata"]["location"])
            for citation in case["acceptable_citations"]:
                self.assertIn(citation["chunk_id"], anchors)
                self.assertEqual(citation["content_sha256"], anchors[citation["chunk_id"]]["content_sha256"])


if __name__ == "__main__":
    unittest.main()
