"""Deterministic evaluator checks using only shared-runtime test doubles."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rag_evaluator
import rag_pipeline


class FakeVector:
    def tolist(self):
        return [0.5, 0.5]


class FakeEmbeddingModel:
    def encode(self, _query):
        return FakeVector()


class FakeCollection:
    def __init__(self, documents, metadatas):
        self.documents = documents
        self.metadatas = metadatas
        self.calls = []

    def get(self, *, include):
        self.include = include
        return {"documents": self.documents, "metadatas": self.metadatas}

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "ids": [["chunk-1"]],
            "documents": [[self.documents[0]]],
            "metadatas": [[self.metadatas[0]]],
            "distances": [[0.1]],
        }


class FakeGenerator:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [{"message": {"content": self.response}}]
        return {"message": {"content": kwargs["messages"][0]["content"].split("QUESTION :")[-1].split("QUESTION REFORMULÉE")[0].strip()}}


def case_for(content_hash, *, answerability="answerable", response_mode="answer"):
    relevance = [] if answerability == "unanswerable" else [{
        "content_sha256": content_hash,
        "label": 2,
    }]
    citations = [] if answerability == "unanswerable" else [{"content_sha256": content_hash}]
    return {
        "id": "case-1",
        "question": "alpha",
        "conversation": [],
        "language": "fr",
        "metadata_filter": {},
        "relevance": relevance,
        "acceptable_citations": citations,
        "answerability": answerability,
        "expected_behavior": {"mode": response_mode, "source_display": "none" if response_mode != "answer" else "expected"},
    }


class DeterministicEvaluatorTests(unittest.TestCase):
    """Verify runner orchestration, formulas, and report formats without Ollama."""

    def setUp(self):
        self.documents = ["alpha policy", "other policy"]
        self.metadatas = [{"source_file": "policy.pdf", "location": "Page 1"}, {"source_file": "other.pdf", "location": "Page 2"}]
        collection = FakeCollection(self.documents, self.metadatas)
        bm25, docs, metas = rag_pipeline.build_bm25_index(collection, len(self.documents))
        self.runtime = rag_evaluator.EvaluationRuntime(
            collection, FakeEmbeddingModel(), bm25, docs, metas, rag_pipeline.RAGConfig()
        )
        self.content_hash = hashlib.sha256(self.documents[0].encode("utf-8")).hexdigest()

    def test_runner_uses_shared_runtime_and_calculates_metrics(self):
        result = rag_evaluator.evaluate_case(
            case_for(self.content_hash), self.runtime, FakeGenerator("Answer [SOURCE 1]")
        )
        self.assertEqual(result["trace"].rewritten_query, "alpha")
        self.assertEqual(result["metrics"]["recall_at_k"], 1.0)
        self.assertEqual(result["metrics"]["precision_at_k"], 1 / 15)
        self.assertEqual(result["metrics"]["hit_rate_at_k"], 1.0)
        self.assertEqual(result["metrics"]["mrr"], 1.0)
        self.assertEqual(result["metrics"]["ndcg_at_k"], 1.0)
        self.assertTrue(result["metrics"]["citation_valid"])
        self.assertTrue(result["metrics"]["expected_source_match"])
        self.assertIsNone(result["metrics"]["refusal_correct"])
        self.assertEqual(self.runtime.collection.calls[0]["n_results"], 10)

    def test_refusal_and_report_serialization_are_deterministic(self):
        result = rag_evaluator.evaluate_case(
            case_for(self.content_hash, answerability="unanswerable", response_mode="refuse_no_coverage"),
            self.runtime,
            FakeGenerator("I cannot find this in the document context. [SOURCE 1]"),
        )
        self.assertTrue(result["metrics"]["refusal_correct"])
        self.assertFalse(result["trace"].citations.display_sources)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            rag_evaluator.write_reports([result], output)
            self.assertTrue((output / "cases.json").exists())
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "summary.md").exists())
            self.assertTrue((output / "forensics" / "case-1.json").exists())
            self.assertTrue((output / "forensics" / "case-1.md").exists())
            payload = json.loads((output / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["case_id"], "case-1")
            self.assertTrue(payload[0]["trace"]["refusal_detected"])

    def test_evaluator_contains_no_duplicate_retrieval_or_generation_implementation(self):
        source = Path(rag_evaluator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("run_retrieval", function_names)
        self.assertNotIn("generate_answer", function_names)
        self.assertNotIn("BM25Okapi", source)
        self.assertNotIn("collection.query(", source)
        for name in (
            "rag_pipeline.build_bm25_index(", "rag_pipeline.rewrite_query(",
            "rag_pipeline.normalize_chroma_filter(",
            "rag_pipeline.hybrid_search(", "rag_pipeline.build_production_prompt(",
            "rag_pipeline.stream_generate(", "rag_pipeline.select_display_sources(",
        ):
            self.assertIn(name, source)

    def test_offline_embedding_loader_uses_cached_production_model(self):
        cached_model = object()
        with patch("rag_evaluator.SentenceTransformer", return_value=cached_model) as loader:
            self.assertIs(
                rag_evaluator.load_offline_embedding_model("paraphrase-multilingual-MiniLM-L12-v2"),
                cached_model,
            )
        loader.assert_called_once_with("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)

    def test_offline_embedding_loader_fails_clearly_when_cache_is_missing(self):
        with patch("rag_evaluator.SentenceTransformer", side_effect=OSError("missing cache")):
            with self.assertRaisesRegex(RuntimeError, "Offline evaluation requires cached embedding model"):
                rag_evaluator.load_offline_embedding_model("paraphrase-multilingual-MiniLM-L12-v2")

    def test_evaluator_uses_shared_filter_normalization_for_multi_field_cases(self):
        case = case_for(self.content_hash)
        case["metadata_filter"] = {"application": "KPSA", "geographical_entity": "OCM"}
        rag_evaluator.evaluate_case(case, self.runtime, FakeGenerator("Answer [SOURCE 1]"))
        self.assertEqual(
            self.runtime.collection.calls[0]["where"],
            {"$and": [{"application": "KPSA"}, {"geographical_entity": "OCM"}]},
        )


if __name__ == "__main__":
    unittest.main()
