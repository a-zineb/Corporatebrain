"""Deterministic parity checks against the current app.py BM25 behavior."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import unittest

from rank_bm25 import BM25Okapi

import rag_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeCollection:
    """Minimal read-only collection contract for deterministic BM25 tests."""

    def __init__(self, documents, metadatas):
        self.documents = documents
        self.metadatas = metadatas

    def get(self, *, include):
        self.last_include = include
        return {"documents": self.documents, "metadatas": self.metadatas}

    def query(self, **_kwargs):
        return {"documents": [[]], "metadatas": [[]]}


class FakeEmbeddingModel:
    """Satisfies app.py's legacy vector call while returning no candidates."""

    def encode(self, _query):
        return FakeEmbeddingVector()


class FakeEmbeddingVector:
    """Minimal NumPy-like result expected by the active hybrid-search code."""

    def tolist(self):
        return [0.0, 1.0]


def load_legacy_function(name, namespace):
    """Load one exact function definition from app.py without importing it."""

    app_tree = ast.parse((PROJECT_ROOT / "app.py").read_text(encoding="utf-8"))
    function = next(node for node in app_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function = copy.deepcopy(function)
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace[name]


class BM25ParityTests(unittest.TestCase):
    """Compare shared helpers with the legacy code currently in app.py."""

    def setUp(self):
        self.documents = [
            "Alpha beta beta",
            "beta gamma",
            "ALPHA alpha gamma",
            "delta only",
        ]
        self.metadatas = [
            {"application": "KPSA", "geographical_entity": "OCM"},
            {"application": "MZ", "geographical_entity": "OCM"},
            {"application": "KPSA", "geographical_entity": "OEG"},
            {"application": "MZ", "geographical_entity": "OJO"},
        ]
        self.collection = FakeCollection(self.documents, self.metadatas)

    def test_bm25_construction_matches_legacy_order_tokenization_and_scores(self):
        legacy_build = load_legacy_function(
            "build_bm25_index",
            {"BM25Okapi": BM25Okapi, "collection": self.collection},
        )

        legacy_bm25, legacy_docs, legacy_metas = legacy_build(self.collection, len(self.documents))
        shared_bm25, shared_docs, shared_metas = rag_pipeline.build_bm25_index(
            self.collection,
            len(self.documents),
        )

        self.assertEqual(legacy_docs, shared_docs)
        self.assertEqual(legacy_metas, shared_metas)
        self.assertEqual([doc.lower().split() for doc in legacy_docs], [doc.lower().split() for doc in shared_docs])

        query_tokens = "alpha beta".lower().split()
        legacy_scores = legacy_bm25.get_scores(query_tokens)
        shared_scores = shared_bm25.get_scores(query_tokens)
        self.assertEqual(list(legacy_scores), list(shared_scores))

        legacy_order = sorted(range(len(legacy_scores)), key=lambda index: legacy_scores[index], reverse=True)
        shared_order = sorted(range(len(shared_scores)), key=lambda index: shared_scores[index], reverse=True)
        self.assertEqual(legacy_order, shared_order)

    def test_empty_corpus_matches_legacy_none_contract(self):
        empty_collection = FakeCollection([], [])
        legacy_build = load_legacy_function(
            "build_bm25_index",
            {"BM25Okapi": BM25Okapi, "collection": empty_collection},
        )

        self.assertEqual(legacy_build(empty_collection, 0), (None, None, None))
        self.assertEqual(rag_pipeline.build_bm25_index(empty_collection, 0), (None, None, None))

    def test_single_and_and_filter_outcomes_match_legacy_hybrid_search(self):
        legacy_hybrid_search = load_legacy_function("hybrid_search", {})
        bm25 = BM25Okapi([doc.lower().split() for doc in self.documents])
        filters = [
            {"application": "KPSA"},
            {"$and": [{"application": "KPSA"}, {"geographical_entity": "OCM"}]},
            {"$and": [{"application": "MZ"}, {"geographical_entity": "OEG"}]},
        ]

        for chroma_filter in filters:
            legacy_docs, legacy_metas, _, _, _ = legacy_hybrid_search(
                "alpha beta",
                self.collection,
                FakeEmbeddingModel(),
                bm25,
                self.documents,
                self.metadatas,
                chroma_filter=chroma_filter,
                top_k=10,
                min_results_before_relax=0,
            )
            scores = bm25.get_scores("alpha beta".lower().split())
            expected_docs, expected_metas = [], []
            for index in sorted(range(len(scores)), key=lambda item: scores[item], reverse=True):
                if len(expected_docs) >= 10 or scores[index] <= 0:
                    break
                if not rag_pipeline.metadata_matches_filter(self.metadatas[index], chroma_filter):
                    continue
                expected_docs.append(self.documents[index])
                expected_metas.append(self.metadatas[index])

            self.assertEqual(legacy_docs, expected_docs)
            self.assertEqual(legacy_metas, expected_metas)


if __name__ == "__main__":
    unittest.main()
