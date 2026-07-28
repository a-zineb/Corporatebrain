"""Deterministic numerical parity checks for the current app.py hybrid search."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import unittest

from rank_bm25 import BM25Okapi

import rag_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def filter_key(value):
    """Make nested filter dictionaries usable as fake-vector-response keys."""

    return json.dumps(value, sort_keys=True)


def load_traced_legacy_hybrid_search(trace):
    """Instrument a local AST copy of app.py's function without importing it.

    The injected statements only observe the legacy function's own variables;
    they do not replace, reorder, or otherwise alter its retrieval operations.
    """

    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(app_source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "hybrid_search")
    source = ast.get_source_segment(app_source, function)
    source = source.replace(
        "        doc_to_meta = {}\n",
        "        doc_to_meta = {}\n        bm25_trace = []\n",
        1,
    )
    source = source.replace(
        "                doc_to_meta[doc_text] = meta\n                bm25_count += 1\n",
        "                doc_to_meta[doc_text] = meta\n"
        "                bm25_trace.append((doc_text, meta, bm25_scores[idx], bm25_count))\n"
        "                bm25_count += 1\n",
        1,
    )
    source = source.replace(
        "        return final_docs, final_metas\n",
        "        __legacy_trace__.append({\n"
        "            'filter': chroma_filter,\n"
        "            'vector_result': vec_results,\n"
        "            'bm25_candidates': list(bm25_trace),\n"
        "            'rrf_scores': list(sorted_docs),\n"
        "            'selected_docs': list(final_docs),\n"
        "            'selected_metas': list(final_metas),\n"
        "        })\n"
        "        return final_docs, final_metas\n",
        1,
    )
    namespace = {"__legacy_trace__": trace}
    exec(compile(source, "app.py", "exec"), namespace)
    return namespace["hybrid_search"]


class FakeVector:
    """NumPy-like embedding result used by the active production function."""

    def tolist(self):
        return [0.25, 0.75]


class FakeEmbeddingModel:
    def encode(self, _query):
        return FakeVector()


class RecordingCollection:
    """Read-only Chroma double that records exact vector query arguments."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return copy.deepcopy(self.responses[filter_key(kwargs["where"])])


def vector_response(ids, documents, metadatas, distances):
    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


class HybridSearchParityTests(unittest.TestCase):
    """Compare each shared retrieval pass with observed legacy internals."""

    def assert_pass_parity(self, legacy_trace, shared_pass):
        legacy_vector = legacy_trace["vector_result"]
        self.assertEqual(shared_pass.vector_query.query_embeddings, ((0.25, 0.75),))
        self.assertEqual(shared_pass.vector_query.n_results, 10)
        self.assertIsNone(shared_pass.vector_query.include)
        self.assertEqual(
            [candidate.chunk.chunk_id for candidate in shared_pass.vector_candidates],
            legacy_vector["ids"][0],
        )
        self.assertEqual(
            [candidate.chunk.text for candidate in shared_pass.vector_candidates],
            legacy_vector["documents"][0],
        )
        self.assertEqual(
            [candidate.chunk.metadata for candidate in shared_pass.vector_candidates],
            legacy_vector["metadatas"][0],
        )
        self.assertEqual(
            [candidate.distance for candidate in shared_pass.vector_candidates],
            legacy_vector["distances"][0],
        )

        self.assertEqual(
            [
                (candidate.chunk.text, candidate.chunk.metadata, candidate.score, candidate.rank)
                for candidate in shared_pass.bm25_candidates
            ],
            legacy_trace["bm25_candidates"],
        )
        self.assertEqual(
            [(score.chunk.text, score.score) for score in shared_pass.rrf_scores],
            legacy_trace["rrf_scores"],
        )
        self.assertEqual(
            [chunk.text for chunk in shared_pass.selected_chunks],
            legacy_trace["selected_docs"],
        )
        self.assertEqual(
            [chunk.metadata for chunk in shared_pass.selected_chunks],
            legacy_trace["selected_metas"],
        )

    def run_both(self, *, query, documents, metadatas, responses, chroma_filter=None, top_k=5, threshold=3):
        bm25 = BM25Okapi([document.lower().split() for document in documents]) if documents else None
        legacy_collection = RecordingCollection(responses)
        shared_collection = RecordingCollection(responses)
        legacy_trace = []
        legacy_hybrid_search = load_traced_legacy_hybrid_search(legacy_trace)

        legacy = legacy_hybrid_search(
            query,
            legacy_collection,
            FakeEmbeddingModel(),
            bm25,
            documents or None,
            metadatas or None,
            chroma_filter=chroma_filter,
            top_k=top_k,
            min_results_before_relax=threshold,
        )
        shared = rag_pipeline.hybrid_search(
            query,
            shared_collection,
            FakeEmbeddingModel(),
            bm25,
            documents or None,
            metadatas or None,
            chroma_filter=chroma_filter,
            top_k=top_k,
            min_results_before_relax=threshold,
        )

        self.assertEqual(legacy_collection.calls, shared_collection.calls)
        self.assertEqual(shared.as_legacy_tuple(), legacy)
        self.assertEqual(len(legacy_trace), 2 if shared.fallback_used else 1)
        self.assert_pass_parity(legacy_trace[0], shared.filtered)
        if shared.fallback_used:
            self.assertIsNotNone(shared.fallback)
            self.assert_pass_parity(legacy_trace[1], shared.fallback)
        return shared, legacy_collection.calls, legacy_trace

    def test_vector_arguments_bm25_candidates_rrf_scores_and_top_k_match_legacy(self):
        documents = ["alpha alpha", "alpha beta", "beta gamma", "delta only", "epsilon only"]
        metadatas = [
            {"application": "KPSA", "geographical_entity": "OCM"},
            {"application": "MZ", "geographical_entity": "OCM"},
            {"application": "KPSA", "geographical_entity": "OEG"},
            {"application": "MZ", "geographical_entity": "OJO"},
            {"application": "MZ", "geographical_entity": "OCI"},
        ]
        responses = {
            filter_key(None): vector_response(
                ["vec-1", "vec-2"],
                ["vector only", "alpha beta"],
                [metadatas[4], metadatas[1]],
                [0.12, 0.34],
            )
        }

        shared, calls, _ = self.run_both(
            query="alpha beta",
            documents=documents,
            metadatas=metadatas,
            responses=responses,
        )

        self.assertEqual(calls, [{"query_embeddings": [[0.25, 0.75]], "n_results": 10, "where": None}])
        self.assertGreater(len(shared.filtered.bm25_candidates), 0)

    def test_empty_results_preserve_empty_output_and_no_fallback_without_filter(self):
        responses = {filter_key(None): vector_response([], [], [], [])}
        shared, calls, _ = self.run_both(
            query="absent",
            documents=[],
            metadatas=[],
            responses=responses,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(shared.filtered.selected_chunks, ())
        self.assertFalse(shared.fallback_used)
        self.assertEqual(shared.fallback_chunks, ())

    def test_duplicate_texts_match_legacy_duplicate_removal_and_metadata_preservation(self):
        documents = ["duplicate", "duplicate", "other one", "other two", "other three"]
        metadatas = [
            {"application": "KPSA", "geographical_entity": "OCM", "version": "first"},
            {"application": "MZ", "geographical_entity": "OCM", "version": "second"},
            {"application": "MZ", "geographical_entity": "OJO"},
            {"application": "MZ", "geographical_entity": "OCI"},
            {"application": "KPSA", "geographical_entity": "OEG"},
        ]
        responses = {
            filter_key(None): vector_response(
                ["first", "second"],
                documents[:2],
                metadatas[:2],
                [0.1, 0.2],
            )
        }

        shared, _, _ = self.run_both(
            query="duplicate",
            documents=documents,
            metadatas=metadatas,
            responses=responses,
        )

        self.assertEqual([chunk.text for chunk in shared.filtered.selected_chunks], ["duplicate"])
        self.assertEqual(shared.filtered.selected_chunks[0].metadata["version"], "second")

    def test_identical_rrf_scores_keep_legacy_stable_insertion_order(self):
        documents = ["bm25 only", "other one", "other two", "other three", "other four"]
        metadatas = [
            {"application": "KPSA", "geographical_entity": "OCM"},
            {"application": "MZ", "geographical_entity": "OCM"},
            {"application": "MZ", "geographical_entity": "OJO"},
            {"application": "MZ", "geographical_entity": "OCI"},
            {"application": "KPSA", "geographical_entity": "OEG"},
        ]
        responses = {
            filter_key(None): vector_response(
                ["vec"],
                ["vector only"],
                [metadatas[1]],
                [0.1],
            )
        }

        shared, _, _ = self.run_both(
            query="bm25",
            documents=documents,
            metadatas=metadatas,
            responses=responses,
        )

        self.assertEqual([item.chunk.text for item in shared.filtered.rrf_scores], ["vector only", "bm25 only"])
        self.assertEqual(shared.filtered.rrf_scores[0].score, shared.filtered.rrf_scores[1].score)

    def test_single_and_and_filters_match_legacy_fallback_trigger_order_and_deduplication(self):
        documents = [
            "alpha ocm",
            "alpha oeg",
            "alpha mz",
            "other one",
            "other two",
            "other three",
            "other four",
            "other five",
        ]
        metadatas = [
            {"application": "KPSA", "geographical_entity": "OCM"},
            {"application": "KPSA", "geographical_entity": "OEG"},
            {"application": "MZ", "geographical_entity": "OCM"},
            {"application": "MZ", "geographical_entity": "OJO"},
            {"application": "MZ", "geographical_entity": "OCI"},
            {"application": "KPSA", "geographical_entity": "OCI"},
            {"application": "MZ", "geographical_entity": "OEG"},
            {"application": "MZ", "geographical_entity": "OJO"},
        ]
        single_filter = {"application": "KPSA"}
        and_filter = {"$and": [{"application": "KPSA"}, {"geographical_entity": "OCM"}]}

        for chroma_filter in (single_filter, and_filter):
            responses = {
                filter_key(chroma_filter): vector_response(
                    ["filtered"],
                    ["alpha ocm"],
                    [metadatas[0]],
                    [0.1],
                ),
                filter_key(None): vector_response(
                    ["filtered", "outside"],
                    ["alpha ocm", "outside source"],
                    [metadatas[0], metadatas[2]],
                    [0.1, 0.2],
                ),
            }
            shared, calls, legacy_trace = self.run_both(
                query="alpha",
                documents=documents,
                metadatas=metadatas,
                responses=responses,
                chroma_filter=chroma_filter,
                top_k=5,
                threshold=3,
            )

            self.assertTrue(shared.fallback_used)
            self.assertEqual([call["where"] for call in calls], [chroma_filter, None])
            self.assertEqual(
                [chunk.text for chunk in shared.fallback_chunks],
                [
                    text
                    for text in legacy_trace[1]["selected_docs"]
                    if text not in legacy_trace[0]["selected_docs"]
                ],
            )


if __name__ == "__main__":
    unittest.main()
