"""Deterministic integration checks for app.py's shared-runtime delegation."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import re
import unittest

from rank_bm25 import BM25Okapi

import rag_pipeline
from baseline_app_reference import BASELINE_COMMIT, source as baseline_source
from baseline_app_reference import top_level_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def current_app_source() -> str:
    return (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")


def current_top_level_function(name: str, namespace: dict[str, object]) -> object:
    tree = ast.parse(current_app_source())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function = copy.deepcopy(function)
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace[name]


def filter_key(value: object) -> str:
    return json.dumps(value, sort_keys=True)


class FakeVector:
    def tolist(self):
        return [0.25, 0.75]


class FakeEmbeddingModel:
    def encode(self, _query):
        return FakeVector()


class RecordingCollection:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return copy.deepcopy(self.responses[filter_key(kwargs["where"])])


class DeterministicOllama:
    def chat(self, **_kwargs):
        return {"message": {"content": "What does OCM mean?"}}


class AppPipelineIntegrationTests(unittest.TestCase):
    """Certify the app delegates to the parity-certified runtime unchanged."""

    def test_current_app_delegates_only_to_shared_runtime(self):
        source = current_app_source()
        self.assertIn("import rag_pipeline", source)
        self.assertIn("return rag_pipeline.build_bm25_index(_collection, _count)", source)
        self.assertIn("return rag_pipeline.hybrid_search(", source)
        self.assertIn("rag_pipeline.build_source_list(", source)
        self.assertIn("rag_pipeline.build_production_prompt(", source)
        self.assertIn("rag_pipeline.select_display_sources(", source)
        self.assertIn("rag_pipeline.deduplicate_sources_by_path(", source)

    def test_rewritten_query_and_language_detection_match_immutable_baseline(self):
        baseline_detect = top_level_function("detect_query_language", {"re": re})
        integrated_detect = current_top_level_function("detect_query_language", {"re": re})
        for query, fallback in (("What does OCM mean?", "French"), ("Bonjour", "English")):
            self.assertEqual(baseline_detect(query, fallback), integrated_detect(query, fallback))

        baseline_rewrite = top_level_function("contextualize_query", {"ollama": DeterministicOllama()})
        integrated_rewrite = current_top_level_function(
            "contextualize_query", {"ollama": DeterministicOllama()}
        )
        history = [{"role": "assistant", "content": "OCM is change management."}]
        self.assertEqual(
            baseline_rewrite("What does it mean?", history, "qwen3:8b"),
            integrated_rewrite("What does it mean?", history, "qwen3:8b"),
        )

    def test_retrieval_output_and_app_delegate_match_immutable_baseline(self):
        documents = ["alpha beta", "beta gamma", "alpha delta"]
        metadatas = [
            {"application": "KPSA", "geographical_entity": "OCM"},
            {"application": "MZ", "geographical_entity": "OCM"},
            {"application": "KPSA", "geographical_entity": "OEG"},
        ]
        active_filter = {"application": "KPSA"}
        responses = {
            filter_key(active_filter): {
                "ids": [["id-1"]],
                "documents": [["alpha beta"]],
                "metadatas": [[metadatas[0]]],
                "distances": [[0.1]],
            },
            filter_key(None): {
                "ids": [["id-1", "id-3", "id-2"]],
                "documents": [["alpha beta", "alpha delta", "beta gamma"]],
                "metadatas": [[metadatas[0], metadatas[2], metadatas[1]]],
                "distances": [[0.1, 0.2, 0.3]],
            },
        }
        bm25 = BM25Okapi([document.lower().split() for document in documents])
        baseline = top_level_function("hybrid_search", {})
        integrated = current_top_level_function("hybrid_search", {"rag_pipeline": rag_pipeline})

        baseline_collection = RecordingCollection(responses)
        integrated_collection = RecordingCollection(responses)
        baseline_result = baseline(
            "alpha", baseline_collection, FakeEmbeddingModel(), bm25, documents, metadatas,
            chroma_filter=active_filter, top_k=2, min_results_before_relax=3,
        )
        integrated_result = integrated(
            "alpha", integrated_collection, FakeEmbeddingModel(), bm25, documents, metadatas,
            chroma_filter=active_filter, top_k=2, min_results_before_relax=3,
        )

        self.assertEqual(integrated_result, baseline_result)
        self.assertEqual(integrated_collection.calls, baseline_collection.calls)

        trace = rag_pipeline.hybrid_search(
            "alpha", RecordingCollection(responses), FakeEmbeddingModel(), bm25, documents, metadatas,
            chroma_filter=active_filter, top_k=2, min_results_before_relax=3,
        )
        self.assertEqual(trace.as_legacy_tuple(), baseline_result)
        self.assertEqual(
            [candidate.chunk.chunk_id for candidate in trace.filtered.vector_candidates],
            ["id-1"],
        )
        self.assertTrue(trace.fallback_used)
        self.assertEqual([chunk.text for chunk in trace.fallback_chunks], integrated_result[2])

    def test_prompt_citations_refusal_and_final_display_match_baseline(self):
        baseline_tree = ast.parse(baseline_source())
        baseline_builder_node = copy.deepcopy(
            next(
                node
                for node in ast.walk(baseline_tree)
                if isinstance(node, ast.FunctionDef) and node.name == "build_source_list"
            )
        )
        baseline_builder_module = ast.Module(body=[baseline_builder_node], type_ignores=[])
        ast.fix_missing_locations(baseline_builder_module)
        baseline_builder_namespace = {"os": os, "STORAGE_DIR": "doc_storage_v2"}
        exec(
            compile(baseline_builder_module, f"{BASELINE_COMMIT}:app.py", "exec"),
            baseline_builder_namespace,
        )
        baseline_builder = baseline_builder_namespace["build_source_list"]
        chunks = ["Premier extrait", "Second excerpt"]
        metas = [
            {"source_file": "OCM.docx", "location": "Page 1"},
            {"source_file": "MZ.docx", "location": "Page 2"},
        ]
        baseline_sources = baseline_builder(chunks, metas)
        integrated_sources = rag_pipeline.build_source_list(chunks, metas, "doc_storage_v2")
        self.assertEqual(
            baseline_sources,
            [
                {
                    "id": source.source_id, "file": source.file_name, "loc": source.location,
                    "text": source.text, "path": source.path, "relaxed": source.relaxed,
                }
                for source in integrated_sources
            ],
        )

        baseline_context = "\n---\n".join(
            f"[SOURCE {source['id']}]\n{source['text']}" for source in baseline_sources
        )
        namespace = {
            "filter_ent": "Tous", "filter_application": "Tous", "recent_chat_history": "",
            "context_str": baseline_context, "relaxed_note": "", "user_query": "What is OCM?",
            "current_lang": "English",
        }
        tree = ast.parse(baseline_source())
        assignment = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "prompt_instructions" for target in node.targets)
        )
        module = ast.Module(body=[copy.deepcopy(assignment)], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, f"{BASELINE_COMMIT}:app.py", "exec"), namespace)
        baseline_prompt = namespace["prompt_instructions"]
        integrated_prompt = rag_pipeline.build_production_prompt(
            user_query="What is OCM?", filter_ent="Tous", filter_application="Tous", history=(),
            sources=integrated_sources, current_lang="English", was_relaxed=False,
        )
        self.assertEqual(integrated_prompt.prompt, baseline_prompt)

        response = "OCM is change management. [SOURCE 1] [SOURCE 1]"
        baseline_cited_ids = tuple(list(set(int(number) for number in __import__("re").findall(r"\[SOURCE (\d+)\]", response))))
        integrated_citations = rag_pipeline.select_display_sources(response, integrated_sources)
        self.assertEqual(integrated_citations.cited_source_ids, baseline_cited_ids)
        self.assertEqual([source.source_id for source in integrated_citations.display_sources], [1])
        self.assertFalse(integrated_citations.no_coverage_detected)

        refusal = "I cannot find this in the document context. [SOURCE 1]"
        refusal_result = rag_pipeline.select_display_sources(refusal, integrated_sources)
        self.assertTrue(refusal_result.no_coverage_detected)
        self.assertEqual(refusal_result.display_sources, ())


if __name__ == "__main__":
    unittest.main()
