"""Contract tests for the declarative shared RAG pipeline API."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

import rag_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RAGPipelineContractTests(unittest.TestCase):
    """Verify that Sub-phase 1.1 remains declarative and dependency-light."""

    def test_public_api_is_versioned_and_explicit(self) -> None:
        self.assertEqual(rag_pipeline.__version__, "1.0")
        self.assertEqual(rag_pipeline.API_VERSION, "1.0")
        self.assertIn("PipelineTrace", rag_pipeline.__all__)
        self.assertIn("RAGConfig", rag_pipeline.__all__)
        self.assertIn("normalize_chroma_filter", rag_pipeline.__all__)
        self.assertEqual(len(rag_pipeline.__all__), len(set(rag_pipeline.__all__)) )

    def test_configuration_defaults_match_current_production_values(self) -> None:
        config = rag_pipeline.RAGConfig()
        self.assertEqual(config.storage_dir, "doc_storage_v2")
        self.assertEqual(config.chroma_path, "chroma_db_local_v2")
        self.assertEqual(config.collection_name, "documents")
        self.assertEqual(config.embedding_model_name, "paraphrase-multilingual-MiniLM-L12-v2")
        self.assertEqual(config.llm_model_name, "qwen3:8b")
        self.assertEqual(config.vector_candidate_count, 10)
        self.assertEqual(config.bm25_candidate_count, 10)
        self.assertEqual(config.rrf_k, 60)
        self.assertEqual(config.default_top_k, 5)
        self.assertEqual(config.production_top_k, 15)
        self.assertEqual(config.min_results_before_relax, 3)
        self.assertEqual(config.rewrite_temperature, 0.0)
        self.assertEqual(config.generation_temperature, 0.2)

    def test_contract_models_are_immutable(self) -> None:
        config = rag_pipeline.RAGConfig()
        trace = rag_pipeline.PipelineTrace(query="question")
        with self.assertRaises(FrozenInstanceError):
            config.rrf_k = 61  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            trace.query = "different question"  # type: ignore[misc]

    def test_trace_schema_is_constructible_without_runtime_dependencies(self) -> None:
        source = rag_pipeline.PromptSource(
            source_id=1,
            file_name="document.pdf",
            location="Page 1",
            text="content",
            path="C:/documents/document.pdf",
        )
        trace = rag_pipeline.PipelineTrace(
            query="question",
            rewritten_query="question",
            retrieval=rag_pipeline.RetrievalResult(
                filtered_chunks=(rag_pipeline.ChunkRecord("content"),)
            ),
            prompt=rag_pipeline.PromptResult(prompt="prompt", sources=(source,)),
            citations=rag_pipeline.CitationResult(cited_source_ids=(1,), display_sources=(source,)),
        )
        self.assertEqual(trace.schema_version, "1.0")
        self.assertEqual(trace.prompt.sources[0].source_id, 1)

    def test_module_has_no_forbidden_runtime_imports(self) -> None:
        module_path = PROJECT_ROOT / "rag_pipeline.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])

        forbidden = {"app", "chromadb", "ollama", "streamlit", "sentence_transformers"}
        self.assertFalse(imported_roots & forbidden)

    def test_module_declares_only_approved_runtime_functions(self) -> None:
        module_path = PROJECT_ROOT / "rag_pipeline.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        module_functions = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
        ]
        self.assertEqual(
            module_functions,
            [
                "build_bm25_index",
                "normalize_chroma_filter",
                "metadata_matches_filter",
                "hybrid_search",
                "build_source_list",
                "build_context",
                "build_recent_chat_history",
                "build_no_match_message",
                "build_clarification_message",
                "build_production_prompt",
                "rewrite_query",
                "stream_generate",
                "parse_cited_source_ids",
                "detect_no_coverage",
                "select_display_sources",
                "deduplicate_sources_by_path",
                "extract_evidence",
                "build_extractive_answer",
            ],
        )

    def test_normalize_chroma_filter_preserves_valid_forms_and_rejects_invalid_ones(self) -> None:
        self.assertIsNone(rag_pipeline.normalize_chroma_filter(None))
        self.assertIsNone(rag_pipeline.normalize_chroma_filter({}))
        self.assertEqual(
            rag_pipeline.normalize_chroma_filter({"application": "KPSA"}),
            {"application": "KPSA"},
        )
        self.assertEqual(
            rag_pipeline.normalize_chroma_filter(
                {"application": "KPSA", "geographical_entity": "OCM"}
            ),
            {"$and": [{"application": "KPSA"}, {"geographical_entity": "OCM"}]},
        )
        self.assertEqual(
            rag_pipeline.normalize_chroma_filter({"$and": [{"application": "KPSA"}]}),
            {"$and": [{"application": "KPSA"}]},
        )
        with self.assertRaisesRegex(TypeError, "must be mappings"):
            rag_pipeline.normalize_chroma_filter("KPSA")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "one operator"):
            rag_pipeline.normalize_chroma_filter({"$and": [{"application": "KPSA"}], "x": "y"})


if __name__ == "__main__":
    unittest.main()
