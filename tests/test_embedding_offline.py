"""Focused tests for the certified offline embedding loader."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chromadb
import rag_evaluator
import rag_pipeline


MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SNAPSHOT = Path.home() / ".cache" / "huggingface" / "hub" / (
    "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
) / "snapshots" / "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


class OfflineEmbeddingTests(unittest.TestCase):
    def test_resolves_certified_local_snapshot(self):
        self.assertEqual(rag_pipeline.resolve_embedding_snapshot(MODEL_NAME), SNAPSHOT)

    def test_missing_cache_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(rag_pipeline.EmbeddingModelLoadError):
                rag_pipeline.resolve_embedding_snapshot(MODEL_NAME, directory)

    def test_loader_enforces_offline_mode_and_dimension(self):
        class FakeModel:
            def get_sentence_embedding_dimension(self):
                return 384

        observed = {}

        def fake_loader(path, **kwargs):
            observed["path"] = path
            observed["kwargs"] = kwargs
            observed["hub_offline"] = os.environ.get("HF_HUB_OFFLINE")
            observed["transformers_offline"] = os.environ.get("TRANSFORMERS_OFFLINE")
            return FakeModel()

        with patch("sentence_transformers.SentenceTransformer", side_effect=fake_loader):
            model = rag_pipeline.load_embedding_model_offline(
                rag_pipeline.RAGConfig(embedding_model_name=MODEL_NAME)
            )
        self.assertIsInstance(model, FakeModel)
        self.assertEqual(observed["kwargs"], {"local_files_only": True})
        self.assertEqual(observed["hub_offline"], "1")
        self.assertEqual(observed["transformers_offline"], "1")
        self.assertTrue(str(observed["path"]).startswith(str(SNAPSHOT)))

    def test_app_and_evaluator_delegate_to_shared_loader(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        evaluator_source = Path("rag_evaluator.py").read_text(encoding="utf-8")
        self.assertIn("rag_pipeline.load_embedding_model_offline(", app_source)
        self.assertIn("rag_pipeline.load_embedding_model_offline(", evaluator_source)
        self.assertNotIn("SentenceTransformer(", app_source)
        self.assertNotIn("SentenceTransformer(", evaluator_source)

    def test_certified_model_name_and_active_collection_compatibility(self):
        config = rag_pipeline.RAGConfig()
        self.assertEqual(config.embedding_model_name, MODEL_NAME)
        collection = chromadb.PersistentClient(
            path=config.chroma_path
        ).get_collection(config.collection_name)
        self.assertEqual(collection.count(), 1072)

    def test_real_cached_model_loads_offline_with_384_dimensions(self):
        model = rag_pipeline.load_embedding_model_offline(rag_pipeline.RAGConfig())
        self.assertEqual(model.get_sentence_embedding_dimension(), 384)


if __name__ == "__main__":
    unittest.main()
