import unittest
from pathlib import Path

import rag_pipeline


ROOT = Path(__file__).resolve().parents[1]


class StructuredExhaustiveRoutingTests(unittest.TestCase):
    def test_production_route_is_opt_in_and_specific_document_only(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("structured_specific_direct_answer_enabled()", source)
        self.assertIn("fetch_structured_specific_chunks(collection, direct_document)", source)
        self.assertIn("selected_document_has_structured_metadata(collection, direct_document)", source)
        self.assertIn("extract_evidence_exhaustive_specific", source)
        self.assertIn('"retrieval_mode": "exhaustive_specific_structured" if structured_exhaustive_mode else "hybrid"', source)
        self.assertIn("selected_document_chunk_count", source)
        self.assertIn("evidence_selection_ms", source)

    def test_legacy_and_non_direct_routes_retain_hybrid_call(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("else:\n            filtered_chunks, filtered_metas", source)
        self.assertIn('answer_mode == "Direct answer"', source)
        self.assertIn('direct_scope == "specific_document"', source)

    def test_structured_result_preserves_selected_source_identity(self):
        chunks = [rag_pipeline.ChunkRecord(
            "Écrit par : = Omar EL HIMASS",
            {"source_file": "p2p.docx", "file_hash": "h", "block_type": "table_row", "chunk_ordinal": 2, "location": "Table 0"},
        )]
        result = rag_pipeline.extract_evidence_exhaustive_specific("Who wrote the P2P specification?", chunks)
        self.assertEqual(result.status, "EVIDENCE_FOUND")
        self.assertEqual(result.passages[0].source_file, "p2p.docx")

    def test_no_ollama_or_production_collection_symbols_added(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("ollama.chat", source[source.find("structured_exhaustive_mode"):source.find("structured_exhaustive_mode") + 5000])
        self.assertIn("collection.get(where={key: value}", source)


if __name__ == "__main__":
    unittest.main()
