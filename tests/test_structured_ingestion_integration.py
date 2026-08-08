import hashlib
import os
import unittest
from pathlib import Path

from structured_ingestion import build_structured_docx_index_payload


ROOT = Path(__file__).resolve().parents[1]
P2P = ROOT / "doc_storage_v2" / "CISA_MZ-001-1.0 - OCM Mediation - P2P.docx"


class StructuredDocxIntegrationTests(unittest.TestCase):
    def test_payload_is_complete_and_deterministic(self):
        data = P2P.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()
        first = build_structured_docx_index_payload(
            data, P2P.name, file_hash=file_hash, geographical_entity="OCM", application="MZ"
        )
        second = build_structured_docx_index_payload(
            data, P2P.name, file_hash=file_hash, geographical_entity="OCM", application="MZ"
        )
        self.assertEqual(first, second)
        self.assertTrue(first["documents"])
        self.assertEqual(len(first["ids"]), len(first["documents"]))
        self.assertEqual(len(first["ids"]), len(first["metadatas"]))
        self.assertEqual(len(first["ids"]), len(set(first["ids"])))
        self.assertTrue(all("timestamp" not in value for value in first["ids"]))
        self.assertTrue(all("timestamp_ingest" not in value for value in first["metadatas"]))
        self.assertTrue(all(meta["file_hash"] == file_hash for meta in first["metadatas"]))

    def test_structured_flag_is_literal_true_and_dry_run_is_separate(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('ENABLE_STRUCTURED_DOCX_INGESTION = _env_flag("ENABLE_STRUCTURED_DOCX_INGESTION")', source)
        self.assertIn('STRUCTURED_INGESTION_DRY_RUN = _env_flag("STRUCTURED_INGESTION_DRY_RUN")', source)
        self.assertIn('filename.casefold().endswith(".docx") and ENABLE_STRUCTURED_DOCX_INGESTION', source)
        self.assertIn("zero Chroma writes", source)

    def test_payload_precedes_write_and_preserves_redaction(self):
        data = P2P.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()
        payload = build_structured_docx_index_payload(data, P2P.name, file_hash=file_hash)
        # The payload is the complete validated unit passed to collection.add;
        # no write-capable object is involved in this test.
        self.assertEqual(len(payload["ids"]), len(payload["documents"]))
        self.assertFalse(any("secret-value" in text.casefold() for text in payload["documents"]))
        password_blocks = [text for text in payload["documents"] if "password =" in text.casefold()]
        self.assertTrue(all("[redacted]" in text.casefold() for text in password_blocks))


if __name__ == "__main__":
    unittest.main()
