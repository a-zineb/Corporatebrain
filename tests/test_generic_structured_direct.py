import unittest
from pathlib import Path

from rag_pipeline import ChunkRecord, extract_evidence_generic_structured


class GenericStructuredDirectTests(unittest.TestCase):
    def test_feature_flag_is_explicit_and_routing_is_opt_in(self):
        source = Path("app.py").read_text(encoding="utf-8")
        self.assertIn('ENABLE_GENERIC_STRUCTURED_DIRECT = _env_flag("ENABLE_GENERIC_STRUCTURED_DIRECT")', source)
        self.assertIn("def generic_structured_direct_answer_enabled():", source)
        self.assertIn("rag_pipeline.extract_evidence_generic_structured", source)

    def setUp(self):
        self.chunks = [
            ChunkRecord(
                "Reviewer = Alice Smith",
                {"source_file": "synthetic.docx", "location": "Table 0 row 0", "block_type": "key_value"},
            ),
            ChunkRecord(
                "Polling Interval = 15 minutes",
                {"source_file": "synthetic.docx", "location": "Table 0 row 1", "block_type": "key_value"},
            ),
            ChunkRecord(
                "Retry Count = 5",
                {"source_file": "synthetic.docx", "location": "Table 0 row 2", "block_type": "key_value"},
            ),
            ChunkRecord(
                "Business Owner = Bob Martin",
                {"source_file": "synthetic.docx", "location": "Table 0 row 3", "block_type": "key_value"},
            ),
            ChunkRecord(
                "System name = FRAUD_ENGINE | Transfer Mode = Passive | Endpoint = /fraud/input",
                {"source_file": "synthetic.docx", "location": "Table 1 column 0", "block_type": "column_record"},
            ),
        ]

    def test_unknown_fields_and_entities_are_answerable(self):
        expected = {
            "Who reviewed the document?": "Alice Smith",
            "What is the polling interval?": "15 minutes",
            "What is the retry count?": "5",
            "Who is the business owner?": "Bob Martin",
            "What is the transfer mode for FRAUD_ENGINE?": "Passive",
            "What is the FRAUD_ENGINE endpoint?": "/fraud/input",
        }
        for query, value in expected.items():
            result = extract_evidence_generic_structured(query, self.chunks)
            self.assertEqual(result.status, "EVIDENCE_FOUND", query)
            self.assertIn(value, result.passages[0].text, query)

    def test_missing_field_is_no_explicit_evidence(self):
        result = extract_evidence_generic_structured("What is the deployment region?", self.chunks)
        self.assertEqual(result.status, "NO_EXPLICIT_EVIDENCE")

    def test_sensitive_field_is_not_admitted(self):
        chunks = self.chunks + [ChunkRecord("Password = [REDACTED]", {"source_file": "synthetic.docx", "block_type": "key_value"})]
        result = extract_evidence_generic_structured("What is the FRAUD_ENGINE password?", chunks)
        self.assertEqual(result.status, "NO_EXPLICIT_EVIDENCE")

    def test_repeated_results_are_deterministic(self):
        first = extract_evidence_generic_structured("What is the transfer mode for FRAUD_ENGINE?", self.chunks).to_json()
        second = extract_evidence_generic_structured("What is the transfer mode for FRAUD_ENGINE?", self.chunks).to_json()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
