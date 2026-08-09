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

    def test_entity_glossary_does_not_beat_requested_field(self):
        chunks = [
            ChunkRecord(
                "Tango = Plateforme qui permet aux utilisateurs d'accéder à des services bancaires.",
                {"source_file": "tango.docx", "location": "Glossary", "block_type": "paragraph"},
            ),
            ChunkRecord(
                "Reviewer = Nawfal ENNAJI",
                {"source_file": "tango.docx", "location": "Table 0", "block_type": "key_value"},
            ),
            ChunkRecord(
                "Collection Directory = /opt/cft/v3.0.1/Transfer_CFT/runtime/pub/DAILY/DONE/",
                {"source_file": "tango.docx", "location": "Table 1", "block_type": "key_value"},
            ),
        ]
        author = extract_evidence_generic_structured("Who reviewed Tango?", chunks)
        directory = extract_evidence_generic_structured("What is the Tango collection directory?", chunks)
        meaning = extract_evidence_generic_structured("What is Tango?", chunks)
        self.assertEqual(author.status, "EVIDENCE_FOUND")
        self.assertIn("Nawfal ENNAJI", author.passages[0].text)
        self.assertEqual(directory.status, "EVIDENCE_FOUND")
        self.assertIn("/opt/cft/v3.0.1", directory.passages[0].text)
        self.assertEqual(meaning.status, "EVIDENCE_FOUND")
        self.assertIn("Tango =", meaning.passages[0].text)

    def test_multilingual_author_and_reviewer_relations_are_distinct(self):
        chunks = [
            ChunkRecord("Écrit par = Alice Martin", {"source_file": "x.docx", "block_type": "table_row"}),
            ChunkRecord("Revue par = Bob Smith", {"source_file": "x.docx", "block_type": "table_row"}),
        ]
        author = extract_evidence_generic_structured("Who wrote the specification?", chunks)
        reviewer = extract_evidence_generic_structured("Who reviewed the specification?", chunks)
        self.assertEqual(author.status, "EVIDENCE_FOUND")
        self.assertIn("Alice Martin", author.passages[0].text)
        self.assertEqual(reviewer.status, "EVIDENCE_FOUND")
        self.assertIn("Bob Smith", reviewer.passages[0].text)


if __name__ == "__main__":
    unittest.main()
