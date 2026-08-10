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

    def test_section_qualified_repeated_fields(self):
        chunks = [
            ChunkRecord("Collection Frequency = 5 minutes", {"source_file": "x.docx", "section": "Primary Feed", "block_type": "key_value"}),
            ChunkRecord("Collection Frequency = 15 minutes", {"source_file": "x.docx", "section": "Validation Feed", "block_type": "key_value"}),
        ]
        primary = extract_evidence_generic_structured("How often are Primary Feed files collected?", chunks)
        validation = extract_evidence_generic_structured("How often are Validation Feed files collected?", chunks)
        self.assertIn("5 minutes", primary.passages[0].text)
        self.assertIn("15 minutes", validation.passages[0].text)

    def test_matrix_entity_collisions_are_hard_filtered(self):
        chunks = [
            ChunkRecord("System name = DATAHUB | Protocol = SFTP | FileDirectory = /datahub/input", {"source_file": "x.docx", "block_type": "column_record"}),
            ChunkRecord("System name = WAREHOUSE | Protocol = SFTP | FileDirectory = /warehouse/input", {"source_file": "x.docx", "block_type": "column_record"}),
            ChunkRecord("System name = REPORTING | Protocol = FTP | FileDirectory = /reporting/input", {"source_file": "x.docx", "block_type": "column_record"}),
        ]
        self.assertIn("/datahub/input", extract_evidence_generic_structured("What is the DATAHUB output directory?", chunks).passages[0].text)
        self.assertIn("/warehouse/input", extract_evidence_generic_structured("What is the WAREHOUSE output directory?", chunks).passages[0].text)
        self.assertIn("FTP", extract_evidence_generic_structured("What protocol is used for REPORTING?", chunks).passages[0].text)

    def test_explicit_factual_prose_is_supported(self):
        chunks = [
            ChunkRecord("Duplicate files are detected using CRC over the complete file.", {"source_file": "x.docx", "block_type": "paragraph"}),
            ChunkRecord("The maximum cache age is 30 days.", {"source_file": "x.docx", "block_type": "paragraph"}),
            ChunkRecord("Archived files are compressed using GZIP.", {"source_file": "x.docx", "block_type": "paragraph"}),
            ChunkRecord("BI has priority over DWH.", {"source_file": "x.docx", "block_type": "paragraph"}),
        ]
        self.assertEqual(extract_evidence_generic_structured("How are duplicate files detected?", chunks).status, "EVIDENCE_FOUND")
        self.assertIn("30 days", extract_evidence_generic_structured("What is the maximum cache age?", chunks).passages[0].text)
        self.assertIn("GZIP", extract_evidence_generic_structured("What compression format is used?", chunks).passages[0].text)
        self.assertIn("BI has priority", extract_evidence_generic_structured("Which system has priority?", chunks).passages[0].text)

    def test_ambiguous_repeated_field_fails_closed(self):
        chunks = [
            ChunkRecord("Collection Frequency = 5 minutes", {"source_file": "x.docx", "block_type": "key_value"}),
            ChunkRecord("Collection Frequency = 15 minutes", {"source_file": "x.docx", "block_type": "key_value"}),
        ]
        self.assertEqual(extract_evidence_generic_structured("How often are files collected?", chunks).status, "NO_EXPLICIT_EVIDENCE")

    def test_collection_verbs_select_collection_over_distribution_frequency(self):
        chunks = [
            ChunkRecord("Collection Frequency = Every 5 minutes", {"source_file": "x.docx", "block_type": "key_value"}),
            ChunkRecord("Distribution Frequency = Once per day", {"source_file": "x.docx", "block_type": "key_value"}),
        ]
        result = extract_evidence_generic_structured("How often are files collected?", chunks)
        self.assertEqual(result.status, "EVIDENCE_FOUND")
        self.assertIn("Every 5 minutes", result.passages[0].text)

    def test_requested_port_rejects_frequency_and_selects_numeric_port(self):
        chunks = [
            ChunkRecord("Collection Frequency = Every 5 minutes", {"source_file": "x.docx", "block_type": "key_value"}),
            ChunkRecord("Connection Protocol = SFTP (port: 2222)", {"source_file": "x.docx", "block_type": "key_value"}),
        ]
        port = extract_evidence_generic_structured("What port is used for collection?", chunks)
        frequency = extract_evidence_generic_structured("How often are files collected?", chunks)
        self.assertEqual(port.status, "EVIDENCE_FOUND")
        self.assertIn("2222", port.passages[0].text)
        self.assertEqual(frequency.status, "EVIDENCE_FOUND")
        self.assertIn("Every 5 minutes", frequency.passages[0].text)

    def test_processing_status_fields_use_exact_generic_relations(self):
        chunks = [
            ChunkRecord("Enrichissement = N/A", {"source_file": "synthetic.docx", "block_type": "key_value"}),
            ChunkRecord("Normalisation = N/A", {"source_file": "synthetic.docx", "block_type": "key_value"}),
            ChunkRecord("Correlation = Enabled", {"source_file": "synthetic.docx", "block_type": "key_value"}),
            ChunkRecord("Database Lookups = Disabled", {"source_file": "synthetic.docx", "block_type": "key_value"}),
        ]
        expected = {
            "Is enrichment performed?": "Enrichissement = N/A",
            "Is normalization performed?": "Normalisation = N/A",
            "Is correlation performed?": "Correlation = Enabled",
            "Are database lookups performed?": "Database Lookups = Disabled",
            "L'enrichissement est-il effectué ?": "Enrichissement = N/A",
            "La normalisation est-elle effectuée ?": "Normalisation = N/A",
        }
        for query, value in expected.items():
            result = extract_evidence_generic_structured(query, chunks)
            self.assertEqual(result.status, "EVIDENCE_FOUND", query)
            self.assertEqual(result.passages[0].text, value)

    def test_file_duplicate_and_udr_relations_are_distinct(self):
        chunks = [
            ChunkRecord("Duplicate files are detected using CRC over the complete collected file.", {"source_file": "x.docx", "block_type": "paragraph"}),
            ChunkRecord("Duplicate UDR Check = N/A", {"source_file": "x.docx", "block_type": "key_value"}),
        ]
        file_result = extract_evidence_generic_structured("How are duplicate files detected?", chunks)
        udr_result = extract_evidence_generic_structured("Is Duplicate UDR Check performed?", chunks)
        self.assertEqual(file_result.status, "EVIDENCE_FOUND")
        self.assertIn("CRC", file_result.passages[0].text)
        self.assertEqual(udr_result.status, "EVIDENCE_FOUND")
        self.assertIn("N/A", udr_result.passages[0].text)

    def test_generic_duplicate_wording_fails_closed_when_concepts_coexist(self):
        chunks = [
            ChunkRecord("Duplicate files are detected using CRC.", {"source_file": "x.docx", "block_type": "paragraph"}),
            ChunkRecord("Duplicate UDR Check = N/A", {"source_file": "x.docx", "block_type": "key_value"}),
        ]
        result = extract_evidence_generic_structured("What is the duplicate checking status?", chunks)
        self.assertEqual(result.status, "NO_EXPLICIT_EVIDENCE")

    def test_reviewer_paraphrases_and_directory_paraphrases(self):
        chunks = [
            ChunkRecord("Revue par = Alice", {"source_file": "x.docx", "block_type": "key_value"}),
            ChunkRecord("FileDirectory = /data/out", {"source_file": "x.docx", "block_type": "key_value"}),
        ]
        for query in ("Who reviewed the specification?", "Who was the reviewer?", "Qui a revu la spécification ?", "Qui a effectué la revue ?"):
            result = extract_evidence_generic_structured(query, chunks)
            self.assertEqual(result.status, "EVIDENCE_FOUND", query)
            self.assertIn("Alice", result.passages[0].text)
        for query in ("What is the output directory?", "Where are output files written?", "What folder receives the output files?", "Quel est le répertoire de sortie ?"):
            result = extract_evidence_generic_structured(query, chunks)
            self.assertEqual(result.status, "EVIDENCE_FOUND", query)
            self.assertIn("/data/out", result.passages[0].text)

    def test_redacted_password_does_not_block_allowed_host(self):
        chunks = [ChunkRecord("System name = BI | Host = 10.0.0.1 | Password = [REDACTED]", {"source_file": "x.docx", "block_type": "column_record"})]
        result = extract_evidence_generic_structured("What is the BI host?", chunks)
        self.assertEqual(result.status, "EVIDENCE_FOUND")
        self.assertIn("10.0.0.1", result.passages[0].text)

    def test_row_qualified_relations_beat_global_fields(self):
        chunks = [
            ChunkRecord("Document Author = Alice Global", {"source_file": "x.docx", "block_type": "key_value"}),
            ChunkRecord("Version = V1.0 | Author = Bob One", {"source_file": "x.docx", "block_type": "table_row"}),
            ChunkRecord("Version = V2.0 | Author = Carol Two", {"source_file": "x.docx", "block_type": "table_row"}),
            ChunkRecord("Version = V3.0 | Author = David Three", {"source_file": "x.docx", "block_type": "table_row"}),
            ChunkRecord("Environment = Test | Endpoint = /test/api", {"source_file": "x.docx", "block_type": "table_row"}),
            ChunkRecord("Environment = Production | Endpoint = /prod/api", {"source_file": "x.docx", "block_type": "table_row"}),
        ]
        v2 = extract_evidence_generic_structured("Who authored version V2.0?", chunks)
        v3 = extract_evidence_generic_structured("Who authored version V3.0?", chunks)
        global_author = extract_evidence_generic_structured("Who wrote the document?", chunks)
        production = extract_evidence_generic_structured("What is the Production endpoint?", chunks)
        self.assertIn("Carol Two", v2.passages[0].text)
        self.assertIn("David Three", v3.passages[0].text)
        self.assertIn("Alice Global", global_author.passages[0].text)
        self.assertIn("/prod/api", production.passages[0].text)

    def test_qualifier_without_same_record_field_fails_closed(self):
        chunks = [
            ChunkRecord("Document Author = Alice Global", {"source_file": "x.docx", "block_type": "key_value"}),
            ChunkRecord("Version = V2.0 | Description = Changed", {"source_file": "x.docx", "block_type": "table_row"}),
        ]
        result = extract_evidence_generic_structured("Who authored version V2.0?", chunks)
        self.assertEqual(result.status, "NO_EXPLICIT_EVIDENCE")

    def test_entity_qualified_directory_rejects_global_and_other_entities(self):
        chunks = [
            ChunkRecord("Directory = /collector/input", {"source_file": "x.docx", "block_type": "key_value"}),
            ChunkRecord("System name = WAREHOUSE | FileDirectory = TO BE DEFINED", {"source_file": "x.docx", "block_type": "column_record"}),
            ChunkRecord("System name = ANALYTICS | FileDirectory = /analytics/output", {"source_file": "x.docx", "block_type": "column_record"}),
        ]
        analytics = extract_evidence_generic_structured("What is the ANALYTICS output directory?", chunks)
        warehouse = extract_evidence_generic_structured("What is the WAREHOUSE output directory?", chunks)
        self.assertIn("/analytics/output", analytics.passages[0].text)
        self.assertIn("TO BE DEFINED", warehouse.passages[0].text)


if __name__ == "__main__":
    unittest.main()
