import unittest

import rag_pipeline


def chunk(text, block_type="table_row", source_file="p2p.docx"):
    return rag_pipeline.ChunkRecord(
        text=text,
        metadata={"block_type": block_type, "source_file": source_file, "location": "Table 0"},
    )


class ExhaustiveSpecificEvidenceTests(unittest.TestCase):
    def test_author_requires_author_relation_not_reviewer_or_approver(self):
        result = rag_pipeline.extract_evidence_exhaustive_specific(
            "Who wrote the P2P specification?",
            [chunk("Revue par : = Reviewer"), chunk("Approuvé par : = Approver"), chunk("Écrit par : = Omar EL HIMASS")],
        )
        self.assertEqual(result.status, "EVIDENCE_FOUND")
        self.assertIn("Omar EL HIMASS", result.passages[0].text)

    def test_empty_approval_is_not_evidence(self):
        result = rag_pipeline.extract_evidence_exhaustive_specific(
            "Who approved the document?", [chunk("Approuvé par : =")]
        )
        self.assertEqual(result.status, "NO_EXPLICIT_EVIDENCE")

    def test_archive_placeholder_is_explicit(self):
        result = rag_pipeline.extract_evidence_exhaustive_specific(
            "What is the exact archive directory?", [chunk("Dossier d’archivage = TO BE DEFINED")]
        )
        self.assertEqual(result.status, "EVIDENCE_FOUND")
        self.assertIn("TO BE DEFINED", result.passages[0].text)

    def test_entity_qualified_port_does_not_use_generic_port(self):
        result = rag_pipeline.extract_evidence_exhaustive_specific(
            "What is the DWH port?", [
                chunk("Connection Protocol = SFTP (port : 22)"),
                chunk("System name = DWH | Protocol = SFTP"),
            ]
        )
        self.assertEqual(result.status, "NO_EXPLICIT_EVIDENCE")

    def test_duplicate_mechanism_is_separate_from_parameter(self):
        result = rag_pipeline.extract_evidence_exhaustive_specific(
            "How are duplicate files detected?",
            [chunk("PARAM_CHECK_DUP_BATCH = Y"), chunk("Contrôle de redondance cyclique (CRC) utilisé")],
        )
        self.assertEqual(result.status, "EVIDENCE_FOUND")
        self.assertIn("CRC", result.passages[0].text)

    def test_repeated_selection_is_deterministic(self):
        chunks = [chunk("Dossier d’archivage = TO BE DEFINED"), chunk("Directory = /input")]
        first = rag_pipeline.extract_evidence_exhaustive_specific("What is the exact archive directory?", chunks)
        second = rag_pipeline.extract_evidence_exhaustive_specific("What is the exact archive directory?", chunks)
        self.assertEqual(first.to_json(), second.to_json())


if __name__ == "__main__":
    unittest.main()
