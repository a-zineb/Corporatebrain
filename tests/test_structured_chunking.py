import unittest

from structured_ingestion import NormalizedBlock, normalized_blocks_to_chunks


class StructuredChunkingTests(unittest.TestCase):
    def test_structured_records_are_atomic_and_contextual(self):
        blocks = [
            NormalizedBlock("Filename Pattern = p2pCommands.yyyy-mm-dd", "table_row", "p.docx", "Table 4", "Collecte", 4, 2, {"normalization_strategy": "key_value"}),
            NormalizedBlock("System name = BI | Protocol = SFTP | Host = 172.26.60.12 | username = mz_user | FileDirectory = /data/input/mz/p2p/", "table_row", "p.docx", "Table 6", "Distribution", 6, 1, {"normalization_strategy": "column_records"}),
        ]
        chunks = normalized_blocks_to_chunks(blocks, max_length=20)
        self.assertEqual(len(chunks), 2)
        self.assertIn("p2pCommands.yyyy-mm-dd", chunks[0].text)
        self.assertIn("FileDirectory = /data/input/mz/p2p/", chunks[1].text)
        self.assertEqual(chunks[1].section, "Distribution")

    def test_paragraph_splitting_and_overlap(self):
        block = NormalizedBlock("One short sentence. Two short sentence. Three short sentence.", "paragraph", "p.docx", "Body", "Section")
        chunks = normalized_blocks_to_chunks([block], max_length=25, overlap=8)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(set(chunks[0].text.split()) & set(chunks[1].text.split()))

    def test_deterministic_identity_and_no_timestamp(self):
        block = NormalizedBlock("Stable text", "paragraph", "p.docx", "Body")
        first = normalized_blocks_to_chunks([block])[0]
        second = normalized_blocks_to_chunks([block])[0]
        self.assertEqual(first.to_preview(), second.to_preview())
        self.assertEqual(first.chunk_ordinal, 0)
        self.assertNotIn("timestamp", first.to_preview())
        self.assertEqual(len(first.content_sha256), 64)

    def test_no_cross_table_grouping(self):
        blocks = [
            NormalizedBlock("A = 1", "table_row", "p.docx", "Table 1", table_index=1, metadata={"normalization_strategy": "key_value"}),
            NormalizedBlock("B = 2", "table_row", "p.docx", "Table 2", table_index=2, metadata={"normalization_strategy": "key_value"}),
        ]
        chunks = normalized_blocks_to_chunks(blocks)
        self.assertEqual([c.table_index for c in chunks], [1, 2])
        self.assertEqual([c.source_block_indices for c in chunks], [(0,), (1,)])

    def test_redaction_is_preserved(self):
        block = NormalizedBlock("Password = [REDACTED]", "table_row", "p.docx", "Table 1", metadata={"normalization_strategy": "key_value"})
        chunk = normalized_blocks_to_chunks([block])[0]
        self.assertNotIn("secret-value", chunk.text)
        self.assertIn("[REDACTED]", chunk.text)


if __name__ == "__main__":
    unittest.main()
