import unittest

from structured_ingestion import _row_record_blocks


class StructuredTableHeaderTests(unittest.TestCase):
    def test_rectangular_rows_preserve_headers(self):
        blocks = _row_record_blocks(
            [["Version", "Description", "Date", "Auteur"], ["V2.0", "Update", "2024-01-01", "Bob"]],
            "x.docx", "Table 0", None, 0,
        )
        self.assertEqual(blocks[0].text, "Version = V2.0 | Description = Update | Date = 2024-01-01 | Auteur = Bob")
        self.assertEqual(blocks[0].row_index, 1)

    def test_empty_cells_do_not_shift_columns(self):
        blocks = _row_record_blocks(
            [["Version", "Description", "Date", "Auteur"], ["V2.0", "", "2024-01-01", "Bob"]],
            "x.docx", "Table 0", None, 0,
        )
        self.assertEqual(blocks[0].text, "Version = V2.0 | Date = 2024-01-01 | Auteur = Bob")

    def test_repeated_headers_are_skipped_and_mixed_width_is_stable(self):
        rows = [["Version", "Description"], ["Version", "Description"], ["V2.0", "Update", "Extra"]]
        first = _row_record_blocks(rows, "x.docx", "Table 0", None, 0)
        second = _row_record_blocks(rows, "x.docx", "Table 0", None, 0)
        self.assertEqual(len(first), 1)
        self.assertEqual(first, second)
        self.assertIn("Column 3 = Extra", first[0].text)

    def test_credential_values_remain_redacted(self):
        blocks = _row_record_blocks(
            [["Field", "Password"], ["Password", "ActualSecret"]],
            "x.docx", "Table 0", None, 0,
        )
        self.assertNotIn("ActualSecret", blocks[0].text)
        self.assertIn("[REDACTED]", blocks[0].text)


if __name__ == "__main__":
    unittest.main()
