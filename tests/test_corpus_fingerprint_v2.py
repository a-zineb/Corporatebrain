import unittest

from corpus_fingerprint import FingerprintSchemaError, canonical_records_v2, compare_fingerprints, fingerprint_v2


def records():
    return [
        {
            "id": "doc_20260101_010101_chunk_1",
            "document": "Deuxième passage.",
            "metadata": {
                "source_file": "doc.pdf", "file_hash": "abc", "application": "MZ",
                "location": "Page 2", "timestamp_ingest": "20260101_010101", "absolute_path": "C:/volatile",
            },
        },
        {
            "id": "doc_20260101_010101_chunk_0",
            "document": "Premier passage café.",
            "metadata": {
                "source_file": "doc.pdf", "file_hash": "abc", "application": "MZ",
                "location": "Page 1", "timestamp_ingest": "20260101_010101",
            },
        },
    ]


class FingerprintV2Tests(unittest.TestCase):
    def test_timestamp_ids_and_insertion_order_are_ignored(self):
        first = fingerprint_v2(records())
        changed = records()[::-1]
        for item in changed:
            item["id"] = item["id"].replace("20260101_010101", "20990101_999999")
            item["metadata"]["timestamp_ingest"] = "20990101_999999"
        self.assertEqual(first["corpus_sha256"], fingerprint_v2(changed)["corpus_sha256"])
        self.assertEqual(first["metadata_sha256"], fingerprint_v2(changed)["metadata_sha256"])

    def test_text_boundary_document_and_metadata_changes_are_detected(self):
        base = fingerprint_v2(records())
        changed = records(); changed[0]["document"] = "Changed passage."
        self.assertNotEqual(base["corpus_sha256"], fingerprint_v2(changed)["corpus_sha256"])
        changed = records(); changed[0]["metadata"]["location"] = "Page 9"
        self.assertNotEqual(base["metadata_sha256"], fingerprint_v2(changed)["metadata_sha256"])
        changed = records() + [{"id": "new_20260101_chunk_0", "document": "new", "metadata": {"source_file": "new.pdf", "file_hash": "new"}}]
        self.assertNotEqual(base["corpus_sha256"], fingerprint_v2(changed)["corpus_sha256"])
        changed = records(); changed[1]["id"] = "doc_20260101_010101_chunk_9"
        self.assertNotEqual(base["corpus_sha256"], fingerprint_v2(changed)["corpus_sha256"])
        self.assertNotEqual(base["corpus_sha256"], fingerprint_v2(records()[:1])["corpus_sha256"])

    def test_volatile_fields_and_unicode_are_stable(self):
        a = fingerprint_v2(records())
        b = records(); b[0]["metadata"]["filesystem_mtime"] = 123; b[0]["metadata"]["absolute_path"] = "D:/other"
        self.assertEqual(a["metadata_sha256"], fingerprint_v2(b)["metadata_sha256"])
        self.assertEqual(fingerprint_v2(records())["corpus_sha256"], fingerprint_v2(records())["corpus_sha256"])

    def test_schema_version_and_comparison_rules(self):
        v2 = fingerprint_v2(records())
        self.assertEqual(v2["fingerprint_schema_version"], 2)
        self.assertEqual(compare_fingerprints(v2, {"fingerprint_schema_version": 1})["status"], "NOT_COMPARABLE")
        with self.assertRaises(FingerprintSchemaError):
            compare_fingerprints({"fingerprint_schema_version": 99}, {"fingerprint_schema_version": 99})

    def test_canonical_records_have_stable_identity(self):
        rows = canonical_records_v2(records())
        self.assertEqual([r["chunk_ordinal"] for r in rows], [0, 1])
        self.assertIn("content_sha256", rows[0])
        self.assertIn("file_hash", rows[0])
        self.assertNotIn("timestamp_ingest", rows[0]["stable_metadata"])


if __name__ == "__main__":
    unittest.main()
