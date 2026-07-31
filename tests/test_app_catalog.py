"""Focused metadata-only tests for the sidebar knowledge-base catalog."""

import ast
import os
from pathlib import Path
import unittest


def catalog_function():
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "list_catalog_documents")
    namespace = {}
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["list_catalog_documents"]


class Collection:
    def __init__(self, metadatas): self.metadatas, self.calls = metadatas, []
    def get(self, **kwargs): self.calls.append(kwargs); return {"metadatas": self.metadatas}


class CatalogTests(unittest.TestCase):
    def test_enumerates_deduplicates_and_orders_all_metadata(self):
        collection = Collection([
            {"source_file": "b.pdf", "file_hash": "b", "application": "MZ", "geographical_entity": "OCM"},
            {"source_file": "a.docx", "file_hash": "a", "application": "KPSA", "geographical_entity": "OEG"},
            {"source_file": "duplicate.pdf", "file_hash": "b", "application": "MZ", "geographical_entity": "OCM"},
            {"source_file": "fallback.csv", "application": "MZ", "geographical_entity": "OCI"},
            {"source_file": "fallback.csv", "application": "MZ", "geographical_entity": "OCI"},
            {"application": "MZ"},
        ])
        result = catalog_function()(collection, {"application": "MZ"})
        self.assertEqual([item["source_file"] for item in result], ["a.docx", "fallback.csv", "b.pdf"])
        self.assertEqual(collection.calls, [{"where": {"application": "MZ"}, "include": ["metadatas"]}])
        self.assertEqual(os.path.basename(os.path.abspath(os.path.join("doc_storage_v2", result[0]["source_file"]))), "a.docx")

    def test_filter_is_forwarded_for_zone_and_application(self):
        collection = Collection([{"source_file": "in-scope.pdf", "file_hash": "1"}])
        chroma_filter = {"$and": [{"geographical_entity": "OCM"}, {"application": "MZ"}]}
        result = catalog_function()(collection, chroma_filter)
        self.assertEqual([item["source_file"] for item in result], ["in-scope.pdf"])
        self.assertEqual(collection.calls, [{"where": chroma_filter, "include": ["metadatas"]}])

    def test_empty_and_malformed_metadata_is_safe(self):
        collection = Collection([None, {}, {"source_file": ""}, {"source_file": 3}, {"source_file": "good.pdf", "file_hash": "g"}])
        self.assertEqual([item["source_file"] for item in catalog_function()(collection)], ["good.pdf"])
        self.assertEqual(collection.calls, [{"where": None, "include": ["metadatas"]}])

    def test_empty_metadata_is_safe(self):
        self.assertEqual(catalog_function()(Collection([])), [])


if __name__ == "__main__":
    unittest.main()
