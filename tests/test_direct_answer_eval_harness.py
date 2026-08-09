import unittest

from direct_answer_eval import EvaluationCase, run_evaluation


class ReadOnlyCollection:
    def __init__(self):
        self.rows = [("Reviewer = Alice", {"file_hash": "doc-1", "source_file": "x.docx", "block_type": "key_value"}, "1")]

    def get(self, where=None, include=None):
        value = (where or {}).get("file_hash")
        rows = [row for row in self.rows if row[1].get("file_hash") == value]
        return {"documents": [r[0] for r in rows], "metadatas": [r[1] for r in rows], "ids": [r[2] for r in rows]}


class DirectAnswerEvaluationHarnessTests(unittest.TestCase):
    def test_harness_is_read_only_and_reports_metrics(self):
        result = run_evaluation(ReadOnlyCollection(), [
            EvaluationCase("doc-1", "Who reviewed the document?", "Alice"),
            EvaluationCase("doc-1", "What is the deployment region?", expected_status="NO_EXPLICIT_EVIDENCE"),
        ])
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["pass"], 2)
        self.assertEqual(result["cross_document"], 0)


if __name__ == "__main__":
    unittest.main()
