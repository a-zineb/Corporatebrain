"""Trace-driven forensic classification and artifact content checks."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import rag_forensics
import rag_pipeline


def trace(*, failure=None, citations=None):
    chunk = rag_pipeline.ChunkRecord("content", {"source_file": "policy.pdf"})
    source = rag_pipeline.PromptSource(1, "policy.pdf", "Page 1", "content", "policy.pdf")
    return rag_pipeline.PipelineTrace(
        query="question",
        rewritten_query="rewritten question",
        retrieval=rag_pipeline.RetrievalResult(filtered_chunks=(chunk,)),
        generation=rag_pipeline.GenerationResult(response="answer"),
        citations=citations or rag_pipeline.CitationResult(display_sources=(source,)),
        failure=failure,
    )


class ForensicTests(unittest.TestCase):
    """Ensure categories and reports are derived only from supplied traces."""

    def test_failure_categories_have_deterministic_priority(self):
        generation = rag_forensics.classify_trace(
            trace(failure=rag_pipeline.PipelineFailure("generation_error", "offline")), {}, {"mode": "answer"}
        )
        self.assertEqual(generation.category, "generation_error")

        missing = rag_forensics.classify_trace(trace(), {"recall_at_k": 0.0}, {"mode": "answer"})
        self.assertEqual(missing.category, "expected_source_missing")

        invalid = rag_forensics.classify_trace(
            trace(citations=rag_pipeline.CitationResult(cited_source_ids=(99,), invalid_source_ids=(99,))),
            {"recall_at_k": 1.0, "citation_valid": False},
            {"mode": "answer"},
        )
        self.assertEqual(invalid.category, "invalid_citation")

        refusal = rag_forensics.classify_trace(
            trace(), {"refusal_correct": False}, {"mode": "refuse_no_coverage"}
        )
        self.assertEqual(refusal.category, "refusal_incorrect")

    def test_json_and_markdown_reports_contain_trace_evidence(self):
        finding = rag_forensics.classify_trace(trace(), {"recall_at_k": 0.0}, {"mode": "answer"})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rag_forensics.write_failure_report("case-7", finding, trace(), output)
            json_report = (output / "case-7.json").read_text(encoding="utf-8")
            markdown_report = (output / "case-7.md").read_text(encoding="utf-8")
            self.assertIn("expected_source_missing", json_report)
            self.assertIn("rewritten question", json_report)
            self.assertIn("# Forensic report: case-7", markdown_report)
            self.assertIn("policy.pdf", markdown_report)

    def test_tracer_keeps_standalone_entrypoint_and_uses_shared_forensics(self):
        tracer_source = (Path(__file__).resolve().parents[1] / "rag_tracer.py").read_text(encoding="utf-8")
        self.assertIn("import rag_forensics", tracer_source)
        self.assertIn("def export_pipeline_trace_report", tracer_source)
        self.assertIn("def run_single_trace", tracer_source)
        self.assertIn("def main", tracer_source)


if __name__ == "__main__":
    unittest.main()
