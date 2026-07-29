"""Optional local judge adapter and deterministic-score separation checks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest

import rag_evaluator
import rag_judge
import rag_pipeline


class FakeGenerator:
    def __init__(self, payload=None, error=None, delay=0.0):
        self.payload = payload
        self.error = error
        self.delay = delay
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return {"message": {"content": self.payload}}


class FakeJudge:
    def evaluate(self, **_kwargs):
        return rag_judge.JudgeOutcome(
            "SCORED", {name: 0.5 for name in rag_judge.JUDGE_METRICS}
        )


def evaluation_result():
    trace = rag_pipeline.PipelineTrace(
        query="question",
        retrieval=rag_pipeline.RetrievalResult(filtered_chunks=(rag_pipeline.ChunkRecord("context"),)),
    )
    return {
        "case_id": "case-1",
        "response": "answer",
        "metrics": {"recall_at_k": 1.0},
        "trace": trace,
    }


class LocalJudgeTests(unittest.TestCase):
    """Judge mode is optional, local-only, and separate from deterministic outputs."""

    def test_native_adapter_returns_valid_scores_from_local_json(self):
        generator = FakeGenerator(json.dumps({name: 0.5 for name in rag_judge.JUDGE_METRICS}))
        outcome = rag_judge.LocalOllamaJudgeAdapter(generator, "qwen3:8b").evaluate(
            question="question", answer="answer", contexts=["context"], reference_answer="reference"
        )
        self.assertEqual(outcome.status, "SCORED")
        self.assertEqual(outcome.metrics["faithfulness"], 0.5)
        self.assertEqual(generator.calls[0]["options"], {"temperature": 0.0})

    def test_failures_return_not_run_without_retry_or_cloud_fallback(self):
        invalid = rag_judge.LocalOllamaJudgeAdapter(FakeGenerator("not json"), "qwen3:8b").evaluate(
            question="q", answer="a", contexts=[], reference_answer=None
        )
        self.assertEqual(invalid.status, "NOT_RUN")
        self.assertEqual(invalid.reason_code, "invalid_judge_output")

        provider = rag_judge.LocalOllamaJudgeAdapter(FakeGenerator(error=RuntimeError("offline")), "qwen3:8b").evaluate(
            question="q", answer="a", contexts=[], reference_answer=None
        )
        self.assertEqual(provider.status, "NOT_RUN")
        self.assertEqual(provider.reason_code, "provider_error")

        timeout = rag_judge.LocalOllamaJudgeAdapter(FakeGenerator("{}", delay=0.05), "qwen3:8b", 0.001).evaluate(
            question="q", answer="a", contexts=[], reference_answer=None
        )
        self.assertEqual(timeout.status, "NOT_RUN")
        self.assertEqual(timeout.reason_code, "timeout")

    def test_ragas_is_explicitly_disabled_until_dependency_smoke_test(self):
        outcome = rag_judge.RagasJudgeAdapter().evaluate()
        self.assertEqual(outcome.status, "NOT_RUN")
        self.assertEqual(outcome.reason_code, "dependency_not_verified")

    def test_judge_artifacts_are_separate_from_deterministic_metrics(self):
        case = {"id": "case-1", "expected_answer": "reference"}
        outcomes = rag_evaluator.evaluate_judge_results([evaluation_result()], [case], FakeJudge())
        self.assertEqual(outcomes[0]["outcome"]["status"], "SCORED")
        self.assertEqual(evaluation_result()["metrics"], {"recall_at_k": 1.0})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rag_evaluator.write_judge_reports(outcomes, output)
            self.assertTrue((output / "judge_cases.json").exists())
            self.assertTrue((output / "judge_summary.json").exists())
            self.assertTrue((output / "judge_summary.md").exists())
            payload = json.loads((output / "judge_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["scored_cases"], 1)


if __name__ == "__main__":
    unittest.main()
