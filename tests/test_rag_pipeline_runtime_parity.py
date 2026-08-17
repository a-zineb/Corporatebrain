"""Strict rewrite and generation parity against the pre-1.6 app runtime."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import subprocess
import unittest

import rag_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "7d1b3f5"


def baseline_source() -> str:
    """Load the immutable app.py implementation that preceded Sub-phase 1.6."""

    return subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:app.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def baseline_function(name: str, namespace: dict[str, object]) -> object:
    """Compile one baseline top-level function without importing Streamlit."""

    tree = ast.parse(baseline_source())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function = copy.deepcopy(function)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, f"{BASELINE_COMMIT}:app.py", "exec"), namespace)
    return namespace[name]


def baseline_stream_call() -> ast.Call:
    """Find the baseline streaming Ollama request embedded in the app flow."""

    tree = ast.parse(baseline_source())
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ollama"
        and node.func.attr == "chat"
        and any(keyword.arg == "stream" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords)
    )


class RecordingGenerator:
    """Deterministic local Ollama double for request, output, and error parity."""

    def __init__(self, response=None, stream=None, error=None):
        self.response = response
        self.stream = stream
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.stream if kwargs.get("stream") else self.response


def clock_from(values):
    iterator = iter(values)
    return lambda: next(iterator)


class RuntimeParityTests(unittest.TestCase):
    """Prove the shared runtime preserves the pre-extraction production contract."""

    def test_rewrite_prompt_result_and_last_three_messages_match_baseline(self):
        baseline_generator = RecordingGenerator(response={"message": {"content": "  OCM means change management.  "}})
        shared_generator = RecordingGenerator(response={"message": {"content": "  OCM means change management.  "}})
        baseline = baseline_function("contextualize_query", {"ollama": baseline_generator})
        history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]

        expected = baseline("What does it mean?", history, "qwen3:8b")
        actual = rag_pipeline.rewrite_query(
            "What does it mean?",
            history,
            "qwen3:8b",
            shared_generator,
            clock=clock_from([1.0, 1.25]),
        )

        self.assertEqual(actual.query, expected)
        self.assertEqual(actual.latency_ms, 250.0)
        self.assertEqual(shared_generator.calls, baseline_generator.calls)
        self.assertNotIn("Utilisateur: one", shared_generator.calls[0]["messages"][0]["content"])
        self.assertIn("Utilisateur: three", shared_generator.calls[0]["messages"][0]["content"])

    def test_rewrite_no_history_empty_response_and_error_fallback_match_baseline(self):
        baseline = baseline_function("contextualize_query", {"ollama": RecordingGenerator()})
        self.assertEqual(baseline("Question", [], "qwen3:8b"), "Question")
        no_history = rag_pipeline.rewrite_query(
            "Question", [], "qwen3:8b", RecordingGenerator(), clock=clock_from([2.0, 2.0])
        )
        self.assertEqual(no_history.query, "Question")
        self.assertEqual(no_history.latency_ms, 0.0)

        empty = rag_pipeline.rewrite_query(
            "Question", [{"role": "user", "content": "context"}], "qwen3:8b",
            RecordingGenerator(response={"message": {"content": "   "}}),
            clock=clock_from([3.0, 3.1]),
        )
        self.assertEqual(empty.query, "Question")

        errors = []
        fallback = rag_pipeline.rewrite_query(
            "Question", [{"role": "user", "content": "context"}], "qwen3:8b",
            RecordingGenerator(error=RuntimeError("offline")),
            error_reporter=errors.append,
            clock=clock_from([4.0, 4.2]),
        )
        self.assertEqual(fallback.query, "Question")
        self.assertEqual(errors, ["Generation provider error in contextualize_query: offline"])

    def test_generation_request_output_callbacks_errors_and_latency_match_baseline_contract(self):
        baseline_call = baseline_stream_call()
        keyword_names = {keyword.arg for keyword in baseline_call.keywords}
        self.assertEqual(keyword_names, {"model", "messages", "options", "stream"})
        self.assertTrue(next(keyword.value.value for keyword in baseline_call.keywords if keyword.arg == "stream"))

        callbacks = []
        generator = RecordingGenerator(
            stream=[
                {"message": {"content": "Bon"}},
                {"message": {"content": ""}},
                {"message": {"content": "jour"}},
                {},
            ]
        )
        result = rag_pipeline.stream_generate(
            "certified prompt",
            "qwen3:8b",
            generator,
            on_token=callbacks.append,
            clock=clock_from([10.0, 10.5]),
        )
        self.assertEqual(
            generator.calls,
            [{
                "model": "qwen3:8b",
                "messages": [{"role": "user", "content": "certified prompt"}],
                "options": {"temperature": 0.2},
                "stream": True,
            }],
        )
        self.assertEqual(result.response, "Bonjour")
        self.assertTrue(result.streamed)
        self.assertEqual(result.latency_ms, 500.0)
        self.assertEqual(callbacks, ["Bon", "Bonjour"])

        with self.assertRaisesRegex(RuntimeError, "offline"):
            rag_pipeline.stream_generate(
                "certified prompt", "qwen3:8b", RecordingGenerator(error=RuntimeError("offline"))
            )

        clarification = rag_pipeline.stream_generate(
            "certified prompt", "qwen3:8b", RecordingGenerator(stream=[]), clarification_language="French"
        )
        self.assertEqual(clarification.response, rag_pipeline.build_clarification_message("French"))


if __name__ == "__main__":
    unittest.main()
