"""Optional local-only model judge adapters for evaluation artifacts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Mapping, Protocol, Sequence


JUDGE_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass(frozen=True, slots=True)
class JudgeOutcome:
    """A model-judged result that never changes deterministic evaluator scores."""

    status: str
    metrics: Mapping[str, float | None]
    reason_code: str | None = None
    reason: str | None = None
    latency_ms: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a report-safe representation."""

        return asdict(self)


class JudgeAdapter(Protocol):
    """Minimal interface for optional, non-deterministic judge backends."""

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        reference_answer: str | None,
    ) -> JudgeOutcome:
        """Return a score set or a safe ``NOT_RUN`` result."""


def not_run(reason_code: str, reason: str, latency_ms: float | None = None) -> JudgeOutcome:
    """Normalize every optional-judge failure without affecting baseline metrics."""

    return JudgeOutcome("NOT_RUN", {name: None for name in JUDGE_METRICS}, reason_code, reason, latency_ms)


class LocalOllamaJudgeAdapter:
    """Judge evaluation artifacts with a supplied local Ollama-compatible client."""

    def __init__(self, generator: Any, model_name: str, timeout_seconds: float = 60.0):
        self.generator = generator
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: Sequence[str],
        reference_answer: str | None,
    ) -> JudgeOutcome:
        started = time.perf_counter()
        prompt = self._prompt(question, answer, contexts, reference_answer)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self.generator.chat,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        try:
            response = future.result(timeout=self.timeout_seconds)
        except TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return not_run("timeout", f"Judge exceeded {self.timeout_seconds} seconds.", self._latency(started))
        except Exception as error:
            executor.shutdown(wait=False, cancel_futures=True)
            return not_run("provider_error", str(error), self._latency(started))
        executor.shutdown(wait=False, cancel_futures=True)
        try:
            payload = json.loads(response["message"]["content"])
            metrics = {name: self._score(payload[name]) for name in JUDGE_METRICS}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return not_run("invalid_judge_output", str(error), self._latency(started))
        return JudgeOutcome("SCORED", metrics, latency_ms=self._latency(started))

    @staticmethod
    def _score(value: Any) -> float:
        score = float(value)
        if not 0.0 <= score <= 1.0:
            raise ValueError("Judge scores must be between 0 and 1.")
        return score

    @staticmethod
    def _latency(started: float) -> float:
        return (time.perf_counter() - started) * 1000

    @staticmethod
    def _prompt(question: str, answer: str, contexts: Sequence[str], reference_answer: str | None) -> str:
        """Build an evaluation-only judge prompt; it is not a production prompt."""

        return json.dumps(
            {
                "task": "Score this RAG answer from 0.0 to 1.0. Return JSON only.",
                "required_metrics": list(JUDGE_METRICS),
                "question": question,
                "answer": answer,
                "contexts": list(contexts),
                "reference_answer": reference_answer,
            },
            ensure_ascii=False,
        )


class RagasJudgeAdapter:
    """Disabled placeholder pending a real pinned-dependency smoke test on Python 3.14."""

    def evaluate(self, **_kwargs: Any) -> JudgeOutcome:
        return not_run(
            "dependency_not_verified",
            "RAGAS is disabled until pinned dependencies pass install/import smoke tests on Python 3.14.",
        )
