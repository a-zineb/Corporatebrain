from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import time

from dotenv import load_dotenv

from backend.llm.provider import GenerationProvider, ProviderError


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().casefold() in {"1", "true", "yes", "on"}


class GeminiProvider(GenerationProvider):
    """Server-side adapter for the official Google Gen AI SDK."""

    def __init__(self, *, api_key: str | None = None, model: str | None = None,
                 timeout: float | None = None, temperature: float | None = None,
                 top_p: float | None = None, max_output_tokens: int | None = None,
                 enable_streaming: bool | None = None, client=None) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.timeout = timeout if timeout is not None else float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
        self.temperature = temperature if temperature is not None else float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
        self.top_p = top_p if top_p is not None else float(os.getenv("GEMINI_TOP_P", "0.95"))
        self.max_output_tokens = (max_output_tokens if max_output_tokens is not None
                                  else int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048")))
        self.enable_streaming = (_bool_env("GEMINI_ENABLE_STREAMING")
                                 if enable_streaming is None else enable_streaming)
        self._client = client

    @property
    def configured(self) -> bool:
        return bool(self.api_key or self._client is not None)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ProviderError(
                "AI Answer is not configured. Add GEMINI_API_KEY to the backend environment."
            )
        from google import genai
        from google.genai import types
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
        )
        return self._client

    def _config(self):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=(
                "You are Corporate Brain, an enterprise knowledge assistant. Write a natural, clear, "
                "professional answer using only the supplied evidence. Never invent company facts, citations, "
                "source names, or missing values. Use headings, bullets, or Markdown tables when useful. Include "
                "all supplied records when the user asks for all/list/every. If evidence is insufficient or the "
                "request is ambiguous, explain that helpfully or ask one concise clarification question. Answer "
                "in the language of the current user question. Sources are handled separately by the application."
            ),
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
            # Corporate Brain already performs retrieval and evidence reasoning.
            # Gemini 3 requires thinking, so use its lowest supported level.
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        )

    @staticmethod
    def _safe_error(error: Exception) -> ProviderError:
        message = str(error).casefold()
        if any(marker in message for marker in ("api key", "unauthenticated", "permission_denied", "401", "403")):
            return ProviderError("The configured AI provider rejected the API credentials.")
        if any(marker in message for marker in ("resource_exhausted", "rate limit", "quota", "429")):
            return ProviderError("AI Answer is temporarily rate-limited. Please try again shortly.")
        if any(marker in message for marker in ("not_found", "not found", "no longer available", "404")):
            return ProviderError("The configured Gemini model is unavailable. Update GEMINI_MODEL in the backend environment.")
        if any(marker in message for marker in ("timeout", "timed out", "deadline", "connection")):
            return ProviderError("AI Answer is temporarily unavailable because the provider could not be reached.")
        return ProviderError("AI Answer is temporarily unavailable.")

    def generate(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                response = self._get_client().models.generate_content(
                    model=self.model, contents=prompt, config=self._config(),
                )
                text = str(response.text or "").strip()
                if not text:
                    raise ProviderError("I couldn't complete that AI response. Please retry.")
                return text
            except ProviderError:
                raise
            except Exception as exc:
                message = str(exc).casefold()
                retryable = any(marker in message for marker in (
                    "503", "500", "502", "504", "unavailable", "overloaded",
                    "timeout", "timed out", "deadline", "connection",
                ))
                if retryable and attempt < 2:
                    time.sleep(1.0 + attempt)
                    continue
                raise self._safe_error(exc) from exc
        raise ProviderError("AI Answer is temporarily unavailable.")

    def stream(self, prompt: str) -> Iterator[str]:
        if not self.enable_streaming:
            yield self.generate(prompt)
            return
        try:
            chunks = self._get_client().models.generate_content_stream(
                model=self.model, contents=prompt, config=self._config(),
            )
            for chunk in chunks:
                if chunk.text:
                    yield chunk.text
        except ProviderError:
            raise
        except Exception as exc:
            raise self._safe_error(exc) from exc
