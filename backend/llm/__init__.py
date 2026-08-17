"""Generation providers used by Corporate Brain AI Answer only."""

import os

from backend.llm.gemini_provider import GeminiProvider
from backend.llm.provider import GenerationProvider, ProviderError


def get_generation_provider() -> GenerationProvider:
    provider = os.getenv("AI_PROVIDER", "gemini").strip().casefold()
    if provider != "gemini":
        raise ProviderError(f"Unsupported AI_PROVIDER: {provider or '(empty)'}. Use 'gemini'.")
    return GeminiProvider()


__all__ = ["GenerationProvider", "GeminiProvider", "ProviderError", "get_generation_provider"]
