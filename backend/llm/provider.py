from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class ProviderError(RuntimeError):
    """Safe, user-facing error raised by a generation provider."""


class GenerationProvider(ABC):
    """Boundary between Corporate Brain retrieval and text generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return one complete provider response."""

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield genuine provider text deltas when streaming is requested."""
        yield self.generate(prompt)
