from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Filters(BaseModel):
    zone: list[str] = Field(default_factory=list)
    application: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    mode: Literal["direct", "ai"] = "direct"
    document_hash: str | None = None
    conversation_id: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    filters: Filters = Field(default_factory=Filters)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=50, ge=1, le=200)
    filters: Filters = Field(default_factory=Filters)


class SourceResponse(BaseModel):
    document: str
    file_hash: str
    file_type: str
    block_id: str
    location: str
    page: int | None = None
    sheet: str | None = None
    row: int | None = None
    section: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    status: str
    result_type: str
    language: str
    method: str
    conversation_id: str
    sources: list[SourceResponse]
    suggestions: list[str] = Field(default_factory=list)
    latency_ms: float

