"""Immutable logical structures reconstructed before fact extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LogicalCell:
    row_index: int
    column_index: int
    row_span: int
    column_span: int
    text: str
    paragraphs: tuple[str, ...]
    style_metadata: dict[str, Any] = field(default_factory=dict)
    is_merged_continuation: bool = False
    source_xml_identity: str = ""


@dataclass(frozen=True, slots=True)
class LogicalTable:
    table_id: str
    section: str | None
    section_path: tuple[str, ...]
    rows: tuple[tuple[LogicalCell, ...], ...]
    logical_columns: int
    source_order: int
    page: int | None = None
    shape: str = "UNKNOWN_TABLE"
    metadata: dict[str, Any] = field(default_factory=dict)

    def values(self, *, propagate_vertical_merges: bool = True) -> list[list[str]]:
        """Return a positional grid; blank coordinates are never removed."""
        result: list[list[str]] = []
        for row in self.rows:
            values: list[str] = []
            for cell in row:
                if cell.is_merged_continuation and not propagate_vertical_merges:
                    values.append("")
                else:
                    values.append(cell.text)
            result.append(values)
        return result
