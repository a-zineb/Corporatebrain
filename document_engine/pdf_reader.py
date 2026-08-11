"""Geometry-aware PDF reconstruction into the shared logical table model."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import re
from typing import Any

import fitz

from .models import LogicalCell, LogicalTable


@dataclass(frozen=True, slots=True)
class PdfTextBlock:
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    font_size: float
    bold: bool
    section_path: tuple[str, ...]
    block_type: str = "paragraph"


@dataclass(frozen=True, slots=True)
class PdfStructure:
    blocks: tuple[PdfTextBlock, ...]
    logical_tables: tuple[LogicalTable, ...]
    warnings: tuple[str, ...] = ()


def _clean(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ")
    text = re.sub(r"(?<=\w)\s*_\s*(?=\w)", "_", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" _")
    return text


def _shape(rows: list[list[str]]) -> str:
    if not rows:
        return "UNKNOWN_TABLE"
    header = {_clean(value).casefold() for value in rows[0]}
    first = {_clean(row[0]).casefold() for row in rows if row}
    if header.intersection({"vers.", "version"}) and {"date", "auteur"} <= header:
        return "VERSION_HISTORY"
    if header.intersection({"param", "parameter"}) and header.intersection({"value", "valeur"}):
        return "PARAMETER_TABLE"
    matrix_fields = {"protocol", "host", "host*", "username", "username*", "filedirectory", "filedirectory*", "cdr format"}
    if len(rows[0]) >= 3 and first.intersection(matrix_fields):
        return "MATRIX"
    if len(rows[0]) == 2 and _clean(rows[0][0]).casefold() in {"article", "terme", "abbreviation"}:
        return "GLOSSARY"
    if len(rows[0]) == 2:
        return "KEY_VALUE"
    if len(rows[0]) >= 3:
        return "RECTANGULAR_RECORD_TABLE"
    return "UNKNOWN_TABLE"


def _table_from_rows(rows: list[list[Any]], table_index: int, page: int,
                     bbox: tuple[float, float, float, float], section_path: tuple[str, ...]) -> LogicalTable:
    width = max((len(row) for row in rows), default=0)
    padded = [[_clean(value) for value in row] + [""] * (width - len(row)) for row in rows]
    # Native PDF extraction represents vertically merged parents as blanks.
    # Propagate only for relational tables whose first column is an entity.
    shape = _shape(padded)
    if shape == "PARAMETER_TABLE":
        parent = ""
        for row in padded[1:]:
            if row[0]:
                parent = row[0]
            elif parent:
                row[0] = parent
        for row in padded[1:]:
            if len(row) > 1:
                parameter = row[1].strip(" _")
                if re.fullmatch(r"[A-Z][A-Z0-9]+(?:\s+[A-Z0-9]+)+", parameter):
                    row[1] = re.sub(r"\s+", "_", parameter)
    if shape == "MATRIX":
        for row in padded:
            if row and row[0].casefold().rstrip("*").strip() in {"username", "login"}:
                row[1:] = [re.sub(r"(?i)^([a-z0-9]+)\s+user$", r"\1_user", value) for value in row[1:]]
    secret_label = re.compile(r"(?i)password|passwd|token|api[_ -]?key|secret|private\s+key")
    secret_assignment = re.compile(
        r"(?i)(\b(?:password|passwd|token|api[_ -]?key|secret|private\s+key)\s*[:=]\s*)(.*?)(?=\s+\w[\w ]{0,30}\s*[:=]|$)"
    )
    for row in padded:
        if row and secret_label.search(row[0]):
            row[1:] = ["[REDACTED]" if value else value for value in row[1:]]
        else:
            row[:] = [secret_assignment.sub(r"\1[REDACTED]", value) for value in row]
    logical_rows: list[tuple[LogicalCell, ...]] = []
    for row_index, row in enumerate(padded):
        logical_rows.append(tuple(
            LogicalCell(
                row_index=row_index, column_index=column_index, row_span=1, column_span=1,
                text=value, paragraphs=(value,) if value else (), style_metadata={},
                source_xml_identity=sha256(
                    f"pdf:{page}:{table_index}:{row_index}:{column_index}".encode()
                ).hexdigest(),
            )
            for column_index, value in enumerate(row)
        ))
    return LogicalTable(
        table_id=f"pdf-table-{table_index}", section=section_path[-1] if section_path else None,
        section_path=section_path, rows=tuple(logical_rows), logical_columns=width,
        source_order=table_index, page=page, shape=shape,
        metadata={"bbox": tuple(round(float(value), 3) for value in bbox),
                  "page_start": page, "page_end": page, "native_pdf_table": True},
    )


def _merge_tables(tables: list[LogicalTable]) -> list[LogicalTable]:
    merged: list[LogicalTable] = []
    for table in tables:
        if merged:
            previous = merged[-1]
            consecutive = table.page == int(previous.metadata.get("page_end", previous.page or 0)) + 1
            compatible_section = table.section_path == previous.section_path
            same_width = table.logical_columns == previous.logical_columns
            continuation = False
            if consecutive and compatible_section and same_width:
                if previous.shape == "MATRIX":
                    continuation = table.shape in {"MATRIX", "RECTANGULAR_RECORD_TABLE"} and bool(table.rows)
                elif previous.shape == "GLOSSARY":
                    continuation = table.logical_columns == 2 and table.shape == "KEY_VALUE"
            if continuation:
                offset = len(previous.rows)
                appended = tuple(tuple(replace(cell, row_index=cell.row_index + offset) for cell in row)
                                 for row in table.rows)
                metadata = {**previous.metadata, "page_end": table.metadata.get("page_end", table.page),
                            "cross_page": True, "fragment_count": int(previous.metadata.get("fragment_count", 1)) + 1}
                merged[-1] = replace(previous, rows=previous.rows + appended, metadata=metadata)
                continue
        merged.append(table)
    return [replace(table, source_order=index, table_id=f"pdf-table-{index}")
            for index, table in enumerate(merged)]


def _coordinate_table_candidates(page_dict: dict[str, Any]) -> list[tuple[list[list[str]], tuple[float, float, float, float]]]:
    """Find borderless row/column alignments when native detection finds none."""
    records: list[tuple[float, tuple[int, ...], list[str], tuple[float, float, float, float]]] = []
    positioned: list[dict[str, Any]] = []
    for raw in page_dict.get("blocks", []):
        for line in raw.get("lines", []):
            spans = [span for span in line.get("spans", []) if _clean(span.get("text"))]
            positioned.extend(spans)
    baselines: dict[int, list[dict[str, Any]]] = {}
    for span in positioned:
        baseline = round(float(span["bbox"][1]) / 3) * 3
        baselines.setdefault(baseline, []).append(span)
    for spans in baselines.values():
            if len(spans) < 2:
                continue
            spans.sort(key=lambda span: float(span["bbox"][0]))
            signature = tuple(round(float(span["bbox"][0]) / 8) * 8 for span in spans)
            bbox = (min(float(span["bbox"][0]) for span in spans),
                    min(float(span["bbox"][1]) for span in spans),
                    max(float(span["bbox"][2]) for span in spans),
                    max(float(span["bbox"][3]) for span in spans))
            records.append((bbox[1], signature, [_clean(span.get("text")) for span in spans], bbox))
    groups: dict[tuple[int, ...], list[tuple[float, list[str], tuple[float, float, float, float]]]] = {}
    for y0, signature, values, bbox in records:
        groups.setdefault(signature, []).append((y0, values, bbox))
    candidates: list[tuple[list[list[str]], tuple[float, float, float, float]]] = []
    for rows in groups.values():
        if len(rows) < 3:
            continue
        rows.sort(key=lambda item: item[0])
        values = [row[1] for row in rows]
        boxes = [row[2] for row in rows]
        bbox = (min(box[0] for box in boxes), min(box[1] for box in boxes),
                max(box[2] for box in boxes), max(box[3] for box in boxes))
        candidates.append((values, bbox))
    return candidates


def read_pdf_structure(data: bytes) -> PdfStructure:
    """Reconstruct PDF headings, sections, geometry blocks and native tables."""
    blocks: list[PdfTextBlock] = []
    tables: list[LogicalTable] = []
    warnings: list[str] = []
    headings: dict[int, str] = {}
    table_index = 0
    with fitz.open(stream=data, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            page_dict = page.get_text("dict")
            page_text = _clean(page.get_text("text"))
            is_toc = "table des mati" in page_text.casefold()
            page_blocks: list[PdfTextBlock] = []
            for raw in page_dict.get("blocks", []):
                lines = raw.get("lines")
                if not lines:
                    continue
                spans = [span for line in lines for span in line.get("spans", []) if _clean(span.get("text"))]
                if not spans:
                    continue
                text = _clean(" ".join(str(span.get("text", "")) for span in spans))
                size = max(float(span.get("size", 0)) for span in spans)
                bold = any("bold" in str(span.get("font", "")).casefold() for span in spans)
                numbered = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", text)
                is_heading = not is_toc and (size >= 13.5 or (numbered and (bold or size >= 11.0)))
                if is_heading:
                    level = numbered.group(1).count(".") + 1 if numbered else 1
                    title = numbered.group(2).strip() if numbered else text
                    headings[level] = title
                    headings = {key: value for key, value in headings.items() if key <= level}
                path = tuple(value for _, value in sorted(headings.items()))
                page_blocks.append(PdfTextBlock(
                    text=text, page=page_number, bbox=tuple(float(value) for value in raw["bbox"]),
                    font_size=size, bold=bold, section_path=path,
                    block_type="heading" if is_heading else "paragraph",
                ))
            blocks.extend(page_blocks)
            try:
                native_tables = page.find_tables().tables
            except Exception as exc:
                native_tables = []
                warnings.append(f"Page {page_number}: native table detection failed ({type(exc).__name__}).")
            if not native_tables and not is_toc:
                for rows, bbox in _coordinate_table_candidates(page_dict):
                    before = [item for item in page_blocks if item.bbox[1] <= bbox[3]]
                    path = before[-1].section_path if before else tuple(value for _, value in sorted(headings.items()))
                    table = _table_from_rows(rows, table_index, page_number, bbox, path)
                    table = replace(table, metadata={**table.metadata, "native_pdf_table": False,
                                                     "coordinate_reconstruction": True})
                    tables.append(table)
                    table_index += 1
            for native in native_tables:
                rows = native.extract()
                if rows:
                    before_table = [item for item in page_blocks if item.bbox[1] <= native.bbox[3]]
                    table_path = before_table[-1].section_path if before_table else tuple(
                        value for _, value in sorted(headings.items())
                    )
                    tables.append(_table_from_rows(rows, table_index, page_number, native.bbox, table_path))
                    table_index += 1
    repeated = {}
    for block in blocks:
        if block.font_size <= 8.5:
            repeated.setdefault(_clean(block.text).casefold(), set()).add(block.page)
    repeated_keys = {key for key, pages in repeated.items() if len(pages) >= 3}
    seen_repeated: set[str] = set()
    deduplicated: list[PdfTextBlock] = []
    for block in blocks:
        key = _clean(block.text).casefold()
        if key in repeated_keys:
            if key in seen_repeated:
                continue
            seen_repeated.add(key)
        deduplicated.append(block)
    return PdfStructure(tuple(deduplicated), tuple(_merge_tables(tables)), tuple(warnings))
