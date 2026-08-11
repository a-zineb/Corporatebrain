from __future__ import annotations

import random
from pathlib import Path

import pytest

from canonical_rag import answer_direct
from document_normalizer import normalize_document


def _zebra():
    paths = list(Path("doc_storage_v2").glob("*ZEBRA*.pdf"))
    if not paths:
        pytest.skip("real ZEBRA PDF unavailable")
    path = paths[0]
    return normalize_document(path.read_bytes(), path.name)


def test_zebra_pdf_reconstructs_cross_page_tables_and_sections():
    document = _zebra()
    assert len(document.logical_tables) == 9
    history = next(table for table in document.logical_tables if table.shape == "VERSION_HISTORY")
    glossary = next(table for table in document.logical_tables if table.shape == "GLOSSARY")
    distribution = next(table for table in document.logical_tables if table.shape == "MATRIX")
    assert (len(history.rows), history.logical_columns) == (4, 4)
    assert (len(glossary.rows), glossary.logical_columns) == (40, 2)
    assert (len(distribution.rows), distribution.logical_columns) == (8, 5)
    assert distribution.metadata["cross_page"] is True
    assert distribution.metadata["page_start"] == 8
    assert distribution.metadata["page_end"] == 9


@pytest.mark.parametrize("question,expected", [
    ("BI host?", "172.26.60.12"),
    ("DWH host?", "172.21.75.61"),
    ("REQLEG host?", "172.21.14.31"),
    ("BI username?", "mz_user"),
    ("BI FileDirectory?", "/data/input/mz/om/"),
    ("parameter controlling DWH?", "ZEBRA_TO_DWH"),
    ("parameter controlling BI?", "ZEBRA_TO_BI"),
    ("parameter controlling SVR CRA?", "ZEBRA_TO_SVRCRA"),
    ("maximum cache age?", "30 jours"),
    ("retention?", "30 jours"),
    ("copyright?", "2020, Atos"),
    ("DWH instance 1?", "Brut"),
])
def test_zebra_qualified_answers(question, expected):
    result = answer_direct(question, _zebra())
    assert result.status == "ANSWER"
    assert result.answer == expected


def test_zebra_sections_multi_values_and_explicit_empty_glossary():
    document = _zebra()
    history = answer_direct("historique des modifications?", document)
    assert history.result_type == "SECTION_RESULT"
    assert all(value in history.answer for value in ("0.1", "1.0", "1.1", "21/11/2020", "09/10/2021"))

    hosts = answer_direct("host?", document)
    assert hosts.result_type == "MULTI_VALUE"
    assert all(value in hosts.answer for value in ("DWH", "BI", "FTP CRA", "REQLOG"))

    directories = answer_direct("directory?", document)
    assert directories.result_type == "MULTI_VALUE"
    assert "/data/input/mz/om/" in directories.answer
    assert "/srv/pretupsvar/DWH/DWH Final Data" in directories.answer

    empty = answer_direct("Proximity?", document)
    assert empty.status == "EXPLICIT_TERM_WITHOUT_VALUE"
    assert empty.result_type == "EXPLICIT_EMPTY_VALUE"


def test_zebra_query_order_is_stateless():
    document = _zebra()
    questions = ["host?", "directory?", "BI host?", "historique des modifications?",
                 "filename pattern?", "audit tables?", "Proximity?"]
    baseline = {question: answer_direct(question, document) for question in questions}
    for ordered in (list(reversed(questions)), random.Random(17).sample(questions, len(questions))):
        for question in ordered:
            assert answer_direct(question, document) == baseline[question]
