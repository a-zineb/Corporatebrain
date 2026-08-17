from __future__ import annotations

import pytest

import canonical_rag
from document_normalizer import CanonicalBlock, CanonicalDocument


def document_with_sections() -> CanonicalDocument:
    blocks = (
        CanonicalBlock("h1", "Abstract", "heading", "project.docx", "hash", section="Abstract"),
        CanonicalBlock("p1", "This project collects and validates network records.", "paragraph",
                       "project.docx", "hash", section="Abstract"),
        CanonicalBlock("h2", "Requirements", "heading", "project.docx", "hash", section="Requirements"),
        CanonicalBlock("r1", "REQ-1 = Validate every input record", "requirement",
                       "project.docx", "hash", section="Requirements"),
    )
    return CanonicalDocument("doc", "hash", "project.docx", "docx", blocks)


@pytest.mark.parametrize("question", [
    "abstrait ?", "what's the abstract of this project?",
    "c'est quoi l'abstrait de ce projet ?", "quel est le résumé du document ?",
    "de quoi parle ce fichier ?",
])
def test_natural_abstract_phrasings_share_the_same_intent(question: str):
    result = canonical_rag.answer_direct(question, document_with_sections())
    assert result.status == "ANSWER"
    assert result.result_type == "ABSTRACT"
    assert "collects and validates" in result.answer


@pytest.mark.parametrize("question", [
    "exigences ?", "c'est quoi l'exigence de ce document ?",
    "quelles sont les exigences de ce fichier ?", "what are the requirements?",
])
def test_natural_requirement_phrasings_share_the_same_intent(question: str):
    result = canonical_rag.answer_direct(question, document_with_sections())
    assert result.status == "ANSWER"
    assert result.result_type == "REQUIREMENTS"
    assert "REQ-1" in result.answer
