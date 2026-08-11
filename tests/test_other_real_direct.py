from pathlib import Path

import pytest

from canonical_rag import answer_direct
from document_normalizer import normalize_document


REAL_CASES = {
    "P2P": {
        "What port is used for collection?": "22",
        "How are duplicate files detected?": "PARAM_CHECK_DUP_BATCH",
        "What is the BI directory?": "/data/input/mz/p2p/",
    },
    "SIMBOX": {
        "What is the server IP?": "172.21.70.36",
        "Is enrichment performed?": "No",
        "Is normalization performed?": "No",
        "What table stores flow parameters?": "MZ_PARAM",
    },
    "IN ZSmart": {
        "How many instances are planned?": "12",
        "What is the BI directory?": "/data/input/mz/in/",
        "What protocol does ROSCOM use?": "FTP",
    },
    "Extract IN": {
        "How many workflows exist?": "2",
        "What are the workflow names?": "IN, ZSmart",
        "Are files transformed?": "No",
        "How are files transferred?": "Direct transfer",
    },
    "Tango": {
        "Who wrote the specification?": "Nawfal ENNAJI",
        "What is the collection directory?": "/opt/cft/v3.0.1/Transfer_CFT/runtime/pub/DAILY/DONE/",
        "What is the protocol?": "SFTP",
        "What is the version?": "V1.0",
    },
}


@pytest.mark.parametrize("name,cases", REAL_CASES.items())
def test_other_real_documents_use_the_same_generic_extractor(name, cases):
    paths = [path for path in Path("doc_storage_v2").iterdir()
             if name.casefold() in path.name.casefold() and path.suffix.casefold() in {".docx", ".pdf", ".xlsx", ".csv"}]
    if not paths:
        pytest.skip(f"real {name} document unavailable")
    path = sorted(paths, key=lambda item: item.name)[-1]
    document = normalize_document(path.read_bytes(), path.name)
    for question, expected in cases.items():
        result = answer_direct(question, document)
        assert result.status == "ANSWER"
        assert result.answer == expected
        assert result.evidence_blocks
        assert all(block.file_hash == document.file_hash for block in result.evidence_blocks)
