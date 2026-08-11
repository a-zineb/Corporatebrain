import random
from pathlib import Path

import pytest

from canonical_rag import FastDirectAnswerEngine, answer_direct
from document_normalizer import normalize_document


TANGO_CASES = {
    "Who wrote the Tango specification?": "Nawfal ENNAJI",
    "What is the Tango specification version?": "V1.0",
    "What does Tango mean?": "Plateforme qui permet aux utilisateurs d’accéder à des services bancaires et financiers à distance",
    "Tango?": "Plateforme qui permet aux utilisateurs d’accéder à des services bancaires et financiers à distance",
    "What is the Tango collection directory?": "/opt/cft/v3.0.1/Transfer_CFT/runtime/pub/DAILY/DONE/",
    "What is the Tango post collection action?": "à vérifier par OCM",
    "How often are Tango files collected?": "j+1",
    "What are the Tango filename patterns?": "Filename patterns:\n- OCM_APGL_report_yyyymmdd.zip\n- OCM_DWH_report_yyyymmdd.zip",
    "What is the Tango CDR format?": "Formats:\n- Collection input specification: archive (.zip)\n- DWH: Brut\n- BIG DATA: Brut\n- FTP_CRA: Brut",
    "CDR Format": "Formats:\n- Collection input specification: archive (.zip)\n- DWH: Brut\n- BIG DATA: Brut\n- FTP_CRA: Brut",
    "input format?": "N/A",
    "connection protocol": "SFTP",
    "What is the DWH host?": "172.21.75.61",
    "What is the BIG DATA host?": "172.26.60.12",
    "What is the BIG DATA output directory?": "/data/input/mz/om/",
    "What is the BIG DATA username?": "mz_user",
    "What is the FTP_CRA username?": "med_user",
    "What protocol is used to distribute Tango files to FTP_CRA?": "SFTP",
    "What is the output filename for BIG DATA?": "meme filename qu'en entrée",
    "What table contains the Tango distribution flags?": "MZ_PARAM",
    "What parameter controls Tango distribution to DWH?": "TANGO_TO_DWH",
    "What parameter controls Tango distribution to SVR_CRA?": "TANGO_TO_SVRCRA",
    "What parameter controls Tango distribution to Big Data?": "TANGO_TO_BigData",
    "How many archive modes are planned?": "2",
    "capacity": "(nombre de CDRs/fichiers par jour)",
    "correlation": "N/A",
    "How are duplicate Tango files detected?": "MediationZone® effectuera une vérification des doublons au niveau des fichiers pour déterminer si le même fichier a déjà été traité. Cet agent utilise le calcul du contrôle de redondance cyclique (CRC) sur tout le contenu du fichier collecté. L'âge maximal du cache sera défini sur 30 jours.",
}


def _tango():
    paths = [path for path in Path("doc_storage_v2").iterdir()
             if "Tango" in path.name and path.suffix.casefold() == ".docx"]
    if not paths:
        pytest.skip("real Tango document unavailable")
    path = paths[0]
    return normalize_document(path.read_bytes(), path.name)


def test_tango_fact_store_compiles_atomic_row_aware_facts():
    document = _tango()
    engine = FastDirectAnswerEngine()
    engine.prepare(document)
    store = engine.fact_stores[document.file_hash]
    assert len(store.facts) >= 90
    assert {"parameter", "username", "capacity", "correlation", "post_action"} <= set(store.categories)
    big_data = [fact for fact in store.facts if fact.entity.casefold() == "big data"]
    assert any(fact.relation == "username" and fact.value == "mz_user" for fact in big_data)


def test_tango_supported_facts_and_query_order_are_deterministic():
    document = _tango()
    baseline = {question: answer_direct(question, document) for question in TANGO_CASES}
    assert all(result.status == "ANSWER" for result in baseline.values())
    assert {question: result.answer for question, result in baseline.items()} == TANGO_CASES
    questions = list(TANGO_CASES)
    shuffled = list(questions)
    random.Random(7).shuffle(shuffled)
    for order in (questions, list(reversed(questions)), shuffled):
        assert {question: answer_direct(question, document) for question in order} == baseline


def test_unqualified_login_returns_all_contextual_values():
    result = answer_direct("login?", _tango())
    assert result.status == "ANSWER"
    assert result.result_type == "MULTI_VALUE"
    assert all(value in result.answer for value in ("DWH", "BIG DATA", "FTP_CRA", "mz_user", "med_user"))
