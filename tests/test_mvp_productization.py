from __future__ import annotations

import canonical_rag
from document_normalizer import CanonicalBlock, CanonicalDocument, normalize_document
from mvp_services import (
    LocalMetrics,
    PreparedDocumentRegistry,
    SourceTarget,
    detect_query_language,
    suggestion_candidates,
)
import ui_components


def csv_document(name: str, host: str) -> CanonicalDocument:
    return normalize_document(
        f"System,Host,Directory\nBI,{host},/data/bi\nDWH,10.0.0.2,/data/dwh\n".encode(),
        name,
    )


def test_language_is_detected_for_each_message_and_utf8_is_clean():
    assert detect_query_language("What is the BI host?") == "English"
    assert detect_query_language("Quel est le répertoire du BI ?") == "French"
    assert detect_query_language("œuvre, où, été, ça, à l’entrée") == "French"
    document = csv_document("référence.csv", "10.0.0.1")
    assert canonical_rag.answer_direct("What is the BI host?", document).query_language == "English"
    assert canonical_rag.answer_direct("Quel est le host BI ?", document).query_language == "French"
    assert "J’ai trouvé" in ui_components.LABELS["French"]["multiple"]
    assert "Ã" not in " ".join(ui_components.LABELS["French"].values())


def test_prepare_is_idempotent_for_hash_and_schema_version():
    data = b"Name,Value\nHost,10.0.0.1\n"
    registry = PreparedDocumentRegistry()
    first = registry.prepare(data, "one.csv")
    second = registry.prepare(data, "renamed.csv")
    assert first.state == "READY"
    assert second.cached is True
    assert second.document is first.document
    assert len(registry.documents) == 1


def test_global_find_searches_every_document_without_changing_active_selection():
    first = csv_document("first.csv", "10.0.0.1")
    second = csv_document("second.csv", "10.0.0.9")
    active = canonical_rag.ActiveDocumentService()
    selected = active.select(first)
    registry = PreparedDocumentRegistry()
    registry.add(first)
    registry.add(second)
    hits = registry.global_search("host")
    assert {hit.document.source_file for hit in hits} == {"first.csv", "second.csv"}
    assert active.active is selected
    assert active.active.file_hash == first.file_hash
    assert all(hit.display_title and hit.display_value for hit in hits)


def test_global_result_humanizes_internal_spreadsheet_columns():
    document = normalize_document(
        b"Connection Protocol,SFTP,FTP\nSystem A,SFTP,FTP\n", "protocols.csv"
    )
    registry = PreparedDocumentRegistry()
    registry.add(document)
    hits = registry.global_search("protocol")
    assert hits
    visible = " ".join(
        f"{hit.display_title} {hit.relation} {hit.display_value} {hit.preview}" for hit in hits
    )
    assert "Column 2 =" not in visible
    assert "Column 3 =" not in visible


def test_source_target_preserves_exact_provenance_for_supported_locations():
    block = CanonicalBlock(
        "b", "Host = 10.0.0.1", "key_value", "sample.pdf", "h",
        section="Distribution", page=7, sheet=None, table_index=3, row_index=14,
        paragraph_index=8, metadata={"bbox": [10, 20, 100, 40], "page_end": 8},
    )
    target = SourceTarget.from_block(block, "pdf")
    assert target.page == 7 and target.page_end == 8
    assert target.section == "Distribution"
    assert target.table_index == 3 and target.row_index == 14
    assert target.paragraph_index == 8
    assert target.bbox == (10.0, 20.0, 100.0, 40.0)
    assert "Page 7" in target.location_label and "Row: 14" in target.location_label


def test_suggestions_are_only_derived_from_document_labels_and_titles():
    document = csv_document("facts.csv", "10.0.0.1")
    suggestions = suggestion_candidates("hst", document)
    assert "Host" in suggestions
    known = {"System", "Host", "Directory"}
    assert set(suggestions) <= known


def test_metrics_report_latency_and_answer_types_without_external_telemetry():
    document = csv_document("facts.csv", "10.0.0.1")
    result = canonical_rag.answer_direct("host?", document)
    metrics = LocalMetrics()
    metrics.record_answer(result, 12.0)
    metrics.record_answer(result, 20.0)
    snapshot = metrics.snapshot()
    assert snapshot["direct_p50_ms"] == 16.0
    assert snapshot["direct_p95_ms"] == 20.0
    assert snapshot["multi_value_count"] == 2


def test_ui_css_does_not_define_a_blank_dark_answer_container():
    css = ui_components.DESIGN_CSS.casefold()
    assert "[data-testid=\"stchatmessage\"]" in css
    assert "background:#000" not in css
    assert "background:black" not in css


def test_only_find_me_owns_a_streamlit_dialog():
    source = open(ui_components.__file__, encoding="utf-8").read()
    assert source.count("@st.dialog") == 1
    assert "def render_find_me" in source
    viewer_prefix = source.split("def render_source_viewer", 1)[0].rsplit("\n", 2)[-2:]
    assert not any("@st.dialog" in line for line in viewer_prefix)
