import ast
from pathlib import Path


APP_SOURCE = Path(__file__).parents[1].joinpath("app.py").read_text(encoding="utf-8")


def test_streamlit_uses_active_canonical_context_before_legacy_retrieval():
    route = APP_SOURCE.index("# Canonical selected-document route")
    legacy = APP_SOURCE.index("# 1. Reformulation de la question", route)
    assert route < legacy
    assert "active_document_service.active" in APP_SOURCE[route:legacy]
    assert "fast_direct_engine.query" in APP_SOURCE[route:legacy]
    assert "ollama.chat" not in APP_SOURCE[route:legacy].casefold()
    assert "stream_generate" not in APP_SOURCE[route:legacy]


def test_ai_synthesis_current_document_scope_filters_by_active_hash():
    assert 'ai_scope = "current_active_document"' in APP_SOURCE
    assert 'hash_filter = {"file_hash": active_for_ai.file_hash}' in APP_SOURCE
    assert '"all_documents_explicit"' in APP_SOURCE


def test_fast_catalog_route_uses_precomputed_index_without_generation():
    start = APP_SOURCE.index("if catalog_route:")
    end = APP_SOURCE.index("st.stop()", start)
    route = APP_SOURCE[start:end]
    assert "fast_catalog_rows" in route
    assert "ollama" not in route.casefold()
    assert "embedding" not in route.casefold()


def test_french_no_evidence_message_is_valid_utf8_without_mojibake():
    expected = "Je n’ai trouvé aucune preuve explicite répondant à cette question dans le document sélectionné."
    tree = ast.parse(APP_SOURCE)
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "CANONICAL_NO_EVIDENCE_FR" for target in node.targets)
    )
    assert ast.literal_eval(assignment.value) == expected
    canonical_section = APP_SOURCE[APP_SOURCE.index("CANONICAL_NO_EVIDENCE_FR"):APP_SOURCE.index("def open_local_file")]
    assert "Ã" not in canonical_section
