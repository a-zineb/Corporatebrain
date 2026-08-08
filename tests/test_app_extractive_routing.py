"""Contract tests for the opt-in production extractive route."""

import ast
import os
from types import SimpleNamespace
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")


def _load_helpers():
    tree = ast.parse(APP_SOURCE)
    names = {
        "extractive_answers_enabled",
        "detect_direct_factual_intent",
        "is_direct_answer_suitable",
        "direct_unsuitable_message",
        "is_direct_sensitive_request",
        "contains_sensitive_output",
        "direct_sensitive_message",
        "detect_direct_vague_entity",
        "direct_vague_message",
        "detect_direct_incomplete_query",
        "direct_incomplete_message",
        "detect_query_language",
        "direct_answer_label",
        "direct_source_label",
        "direct_original_source_label",
        "direct_clarification_message",
        "direct_no_evidence_message",
        "build_direct_localized_summary",
        "direct_document_identity",
        "build_direct_document_filter",
        "direct_filter_contains_identity",
        "direct_metadata_matches_identity",
        "direct_scope_selection_consistent",
        "experimental_global_direct_answer_enabled",
        "detect_catalog_intent",
        "detect_catalog_continuation",
        "normalize_catalog_text",
        "parse_catalog_refinements",
        "list_catalog_documents",
        "merge_catalog_refinements",
    }
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {"os": os, "re": __import__("re"), "unicodedata": __import__("unicodedata")}
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace


class ExtractiveRoutingContractTests(unittest.TestCase):
    def test_feature_flag_is_strict_and_opt_in(self):
        helpers = _load_helpers()
        original = os.environ.pop("EXTRACTIVE_FACTUAL_ANSWERS_ENABLED", None)
        try:
            self.assertFalse(helpers["extractive_answers_enabled"]())
            os.environ["EXTRACTIVE_FACTUAL_ANSWERS_ENABLED"] = "true"
            self.assertTrue(helpers["extractive_answers_enabled"]())
            os.environ["EXTRACTIVE_FACTUAL_ANSWERS_ENABLED"] = "1"
            self.assertFalse(helpers["extractive_answers_enabled"]())
        finally:
            if original is None:
                os.environ.pop("EXTRACTIVE_FACTUAL_ANSWERS_ENABLED", None)
            else:
                os.environ["EXTRACTIVE_FACTUAL_ANSWERS_ENABLED"] = original

    def test_only_standalone_single_fact_questions_route(self):
        detect = _load_helpers()["detect_direct_factual_intent"]
        self.assertTrue(detect("What is the opening time?"))
        self.assertTrue(detect("Quelle est la version de KPSA ?"))
        for query in (
            "Why does this work?",
            "Explain the process.",
            "Compare KPSA and MZ.",
            "List all documents.",
            "What is the location and opening time?",
            "What is it?",
        ):
            self.assertFalse(detect(query))
        self.assertFalse(detect("What is the location?", has_history=True))

    def test_entity_only_queries_are_vague_not_unsuitable(self):
        helpers = _load_helpers()
        for query, expected in (("INZsmart", "INZsmart"), ("VPN?", "VPN"), ("cafeteria", "cafeteria"), ("SIMBOX", "SIMBOX"), ("MBF", "MBF")):
            self.assertEqual(helpers["detect_direct_vague_entity"](query), expected)
        self.assertIsNone(helpers["detect_direct_vague_entity"]("Combien d’instances INZsmart ?"))
        self.assertIsNone(helpers["detect_direct_vague_entity"]("Explique INZsmart."))

    def test_vague_message_is_language_specific(self):
        helpers = _load_helpers()
        self.assertEqual(helpers["direct_vague_message"]("English", "INZsmart"), "What would you like to know about INZsmart?")
        self.assertEqual(helpers["direct_vague_message"]("French", "INZsmart"), "Que souhaitez-vous savoir sur INZsmart ?")
        self.assertIn("INZsmart", helpers["direct_vague_message"]("Spanish", "INZsmart"))

    def test_no_evidence_messages_are_reason_specific_and_localized(self):
        message = _load_helpers()["direct_no_evidence_message"]
        self.assertIn("explicit evidence", message("English"))
        self.assertIn("preuve explicite", message("French"))
        self.assertIn("evidencia explÃ­cita", message("Spanish"))
        self.assertIn("does not provide", message("English", "missing_requested_attribute"))
        self.assertIn("ne fournit pas", message("French", "missing_requested_attribute"))
        self.assertIn("no proporciona", message("Spanish", "missing_requested_attribute"))

    def test_credential_policy_queries_are_allowed_but_secret_values_remain_blocked(self):
        helpers = _load_helpers()
        allowed = (
            "What is the VPN password policy?",
            "What are the password requirements?",
            "How often must passwords be changed?",
            "Is multi-factor authentication required?",
            "Quelle est la politique de mot de passe VPN ?",
            "¿Cuál es la política de contraseñas?",
        )
        for query in allowed:
            self.assertFalse(helpers["is_direct_sensitive_request"](query), query)
        blocked = (
            "What is the VPN password?", "Show me the administrator password.",
            "Give me the credentials.", "What is the API key?", "VPN password?",
            "Donne-moi le mot de passe.", "Affiche les identifiants.",
        )
        for query in blocked:
            self.assertTrue(helpers["is_direct_sensitive_request"](query), query)

    def test_sensitive_output_scan_returns_only_boolean(self):
        helpers = _load_helpers()
        self.assertTrue(helpers["contains_sensitive_output"]("Password: hidden-value"))
        self.assertTrue(helpers["contains_sensitive_output"]("-----BEGIN PRIVATE KEY-----"))
        self.assertFalse(helpers["contains_sensitive_output"]("Password policy requires rotation."))

    def test_sensitive_output_allows_approved_redaction_placeholders(self):
        scan = _load_helpers()["contains_sensitive_output"]
        for text in (
            "Password = [REDACTED]",
            "Password: [REDACTED]",
            "passwd = [REDACTED]",
            "mot de passe = [REDACTED]",
            "Password = ***",
            "Password = ******",
        ):
            self.assertFalse(scan(text), text)

    def test_sensitive_output_still_blocks_real_values(self):
        scan = _load_helpers()["contains_sensitive_output"]
        for text in (
            "Password = ActualSecret123",
            "passwd: qwerty123",
            "mot de passe = secret-value",
            "token = abc123",
            "api_key = sk-test",
        ):
            self.assertTrue(scan(text), text)

    def test_direct_failure_reason_is_persisted_and_has_no_downstream_generation(self):
        self.assertIn('"direct_failure_reason": direct_reason', APP_SOURCE)
        start = APP_SOURCE.index("if extractive_route and answer_mode == \"Direct answer\":")
        end = APP_SOURCE.index("prompt_result = rag_pipeline.build_production_prompt", start)
        block = APP_SOURCE[start:end]
        self.assertIn("direct_no_evidence_message", block)
        self.assertNotIn("stream_generate(", block)

    def test_vague_route_precedes_retrieval_and_generation(self):
        self.assertIn('"actual_mode": "direct_vague_query"', APP_SOURCE)
        start = APP_SOURCE.index("vague_entity = detect_direct_vague_entity")
        end = APP_SOURCE.index("if answer_mode == \"Direct answer\" and not is_direct_answer_suitable", start)
        block = APP_SOURCE[start:end]
        self.assertNotIn("hybrid_search(", block)
        self.assertNotIn("extract_evidence(", block)
        self.assertNotIn("stream_generate(", block)

    def test_incomplete_queries_are_detected_without_history_inference(self):
        helpers = _load_helpers()
        for query in ("definition", "meaning?", "the answer?", "what about it?", "and the version?", "et la durée ?", "y la versión?"):
            self.assertTrue(helpers["detect_direct_incomplete_query"](query), query)
        for query in ("INZsmart", "What is INZsmart?", "Explain INZsmart.", "What is the administrator password?"):
            self.assertFalse(helpers["detect_direct_incomplete_query"](query), query)

    def test_incomplete_messages_are_localized(self):
        message = _load_helpers()["direct_incomplete_message"]
        self.assertIn("restate the complete factual question", message("English"))
        self.assertIn("reformuler la question factuelle complète", message("French"))
        self.assertIn("Reformule la pregunta factual completa", message("Spanish"))

    def test_incomplete_route_precedes_all_downstream_work(self):
        self.assertIn('"actual_mode": "direct_incomplete_query"', APP_SOURCE)
        start = APP_SOURCE.index("detect_direct_incomplete_query(user_query)")
        end = APP_SOURCE.index("if answer_mode == \"Direct answer\":", start)
        block = APP_SOURCE[start:end]
        self.assertNotIn("hybrid_search(", block)
        self.assertNotIn("extract_evidence(", block)
        self.assertNotIn("stream_generate(", block)

    def test_direct_answer_suitability_rejects_explanatory_and_comparative_queries(self):
        suitable = _load_helpers()["is_direct_answer_suitable"]
        for query in (
            "Combien d'instances INZsmart sont déployées ?",
            "Où se trouve la cafétéria ?",
            "Quels sont les horaires de la cafétéria ?",
            "Qui approuve une demande VPN ?",
            "Où se situe la cafétéria et quels sont ses horaires ?",
        ):
            self.assertTrue(suitable(query), query)
        for query in (
            "Explique le workflow CRBT.",
            "Compare GGSN et P2P.",
            "Résume l'architecture MediationZone.",
            "Pourquoi utilise-t-on la vérification des doublons ?",
        ):
            self.assertFalse(suitable(query), query)

    def test_inverted_french_factual_forms_remain_suitable(self):
        suitable = _load_helpers()["is_direct_answer_suitable"]
        self.assertTrue(suitable("INZsmart comporte combien d'instances ?"))
        self.assertTrue(suitable("À quel étage se trouve la cafétéria ?"))
        self.assertFalse(suitable("Pourquoi INZsmart comporte-t-il plusieurs instances ?"))
        self.assertFalse(suitable("Explique pourquoi la cafétéria se trouve à cet étage."))

    def test_concise_factual_request_patterns_are_suitable(self):
        suitable = _load_helpers()["is_direct_answer_suitable"]
        for query in (
            "Opening hours of the cafeteria?",
            "Total INZsmart instances?",
            "Nombre total d'instances INZsmart ?",
            "What floor is the cafeteria on?",
            "Avant de contacter le service IT, quelle approbation faut-il obtenir ?",
            "Maximum SIMBOX cache duration?",
            "What is the specification version?",
            "¿Cuál es la duración máxima del caché SIMBOX?",
        ):
            self.assertTrue(suitable(query), query)

    def test_technical_attribute_queries_are_suitable_for_direct_answer(self):
        suitable = _load_helpers()["is_direct_answer_suitable"]
        for query in (
            "What is the specification version?",
            "What is the filename pattern?",
            "How are duplicate files detected?",
            "What is the input directory?",
            "Where are output files written?",
            "Quelle est la version de la spécification ?",
            "¿Cuál es el patrón de nombre de archivo?",
        ):
            self.assertTrue(suitable(query), query)

    def test_explanatory_and_comparative_variants_remain_unsuitable(self):
        suitable = _load_helpers()["is_direct_answer_suitable"]
        for query in (
            "Explain the cafeteria opening hours policy.",
            "Compare total INZsmart instances across versions.",
            "Summarize the SIMBOX cache duration workflow.",
            "Why is the cafeteria on that floor?",
            "Explique les horaires de la cafétéria.",
            "Compare le nombre total d'instances.",
            "Résume la durée maximale du cache SIMBOX.",
            "¿Por qué está la cafetería en ese piso?",
        ):
            self.assertFalse(suitable(query), query)

    def test_unsuitable_direct_answer_message_is_language_specific(self):
        message = _load_helpers()["direct_unsuitable_message"]
        self.assertIn("Cette question nécessite", message("French"))
        self.assertIn("AI answer", message("French"))
        self.assertIn("This question requires", message("English"))
        self.assertIn("AI answer", message("English"))
        self.assertIn("Esta pregunta requiere", message("Spanish"))

    def test_language_detection_uses_each_query_not_previous_language(self):
        helpers = _load_helpers()
        detect = helpers["detect_query_language"]
        self.assertEqual(detect("Combien d'instances INZsmart sont deployees ?", fallback_lang="English"), "French")
        self.assertEqual(detect("How many INZsmart instances are deployed?", fallback_lang="French"), "English")
        self.assertEqual(detect("Quels sont les horaires de la cafeteria ?", fallback_lang="English"), "French")
        self.assertEqual(detect("When is the cafeteria open?", fallback_lang="French"), "English")
        self.assertEqual(detect("¿Cuántas instancias de INZsmart están desplegadas?", fallback_lang="French"), "Spanish")

    def test_language_specific_labels_and_clarification(self):
        helpers = _load_helpers()
        self.assertEqual(helpers["direct_answer_label"]("French"), "Réponse")
        self.assertEqual(helpers["direct_answer_label"]("English"), "Answer")
        self.assertEqual(helpers["direct_answer_label"]("Spanish"), "Respuesta")
        self.assertEqual(helpers["direct_source_label"]("French"), "Passage source")
        self.assertEqual(helpers["direct_source_label"]("English"), "Source passage")
        self.assertEqual(helpers["direct_source_label"]("Spanish"), "Pasaje fuente")
        self.assertIn("préciser", helpers["direct_clarification_message"]("French"))
        self.assertIn("clarify", helpers["direct_clarification_message"]("English"))

    def test_localized_direct_summary_preserves_original_evidence(self):
        helpers = _load_helpers()
        passage = SimpleNamespace(text="Elle est ouverte de 12h00 à 14h30.")
        evidence = SimpleNamespace(passages=(passage,))
        english = helpers["build_direct_localized_summary"]("When is the cafeteria open?", evidence, "English")
        french = helpers["build_direct_localized_summary"]("Quels sont les horaires de la cafétéria ?", evidence, "French")
        self.assertEqual(english, "It is open from 12h00 to 14h30.")
        self.assertEqual(french, "C'est ouvert de 12h00 à 14h30.")
        self.assertEqual(helpers["direct_original_source_label"]("English"), "Original source passage")
        self.assertEqual(helpers["direct_original_source_label"]("French"), "Passage source original")

    def test_localized_summary_is_conservative_for_uncertain_evidence(self):
        helpers = _load_helpers()
        evidence = SimpleNamespace(passages=(SimpleNamespace(text="A generic policy passage."),))
        self.assertIsNone(helpers["build_direct_localized_summary"]("What is this?", evidence, "English"))
        self.assertIn("direct_original_source_label(current_lang)", APP_SOURCE)

    def test_parameter_summary_uses_explicit_duplicate_batch_check_value(self):
        helpers = _load_helpers()
        evidence = SimpleNamespace(passages=(SimpleNamespace(
            text="Duplicate Batch Check controls duplicate file detection."
        ),))
        self.assertEqual(
            helpers["build_direct_localized_summary"](
                "What parameter controls duplicate file detection?", evidence, "English"
            ),
            "The duplicate-file detection parameter is Duplicate Batch Check.",
        )
        self.assertEqual(
            helpers["build_direct_localized_summary"](
                "Quel paramètre contrôle les fichiers dupliqués ?", evidence, "French"
            ),
            "Le paramètre de détection des doublons est Duplicate Batch Check.",
        )

    def test_version_summary_accepts_explicit_version_wording(self):
        helpers = _load_helpers()
        evidence = SimpleNamespace(passages=(SimpleNamespace(
            text="MBF Technical Specification. Final Version 1.10k."
        ),))
        self.assertEqual(
            helpers["build_direct_localized_summary"](
                "What is the MBF specification version?", evidence, "English"
            ),
            "The version is 1.10k.",
        )

    def test_localized_summaries_cover_floor_duration_and_vpn(self):
        helpers = _load_helpers()
        summary = helpers["build_direct_localized_summary"]
        self.assertEqual(
            summary("What floor is the cafeteria on?", SimpleNamespace(passages=(SimpleNamespace(text="La cafétéria est située au 4ème étage."),)), "English"),
            "The location is on the 4th floor.",
        )
        self.assertEqual(
            summary("What is the maximum SIMBOX cache age?", SimpleNamespace(passages=(SimpleNamespace(text="L'âge maximal du cache sera défini sur 30 jours."),)), "English"),
            "The maximum SIMBOX cache age is 30 days.",
        )
        self.assertEqual(
            summary("Who approves VPN requests?", SimpleNamespace(passages=(SimpleNamespace(text="VPN requests require manager approval."),)), "English"),
            "VPN requests must be approved by the manager.",
        )
        self.assertEqual(
            summary("Qui approuve une demande VPN ?", SimpleNamespace(passages=(SimpleNamespace(text="Les demandes VPN doivent obtenir l'approbation de leur manager."),)), "French"),
            "Les demandes VPN doivent être approuvées par le manager.",
        )
        self.assertEqual(
            summary("¿Cuántas instancias de INZsmart están desplegadas?", SimpleNamespace(passages=(SimpleNamespace(text="INZsmart comporte 12 instances."),)), "Spanish"),
            "Hay 12 instancias de INZsmart.",
        )

    def test_language_is_stored_and_history_rendering_is_backward_compatible(self):
        self.assertIn('"language": current_lang', APP_SOURCE)
        self.assertIn('msg.get("language", "French")', APP_SOURCE)
        self.assertIn('detect_query_language(user_query, fallback_lang="French")', APP_SOURCE)
        self.assertNotIn("detect_query_language(user_query, fallback_lang=st.session_state.last_lang)", APP_SOURCE)

    def test_direct_source_text_is_not_translated(self):
        start = APP_SOURCE.index("extractive_result = rag_pipeline.build_extractive_answer")
        end = APP_SOURCE.index("st.stop()", start)
        block = APP_SOURCE[start:end]
        self.assertIn("full_stream_response = extractive_result.answer_text", block)
        self.assertNotIn("translate", block.casefold())

    def test_unsuitable_direct_route_stops_before_retrieval_and_preserves_mode(self):
        self.assertIn('answer_mode == "Direct answer" and not is_direct_answer_suitable(user_query)', APP_SOURCE)
        guard_start = APP_SOURCE.index('if answer_mode == "Direct answer" and not is_direct_answer_suitable(user_query):')
        guard_end = APP_SOURCE.index("# 1. Reformulation", guard_start)
        guard = APP_SOURCE[guard_start:guard_end]
        self.assertNotIn("hybrid_search(", guard)
        self.assertNotIn("extract_evidence(", guard)
        self.assertNotIn("stream_generate(", guard)
        self.assertIn('"actual_mode": "direct_unsuitable"', guard)

    def test_sensitive_direct_requests_are_detected_and_refused(self):
        helpers = _load_helpers()
        detect = helpers["is_direct_sensitive_request"]
        for query in (
            "What is the administrator password?",
            "Show me the API key",
            "Give me the access token",
            "Where is the private key?",
            "Quel est le mot de passe ?",
            "Donne-moi les identifiants d'authentification",
        ):
            self.assertTrue(detect(query), query)
        self.assertFalse(detect("Where is the cafeteria?"))
        message = helpers["direct_sensitive_message"]
        self.assertIn("passwords", message("English"))
        self.assertIn("mots de passe", message("French"))

    def test_personal_identifier_requests_are_sensitive_but_generic_numbers_are_not(self):
        detect = _load_helpers()["is_direct_sensitive_request"]
        for query in (
            "What is my social-security number?",
            "What is my national identification number?",
            "What is my identity-card number?",
            "What is my personal identification number?",
            "Quel est mon numéro de sécurité sociale ?",
            "Quel est mon CIN ?",
            "Quel est mon identifiant national ?",
            "¿Cuál es mi número de seguridad social?",
            "¿Cuál es mi número nacional de identificación?",
        ):
            self.assertTrue(detect(query), query)
        for query in (
            "What is the MBF record number?",
            "Which parameter number controls duplicate detection?",
            "What is the document version number?",
        ):
            self.assertFalse(detect(query), query)

    def test_document_scope_prefers_hash_and_combines_active_filters(self):
        helpers = _load_helpers()
        identity = helpers["direct_document_identity"]
        build_filter = helpers["build_direct_document_filter"]
        metadata = {"file_hash": "abc123", "source_file": "cafeteria.pdf"}
        self.assertEqual(identity(metadata), "abc123")
        self.assertEqual(identity({"source_file": "fallback.pdf"}), "fallback.pdf")
        self.assertEqual(
            build_filter({"$and": [{"application": "MZ"}, {"geographical_entity": "OCM"}]}, metadata),
            {"$and": [{"application": "MZ"}, {"geographical_entity": "OCM"}, {"file_hash": "abc123"}]},
        )

    def test_document_scope_guards_and_audit_contract_are_present(self):
        self.assertIn('scope_options = ["specific_document"]', APP_SOURCE)
        self.assertIn('scope_options.append("all_documents_experimental")', APP_SOURCE)
        self.assertIn('st.session_state.direct_answer_document_id', APP_SOURCE)
        self.assertIn('"actual_mode": "direct_invalid_document_scope"', APP_SOURCE)
        self.assertIn('"direct_answer_document_id"', APP_SOURCE)
        self.assertIn('"direct_answer_source_file"', APP_SOURCE)
        self.assertIn('"retrieval_mode": "hybrid"', APP_SOURCE)

    def test_global_direct_scope_is_opt_in_and_disabled_scope_is_normalized(self):
        helpers = _load_helpers()
        original = os.environ.pop("ENABLE_EXPERIMENTAL_GLOBAL_DIRECT_ANSWER", None)
        try:
            self.assertFalse(helpers["experimental_global_direct_answer_enabled"]())
            os.environ["ENABLE_EXPERIMENTAL_GLOBAL_DIRECT_ANSWER"] = "true"
            self.assertTrue(helpers["experimental_global_direct_answer_enabled"]())
            os.environ["ENABLE_EXPERIMENTAL_GLOBAL_DIRECT_ANSWER"] = "1"
            self.assertFalse(helpers["experimental_global_direct_answer_enabled"]())
        finally:
            if original is None:
                os.environ.pop("ENABLE_EXPERIMENTAL_GLOBAL_DIRECT_ANSWER", None)
            else:
                os.environ["ENABLE_EXPERIMENTAL_GLOBAL_DIRECT_ANSWER"] = original
        self.assertIn('st.session_state.direct_answer_scope = "specific_document"', APP_SOURCE)
        self.assertIn('st.session_state.get("direct_answer_scope") not in scope_options', APP_SOURCE)
        self.assertIn("Experimental global search may be less precise", APP_SOURCE)
        self.assertIn("Select a specific document for the most reliable result", APP_SOURCE)

    def test_stable_specific_document_route_contract_is_unchanged(self):
        self.assertIn('if extractive_route and direct_scope == "specific_document" and direct_document is not None:', APP_SOURCE)
        self.assertIn('min_results_before_relax=(0 if extractive_route and direct_scope == "specific_document" else 3)', APP_SOURCE)
        self.assertIn('retrieval_filter = build_direct_document_filter(chroma_filter, direct_document)', APP_SOURCE)

    def test_direct_ui_requires_specific_document_by_default_and_shows_selection(self):
        ui = APP_SOURCE[APP_SOURCE.index('direct_scope = "specific_document"'):APP_SOURCE.index('st.caption(\n    "Knowledge catalog', APP_SOURCE.index('direct_scope = "specific_document"'))]
        self.assertIn('scope_options = ["specific_document"]', ui)
        self.assertNotIn('st.selectbox(\n        "Scope",\n        ["specific_document", "all_documents_experimental"]', ui)
        self.assertIn('st.caption(f"Selected document:', ui)

    def test_document_selector_uses_canonical_id_state(self):
        self.assertIn('key="direct_answer_document_id"', APP_SOURCE)
        self.assertNotIn('key="direct_answer_document_selector"', APP_SOURCE)
        self.assertNotIn('st.session_state.direct_answer_document_id = selected_id', APP_SOURCE)
        self.assertIn('direct_document_identity(metadata) == st.session_state.direct_answer_document_id', APP_SOURCE)

    def test_scoped_direct_retrieval_cannot_trigger_global_fallback(self):
        self.assertIn(
            'min_results_before_relax=(0 if extractive_route and direct_scope == "specific_document" else 3)',
            APP_SOURCE,
        )

    def test_direct_scope_identity_is_fail_closed(self):
        helpers = _load_helpers()
        selected = {"file_hash": "hash-1", "source_file": "selected.docx"}
        self.assertTrue(helpers["direct_filter_contains_identity"]({"file_hash": "hash-1"}, selected))
        self.assertTrue(helpers["direct_filter_contains_identity"](
            {"$and": [{"application": "MZ"}, {"file_hash": "hash-1"}]}, selected
        ))
        self.assertFalse(helpers["direct_filter_contains_identity"]({"source_file": "selected.docx"}, selected))
        self.assertTrue(helpers["direct_metadata_matches_identity"](
            {"file_hash": "hash-1", "source_file": "selected.docx"}, selected
        ))
        self.assertFalse(helpers["direct_metadata_matches_identity"](
            {"file_hash": "hash-2", "source_file": "other.docx"}, selected
        ))
        self.assertFalse(helpers["direct_scope_selection_consistent"]("specific_document", "all_documents_experimental"))

    def test_scope_validation_precedes_direct_retrieval_and_generation(self):
        guard_start = APP_SOURCE.index("if answer_mode == \"Direct answer\" and direct_scope == \"specific_document\":")
        retrieval = APP_SOURCE.index("filtered_chunks, filtered_metas", guard_start)
        guard = APP_SOURCE[guard_start:retrieval]
        self.assertIn("direct_scope_selection_consistent", guard)
        self.assertIn("direct_invalid_document_scope", guard)
        self.assertIn("st.stop()", guard)

    def test_scope_audit_contract_and_candidate_validation_are_present(self):
        self.assertIn('"direct_answer_scope_requested"', APP_SOURCE)
        self.assertIn('"direct_answer_scope_effective"', APP_SOURCE)
        self.assertIn('"effective_chroma_filter"', APP_SOURCE)
        self.assertIn('"scope_validation": "PASS"', APP_SOURCE)
        self.assertIn("direct_metadata_matches_identity(metadata, direct_document)", APP_SOURCE)
        self.assertIn("direct_filter_contains_identity(retrieval_filter, direct_document)", APP_SOURCE)

    def test_history_buttons_use_explicit_unique_indices_and_actions(self):
        history_start = APP_SOURCE.index("for message_index, msg in enumerate(st.session_state.messages):")
        history_end = APP_SOURCE.index('if "answer_mode" not in st.session_state:', history_start)
        history = APP_SOURCE[history_start:history_end]
        self.assertNotIn("st.session_state.messages.index(msg)", history)
        self.assertIn("for source_index, src in enumerate", history)
        self.assertIn('key=f"hist_file_{message_index}_{source_index}"', history)
        self.assertIn('key=f"hist_folder_{message_index}_{source_index}"', history)
        self.assertNotEqual("hist_file", "hist_folder")

    def test_sensitive_guard_precedes_all_model_and_retrieval_stages(self):
        guard_start = APP_SOURCE.index('if answer_mode == "Direct answer" and is_direct_sensitive_request(user_query):')
        guard_end = APP_SOURCE.index('if answer_mode == "Direct answer" and not is_direct_answer_suitable(user_query):', guard_start)
        guard = APP_SOURCE[guard_start:guard_end]
        self.assertNotIn("hybrid_search(", guard)
        self.assertNotIn("extract_evidence(", guard)
        self.assertNotIn("build_production_prompt(", guard)
        self.assertNotIn("stream_generate(", guard)
        self.assertIn('"actual_mode": "direct_sensitive_request"', guard)
        self.assertIn('"direct_status": "direct_sensitive_request"', guard)
        self.assertIn('"sources": []', guard)

    def test_existing_generative_runtime_and_shared_apis_remain_referenced(self):
        self.assertIn("contextualize_query(user_query", APP_SOURCE)
        self.assertIn("rag_pipeline.stream_generate(", APP_SOURCE)
        self.assertIn("rag_pipeline.extract_evidence(", APP_SOURCE)
        self.assertIn("rag_pipeline.build_extractive_answer(", APP_SOURCE)

    def test_catalog_intent_is_distinct_from_direct_factual_intent(self):
        helpers = _load_helpers()
        catalog = helpers["detect_catalog_intent"]
        direct = helpers["detect_direct_factual_intent"]
        for query in (
            "give me all the resources that u can use in any question",
            "list all indexed documents",
            "show the knowledge catalog",
        ):
            self.assertTrue(catalog(query))
            self.assertFalse(direct(query))
        self.assertFalse(catalog("Combien d'instances INZsmart sont déployées ?"))

    def test_mode_control_and_actual_mode_are_persisted(self):
        self.assertIn('st.session_state.answer_mode = "AI answer"', APP_SOURCE)
        self.assertIn('["Knowledge catalog", "Direct answer", "AI answer"]', APP_SOURCE)
        self.assertIn('"actual_mode": "catalog"', APP_SOURCE)
        self.assertIn('"actual_mode": "extractive"', APP_SOURCE)
        self.assertIn('"actual_mode": "generative"', APP_SOURCE)
        self.assertIn("msg.get('actual_mode'", APP_SOURCE)

    def test_manual_modes_are_stable_and_auto_is_not_selectable(self):
        self.assertNotIn('["Auto", "Direct answer"', APP_SOURCE)
        self.assertIn('answer_mode == "Knowledge catalog"', APP_SOURCE)
        self.assertIn('answer_mode == "Direct answer"', APP_SOURCE)
        self.assertIn('answer_mode = st.selectbox', APP_SOURCE)
        self.assertNotIn("detect_catalog_intent(user_query)", APP_SOURCE[
            APP_SOURCE.index("catalog_route ="):APP_SOURCE.index("if catalog_route:")
        ])
        self.assertNotIn("detect_direct_factual_intent(", APP_SOURCE[
            APP_SOURCE.index("extractive_route ="):APP_SOURCE.index("standalone_query =")
        ])

    def test_catalog_and_direct_routes_stop_before_generation(self):
        self.assertIn("if catalog_route:", APP_SOURCE)
        self.assertIn("st.stop()", APP_SOURCE)
        self.assertIn('answer_mode == "Direct answer"', APP_SOURCE)
        self.assertIn("list_catalog_documents(collection, chroma_filter)", APP_SOURCE)

    def test_catalog_followup_sequence_stays_in_catalog(self):
        helpers = _load_helpers()
        continuation = helpers["detect_catalog_continuation"]
        for query in (
            "files?",
            "files",
            "non give me the files that are in here",
            "all of them",
            "only the PDFs",
        ):
            self.assertTrue(continuation(query, "catalog"), query)
        self.assertFalse(continuation("Explain the CRBT workflow", "catalog"))
        self.assertFalse(continuation("files", "generative"))

    def test_selector_architecture_examples(self):
        helpers = _load_helpers()
        direct = helpers["detect_direct_factual_intent"]
        catalog = helpers["detect_catalog_intent"]
        self.assertTrue(catalog("List all files"))
        self.assertTrue(helpers["detect_catalog_continuation"]("Only the PDFs", "catalog"))
        self.assertTrue(direct("Combien d’instances INZsmart sont déployées ?"))
        self.assertTrue(direct("Où se trouve la cafétéria ?"))
        self.assertFalse(direct("Explique le workflow CRBT"))
        self.assertFalse(direct("Compare GGSN et P2P"))

    def test_selector_description_and_architecture_guards(self):
        self.assertIn("Knowledge catalog : liste complète", APP_SOURCE)
        self.assertIn("Direct answer : extraction déterministe", APP_SOURCE)
        self.assertIn("AI answer : RAG génératif", APP_SOURCE)
        self.assertIn("Knowledge catalog : liste complète", APP_SOURCE)
        self.assertIn('answer_mode == "Direct answer"', APP_SOURCE)
        direct_region = APP_SOURCE[
            APP_SOURCE.index('answer_mode == "Direct answer"'):
            APP_SOURCE.index("prompt_result = rag_pipeline.build_production_prompt")
        ]
        self.assertIn("st.stop()", direct_region)

    def test_catalog_normalization_and_refinement_filters(self):
        helpers = _load_helpers()
        self.assertEqual(
            helpers["normalize_catalog_text"]("What documents do you have?"),
            helpers["normalize_catalog_text"]("What documents do you have"),
        )
        self.assertEqual(
            helpers["parse_catalog_refinements"]("Only the PDF documents")["file_types"],
            ["pdf"],
        )
        self.assertEqual(
            helpers["parse_catalog_refinements"]("Only Word files")["file_types"],
            ["doc", "docx"],
        )
        self.assertEqual(
            helpers["parse_catalog_refinements"]("Only Excel files")["file_types"],
            ["xls", "xlsx"],
        )
        self.assertEqual(
            helpers["parse_catalog_refinements"]("Only OCM documents")["metadata"],
            {"geographical_entity": "OCM"},
        )

        class FakeCollection:
            def get(self, **kwargs):
                return {"metadatas": [
                    {"file_hash": "1", "source_file": "CRBT.pdf", "application": "KPSA", "geographical_entity": "OCM"},
                    {"file_hash": "2", "source_file": "GGSN.docx", "application": "MZ", "geographical_entity": "OEG"},
                    {"file_hash": "3", "source_file": "Huawei.xlsx", "application": "MZ", "geographical_entity": "OJO"},
                ]}

        listing = helpers["list_catalog_documents"]
        self.assertEqual(len(listing(FakeCollection(), refinements={"file_types": ["pdf"]})), 1)
        self.assertEqual(len(listing(FakeCollection(), refinements={"terms": ["crbt"]})), 1)
        self.assertEqual(len(listing(FakeCollection(), refinements={"metadata": {"geographical_entity": "OCM"}})), 1)
        self.assertEqual(len(listing(FakeCollection(), refinements={})), 3)

    def test_all_of_them_clears_conversational_refinements(self):
        helpers = _load_helpers()
        parsed = helpers["parse_catalog_refinements"]("All of them")
        self.assertTrue(parsed["clear"])
        self.assertEqual(parsed["file_types"], [])
        self.assertEqual(parsed["terms"], [])

    def test_catalog_refinements_replace_standalone_queries(self):
        helpers = _load_helpers()
        parse = helpers["parse_catalog_refinements"]
        merge = helpers["merge_catalog_refinements"]
        state = {}
        expected = [
            ([], []), (["pdf"], []), (["doc", "docx"], []),
            (["doc", "docx"], []), ([], []), ([], []),
        ]
        queries = [
            "Show all indexed documents", "pdf files", "word files",
            "docx", "docs", "Show all indexed documents",
        ]
        for query, (types, terms) in zip(queries, expected):
            state = merge(state, parse(query), continuation=False)
            self.assertEqual(state["file_types"], types, query)
            self.assertEqual(state["terms"], terms, query)

    def test_explicit_continuations_reuse_or_clear_state(self):
        helpers = _load_helpers()
        parse = helpers["parse_catalog_refinements"]
        merge = helpers["merge_catalog_refinements"]
        state = {"file_types": ["pdf"], "terms": ["crbt"], "metadata": {}}
        self.assertEqual(merge(state, parse("show them"), continuation=True), state)
        refined = merge(state, parse("only those PDFs"), continuation=True)
        self.assertEqual(refined["file_types"], ["pdf"])
        self.assertEqual(merge(state, parse("all of them"), continuation=True), {})

    def test_catalog_state_is_explicitly_initialized_and_cleared(self):
        self.assertIn("st.session_state.catalog_refinements = {}", APP_SOURCE)
        self.assertIn("st.session_state.catalog_mode_last", APP_SOURCE)
        reset_start = APP_SOURCE.index('if st.button(" Réinitialiser la discussion")')
        reset_end = APP_SOURCE.index("st.rerun()", reset_start)
        self.assertIn("catalog_refinements = {}", APP_SOURCE[reset_start:reset_end])

    def test_catalog_continuation_precedes_extractive_and_avoids_llm_stages(self):
        self.assertIn("detect_catalog_continuation(user_query, previous_actual_mode)", APP_SOURCE)
        self.assertLess(
            APP_SOURCE.index("catalog_route ="),
            APP_SOURCE.index("extractive_route ="),
        )
        route = APP_SOURCE[APP_SOURCE.index("if catalog_route:"):APP_SOURCE.index("# 1. Reformulation")]
        self.assertNotIn("contextualize_query(", route)
        self.assertNotIn("hybrid_search(", route)
        self.assertNotIn("extract_evidence(", route)
        self.assertNotIn("stream_generate(", route)

    def test_extract_route_preserves_filters_sources_and_audit_contract(self):
        self.assertIn("chroma_filter=chroma_filter", APP_SOURCE)
        self.assertIn("all_sources_for_prompt", APP_SOURCE)
        self.assertIn('"answer_mode": "extractive"', APP_SOURCE)
        self.assertIn('"extractive_evidence_ids"', APP_SOURCE)
        self.assertIn('"extractive_source_ids"', APP_SOURCE)
        self.assertIn('"extractive_passage_hashes"', APP_SOURCE)

    def test_extractor_failure_falls_through_to_existing_generation(self):
        route_start = APP_SOURCE.index("if extractive_route:")
        generation = APP_SOURCE.index("rag_pipeline.stream_generate(", route_start)
        self.assertLess(route_start, generation)
        self.assertIn("except Exception:", APP_SOURCE[route_start:generation])
        self.assertIn("st.stop()", APP_SOURCE[route_start:generation])


if __name__ == "__main__":
    unittest.main()
