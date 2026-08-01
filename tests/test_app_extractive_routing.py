"""Contract tests for the opt-in production extractive route."""

import ast
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")


def _load_helpers():
    tree = ast.parse(APP_SOURCE)
    names = {
        "extractive_answers_enabled",
        "detect_direct_factual_intent",
        "detect_catalog_intent",
        "detect_catalog_continuation",
    }
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = {"os": os, "re": __import__("re")}
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
        self.assertIn('st.session_state.answer_mode = "Auto"', APP_SOURCE)
        self.assertIn('["Auto", "Direct answer", "AI answer", "Knowledge catalog"]', APP_SOURCE)
        self.assertIn('"actual_mode": "catalog"', APP_SOURCE)
        self.assertIn('"actual_mode": "extractive"', APP_SOURCE)
        self.assertIn('"actual_mode": "generative"', APP_SOURCE)
        self.assertIn("msg.get('actual_mode'", APP_SOURCE)

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
