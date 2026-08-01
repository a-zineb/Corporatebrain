"""Contract tests for the opt-in production extractive route."""

import ast
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")


def _load_helpers():
    tree = ast.parse(APP_SOURCE)
    names = {"extractive_answers_enabled", "detect_direct_factual_intent"}
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
