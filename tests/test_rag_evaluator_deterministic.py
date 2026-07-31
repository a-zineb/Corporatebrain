"""Deterministic evaluator checks using only shared-runtime test doubles."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import rag_evaluator
import rag_pipeline


class FakeVector:
    def tolist(self):
        return [0.5, 0.5]


class FakeEmbeddingModel:
    def encode(self, _query):
        return FakeVector()


class FakeCollection:
    def __init__(self, documents, metadatas):
        self.documents = documents
        self.metadatas = metadatas
        self.calls = []

    def get(self, *, include):
        self.include = include
        return {"documents": self.documents, "metadatas": self.metadatas}

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "ids": [["chunk-1"]],
            "documents": [[self.documents[0]]],
            "metadatas": [[self.metadatas[0]]],
            "distances": [[0.1]],
        }


class FakeGenerator:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [{"message": {"content": self.response}}]
        return {"message": {"content": kwargs["messages"][0]["content"].split("QUESTION :")[-1].split("QUESTION REFORMULÉE")[0].strip()}}


def case_for(content_hash, *, answerability="answerable", response_mode="answer"):
    relevance = [] if answerability == "unanswerable" else [{
        "content_sha256": content_hash,
        "label": 2,
    }]
    citations = [] if answerability == "unanswerable" else [{"content_sha256": content_hash}]
    return {
        "id": "case-1",
        "category": "direct",
        "question": "alpha",
        "conversation": [],
        "language": "fr",
        "metadata_filter": {},
        "relevance": relevance,
        "acceptable_citations": citations,
        "answerability": answerability,
        "expected_behavior": {"mode": response_mode, "source_display": "none" if response_mode != "answer" else "expected"},
    }


class DeterministicEvaluatorTests(unittest.TestCase):
    """Verify runner orchestration, formulas, and report formats without Ollama."""

    def setUp(self):
        self.documents = ["alpha policy", "other policy"]
        self.metadatas = [{"source_file": "policy.pdf", "location": "Page 1"}, {"source_file": "other.pdf", "location": "Page 2"}]
        collection = FakeCollection(self.documents, self.metadatas)
        bm25, docs, metas = rag_pipeline.build_bm25_index(collection, len(self.documents))
        self.runtime = rag_evaluator.EvaluationRuntime(
            collection, FakeEmbeddingModel(), bm25, docs, metas, rag_pipeline.RAGConfig()
        )
        self.content_hash = hashlib.sha256(self.documents[0].encode("utf-8")).hexdigest()

    def test_runner_uses_shared_runtime_and_calculates_metrics(self):
        result = rag_evaluator.evaluate_case(
            case_for(self.content_hash), self.runtime, FakeGenerator("Answer [SOURCE 1]")
        )
        self.assertEqual(result["trace"].rewritten_query, "alpha")
        self.assertEqual(result["metrics"]["recall_at_k"], 1.0)
        self.assertEqual(result["metrics"]["precision_at_k"], 1 / 15)
        self.assertEqual(result["metrics"]["hit_rate_at_k"], 1.0)
        self.assertEqual(result["metrics"]["mrr"], 1.0)
        self.assertEqual(result["metrics"]["ndcg_at_k"], 1.0)
        self.assertTrue(result["metrics"]["citation_valid"])
        self.assertTrue(result["metrics"]["expected_source_match"])
        self.assertIsNone(result["metrics"]["refusal_correct"])
        self.assertEqual(self.runtime.collection.calls[0]["n_results"], 10)

    def test_refusal_and_report_serialization_are_deterministic(self):
        result = rag_evaluator.evaluate_case(
            case_for(self.content_hash, answerability="unanswerable", response_mode="refuse_no_coverage"),
            self.runtime,
            FakeGenerator("I cannot find this in the document context."),
        )
        self.assertTrue(result["metrics"]["refusal_correct"])
        self.assertFalse(result["trace"].citations.display_sources)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            rag_evaluator.write_reports([result], output)
            self.assertTrue((output / "cases.json").exists())
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "summary.md").exists())
            self.assertFalse((output / "forensics" / "case-1.json").exists())
            self.assertFalse((output / "forensics" / "case-1.md").exists())
            payload = json.loads((output / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["case_id"], "case-1")
            self.assertTrue(payload[0]["trace"]["refusal_detected"])

    def test_evaluator_contains_no_duplicate_retrieval_or_generation_implementation(self):
        source = Path(rag_evaluator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("run_retrieval", function_names)
        self.assertNotIn("generate_answer", function_names)
        self.assertNotIn("BM25Okapi", source)
        self.assertNotIn("collection.query(", source)
        for name in (
            "rag_pipeline.build_bm25_index(", "rag_pipeline.rewrite_query(",
            "rag_pipeline.normalize_chroma_filter(",
            "rag_pipeline.hybrid_search(", "rag_pipeline.build_production_prompt(",
            "rag_pipeline.stream_generate(", "rag_pipeline.select_display_sources(",
        ):
            self.assertIn(name, source)

    def test_offline_embedding_loader_uses_cached_production_model(self):
        cached_model = object()
        with patch("rag_evaluator.SentenceTransformer", return_value=cached_model) as loader:
            self.assertIs(
                rag_evaluator.load_offline_embedding_model("paraphrase-multilingual-MiniLM-L12-v2"),
                cached_model,
            )
        loader.assert_called_once_with("paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)

    def test_offline_embedding_loader_fails_clearly_when_cache_is_missing(self):
        with patch("rag_evaluator.SentenceTransformer", side_effect=OSError("missing cache")):
            with self.assertRaisesRegex(RuntimeError, "Offline evaluation requires cached embedding model"):
                rag_evaluator.load_offline_embedding_model("paraphrase-multilingual-MiniLM-L12-v2")

    def test_evaluator_uses_shared_filter_normalization_for_multi_field_cases(self):
        case = case_for(self.content_hash)
        case["metadata_filter"] = {"application": "KPSA", "geographical_entity": "OCM"}
        rag_evaluator.evaluate_case(case, self.runtime, FakeGenerator("Answer [SOURCE 1]"))
        self.assertEqual(
            self.runtime.collection.calls[0]["where"],
            {"$and": [{"application": "KPSA"}, {"geographical_entity": "OCM"}]},
        )

    def test_run_cases_checkpoints_and_resumes_without_rerunning_completed_cases(self):
        first = case_for(self.content_hash)
        second = case_for(self.content_hash)
        second["id"] = "case-2"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            with patch("rag_evaluator.evaluate_case", wraps=rag_evaluator.evaluate_case) as evaluate:
                rag_evaluator.run_cases([first], self.runtime, FakeGenerator("Answer [SOURCE 1]"), output)
                results = rag_evaluator.run_cases([first, second], self.runtime, FakeGenerator("Answer [SOURCE 1]"), output, resume=True)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual([result["case_id"] for result in results], ["case-1", "case-2"])
            self.assertEqual(len(rag_evaluator.load_checkpoint(output)), 2)

    def test_run_cases_rejects_invalid_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                rag_evaluator.run_cases([], self.runtime, FakeGenerator(""), Path(temporary), case_timeout_seconds=0)

    def test_generation_stage_timeout_is_checkpointable(self):
        class SlowGenerator(FakeGenerator):
            def chat(self, **kwargs):
                if kwargs.get("stream"):
                    time.sleep(0.05)
                return super().chat(**kwargs)
        result = rag_evaluator.evaluate_case(
            case_for(self.content_hash), self.runtime, SlowGenerator("Answer [SOURCE 1]"), stage_timeout_seconds=0.001
        )
        self.assertEqual(result["trace"].failure.code, "generation_timeout")
        self.assertIn("query_rewriting", result["stage_timings_ms"])

    def test_evaluator_generation_budget_caps_only_streaming_output(self):
        generator = FakeGenerator("Answer")
        budget = rag_evaluator.EvaluatorGenerationBudget(generator, 17)
        budget.chat(model="qwen3:8b", messages=[], options={"temperature": 0.2}, stream=True)
        budget.chat(model="qwen3:8b", messages=[{"content": "QUESTION : x QUESTION REFORMULÉE"}], options={"temperature": 0.0}, stream=False)
        self.assertEqual(generator.calls[0]["options"], {"temperature": 0.2, "num_predict": 17})
        self.assertEqual(generator.calls[1]["options"], {"temperature": 0.0})

    def test_summary_separates_generation_timeouts(self):
        class SlowGenerator(FakeGenerator):
            def chat(self, **kwargs):
                if kwargs.get("stream"):
                    time.sleep(0.05)
                return super().chat(**kwargs)
        timed_out = rag_evaluator.evaluate_case(
            case_for(self.content_hash), self.runtime,
            SlowGenerator("Answer [SOURCE 1]"), stage_timeout_seconds=0.001,
        )
        successful = rag_evaluator.evaluate_case(
            case_for(self.content_hash), self.runtime, FakeGenerator("Answer [SOURCE 1]")
        )
        summary = rag_evaluator.aggregate([timed_out, successful])
        self.assertEqual(summary["generation_timeout_count"], 1)
        self.assertEqual(summary["successful_generation_count"], 1)

    def test_segmented_diagnostics_use_benchmark_and_forensic_dimensions(self):
        first = rag_evaluator.result_to_json(
            rag_evaluator.evaluate_case(case_for(self.content_hash), self.runtime, FakeGenerator("Answer [SOURCE 1]"))
        )
        second_case = case_for(self.content_hash, answerability="unanswerable", response_mode="refuse_no_coverage")
        second_case.update({"id": "case-2", "language": "en", "category": "typo", "metadata_filter": {"application": "KPSA"}})
        second = rag_evaluator.result_to_json(
            rag_evaluator.evaluate_case(second_case, self.runtime, FakeGenerator("I cannot find this in the document context."))
        )
        segments = rag_evaluator.segmented_diagnostics([first, second])
        self.assertEqual(segments["language"]["fr"]["case_count"], 1)
        self.assertEqual(segments["language"]["en"]["case_count"], 1)
        self.assertEqual(segments["query_type"]["typo"]["case_count"], 1)
        self.assertEqual(segments["metadata_filter_state"]["filtered"]["case_count"], 1)
        self.assertEqual(segments["fallback_usage"]["used"]["case_count"], 1)
        self.assertIn("passed", segments["forensic_category"])

    def test_metadata_filter_audit_identifies_coverage_and_filtered_outcomes(self):
        case = case_for(self.content_hash)
        case["metadata_filter"] = {"application": "Unknown", "stale": "missing"}
        result = rag_evaluator.evaluate_case(case, self.runtime, FakeGenerator("Answer [SOURCE 1]"))
        audit = rag_evaluator.metadata_filter_audit(
            [case],
            [{"application": "KPSA", "region": "OCM"}, {"application": "MZ"}],
            [result],
        )
        self.assertEqual(audit["active_metadata"]["fields"]["application"]["missing_count"], 0)
        self.assertEqual(audit["inconsistent_metadata"]["region"]["missing_count"], 1)
        self.assertEqual(audit["benchmark_filter_coverage"]["unmapped_field_count"], 1)
        self.assertEqual(audit["benchmark_filter_coverage"]["stale_value_count"], 1)
        self.assertEqual(audit["filtered_query_outcomes"]["case_count"], 1)
        self.assertEqual(audit["fallback_activation"]["fallback_used_count"], 1)

    def test_fixed_experiment_matrix_preserves_control_and_reports_variants(self):
        matrix = rag_evaluator.fixed_experiment_matrix(self.runtime.config)
        self.assertEqual(matrix[0], rag_evaluator.certified_control(self.runtime.config))
        report = rag_evaluator.run_experiment_matrix(
            [case_for(self.content_hash)], self.runtime, FakeGenerator("Answer [SOURCE 1]"), matrix
        )
        self.assertEqual(report["control"], "control")
        self.assertEqual(report["variants"]["control"]["configuration"]["final_top_k"], 15)
        self.assertIn("recall_at_k", report["variants"]["fusion_depth_5"]["metrics"])
        self.assertIn("fallback_rate", report["variants"]["fallback_threshold_5"])
        self.assertIn("forensic_counts", report["variants"]["candidate_depth_20"])

    def test_reranking_opportunities_identify_ranking_only_misses(self):
        relevant_hash = self.content_hash
        case = case_for(relevant_hash)
        result = {
            "case_id": "case-1", "metrics": {}, "trace": {
                "candidate_pool": [{"content_sha256": relevant_hash, "source": "vector", "rank": 12}],
                "selected_chunks": ["other policy"],
            },
        }
        opportunities = rag_evaluator.reranking_opportunity_diagnostics(
            [case], [result], [{"application": "KPSA"}]
        )
        self.assertEqual(opportunities["opportunity_count"], 1)
        self.assertEqual(opportunities["affected_cases"][0]["candidate_ranks"][0]["rank"], 12)
        self.assertEqual(opportunities["affected_cases"][0]["exclusion_reason"], "relevant_candidate_ranked_below_final_context_cutoff")
        self.assertFalse(opportunities["gates"]["candidate_opportunity"]["met"])

    def test_blank_generation_is_a_language_appropriate_clarification(self):
        case = case_for(self.content_hash, answerability="ambiguous", response_mode="request_clarification")
        result = rag_evaluator.evaluate_case(case, self.runtime, FakeGenerator(""))
        self.assertEqual(result["response"], rag_pipeline.build_clarification_message("French"))
        self.assertTrue(result["metrics"]["clarification_correct"])

    def test_unanswerable_no_source_case_accepts_the_approved_clarification_fallback(self):
        case = case_for(self.content_hash, answerability="unanswerable", response_mode="refuse_no_coverage")
        result = rag_evaluator.evaluate_case(case, self.runtime, FakeGenerator(""))
        self.assertFalse(result["trace"].citations.no_coverage_detected)
        self.assertTrue(result["metrics"]["clarification_correct"])
        self.assertTrue(result["metrics"]["refusal_correct"])

    def test_timeout_cases_are_checkpointed_and_excluded_from_quality_metrics(self):
        timed_out = rag_evaluator.timeout_result(
            case_for(self.content_hash), rag_evaluator.StageTimeoutError("generation", 1.0), 0.0, {}, lambda: 1.0
        )
        summary = rag_evaluator.aggregate([timed_out])
        completeness = rag_evaluator.completeness_report([timed_out], 1)
        self.assertIsNone(summary["recall_at_k"])
        self.assertEqual(completeness["timeout_case_count"], 1)
        self.assertFalse(completeness["baseline_comparison_allowed"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            rag_evaluator.write_reports([timed_out], output, expected_case_count=2)
            self.assertTrue((output / "completeness.json").exists())

    def test_citation_denominator_mismatch_is_not_comparable(self):
        summary = {"citation_evaluable_case_count": 0, "expected_source_evaluable_case_count": 0}
        baseline = {"evaluability": {"citation_evaluable_case_count": 9, "expected_source_evaluable_case_count": 9}}
        self.assertEqual(
            rag_evaluator.citation_metric_comparability(summary, baseline),
            {"citation_valid": "NOT_COMPARABLE", "expected_source_match": "NOT_COMPARABLE"},
        )

    def test_resume_retries_timeout_cases_but_skips_successful_cases(self):
        timed_out = rag_evaluator.timeout_result(
            case_for(self.content_hash), rag_evaluator.StageTimeoutError("generation", 1.0), 0.0, {}, lambda: 1.0
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            rag_evaluator.write_reports([timed_out], output, expected_case_count=1)
            with patch("rag_evaluator.evaluate_case", wraps=rag_evaluator.evaluate_case) as evaluate:
                resumed = rag_evaluator.run_cases([case_for(self.content_hash)], self.runtime, FakeGenerator("Answer [SOURCE 1]"), output, resume=True)
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(len(resumed), 1)


if __name__ == "__main__":
    unittest.main()
