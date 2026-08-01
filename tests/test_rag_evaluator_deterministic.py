"""Deterministic evaluator checks using only shared-runtime test doubles."""

from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import rag_evaluator
import rag_pipeline


def isolated_success(value):
    """Pickle-safe child operation used to validate evaluator process cleanup."""

    return f"completed:{value}"


def isolated_sleep(seconds):
    """Pickle-safe slow child operation used to validate hard cancellation."""

    time.sleep(seconds)
    return "too-late"


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
        "acceptable_answer_points": ["alpha"] if answerability == "answerable" else [],
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

    def test_missing_citations_are_not_evaluable(self):
        metrics = rag_evaluator.citation_metrics(None, [{"content_sha256": self.content_hash}])
        self.assertEqual(metrics["citation_status"], "NOT_EVALUABLE")
        self.assertEqual(metrics["citation_evaluable_count"], 0)
        self.assertIsNone(metrics["citation_valid"])
        self.assertIsNone(metrics["expected_source_match"])

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

    def test_isolated_model_stage_returns_normal_success_and_cleans_up(self):
        before = {child.pid for child in multiprocessing.active_children()}
        value, elapsed = rag_evaluator.run_isolated_stage(
            "generation", isolated_success, "answer", 20.0,
        )
        self.assertEqual(value, "completed:answer")
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_isolated_model_stage_terminates_timed_out_child_without_orphan(self):
        before = {child.pid for child in multiprocessing.active_children()}
        with self.assertRaisesRegex(rag_evaluator.StageTimeoutError, "generation exceeded 0.01 seconds"):
            rag_evaluator.run_isolated_stage("generation", isolated_sleep, 1.0, 0.01)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_multiple_isolated_timeouts_remain_sequential_and_leave_no_children(self):
        before = {child.pid for child in multiprocessing.active_children()}
        for _ in range(2):
            with self.assertRaises(rag_evaluator.StageTimeoutError):
                rag_evaluator.run_isolated_stage("generation", isolated_sleep, 1.0, 0.01)
            self.assertEqual({child.pid for child in multiprocessing.active_children()}, before)

    def test_local_ollama_generator_uses_the_isolated_model_boundary(self):
        request = rag_evaluator.LocalModelStageRequest(
            stage="query_rewriting", model_name="qwen3:8b", question="alpha",
        )
        generator = rag_evaluator.EvaluatorGenerationBudget(rag_evaluator.ollama, 256)
        expected = rag_pipeline.QueryRewriteResult("alpha", 0.0)
        with patch("rag_evaluator.run_isolated_stage", return_value=(expected, 3.0)) as isolated:
            result, elapsed = rag_evaluator.run_model_stage(
                request, generator, lambda: self.fail("local generator must not use the in-process fallback"),
            )
        self.assertEqual((result, elapsed), (expected, 3.0))
        isolated.assert_called_once()

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

    def test_grounding_context_orders_deduplicates_and_bounds_existing_sources(self):
        sources = (
            rag_pipeline.PromptSource(1, "noise.pdf", "Page 1", "noise", "noise.pdf"),
            rag_pipeline.PromptSource(2, "answer.pdf", "Page 1", "alpha", "answer.pdf"),
            rag_pipeline.PromptSource(3, "duplicate.pdf", "Page 2", "noise", "duplicate.pdf"),
        )
        answer_hash = hashlib.sha256(b"alpha").hexdigest()
        evidence_first, _, _ = rag_evaluator.grounding_context(
            sources, [{"content_sha256": answer_hash, "label": 2}], rag_evaluator.GROUNDING_EXPERIMENTS[1]
        )
        deduplicated, _, duplicates = rag_evaluator.grounding_context(
            sources, [], rag_evaluator.GROUNDING_EXPERIMENTS[2]
        )
        focused, _, bounded = rag_evaluator.grounding_context(
            sources, [{"content_sha256": answer_hash, "label": 2}],
            rag_evaluator.ContextGroundingExperiment("test", "focused", max_sources=1),
        )
        self.assertEqual([source.source_id for source in evidence_first], [2, 1, 3])
        self.assertEqual([source.source_id for source in deduplicated], [1, 2])
        self.assertEqual(duplicates[0]["reason"], "duplicate_content")
        self.assertEqual([source.source_id for source in focused], [2])
        self.assertEqual({item["reason"] for item in bounded}, {"outside_focused_bound"})

    def test_grounding_experiments_reuse_one_control_retrieval_and_report_parity(self):
        case = case_for(self.content_hash)
        report, rows = rag_evaluator.run_grounding_experiments(
            [case], self.runtime, FakeGenerator("alpha [SOURCE 1]")
        )
        self.assertEqual(len(self.runtime.collection.calls), 1)
        self.assertEqual(set(rows), {
            "G0_control", "G1_evidence_first", "G2_deduplicated_context", "G3_focused_context",
            "G4_explicit_facts", "G5_citation_required",
        })
        self.assertTrue(all(report["variants"][name]["retrieval_parity"] for name in rows))
        self.assertEqual(report["variants"]["G0_control"]["metrics"]["answer_use"], 1.0)
        self.assertEqual(report["variants"]["G0_control"]["metrics"]["grounded_answer_correct"], 1.0)
        self.assertEqual(report["variants"]["G5_citation_required"]["metrics"]["citation_obligation_coverage"], 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            rag_evaluator.write_grounding_experiment_report(report, rows, Path(temporary))
            payload = json.loads((Path(temporary) / "context_grounding_experiments.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["summary"]["variants"]["G3_focused_context"]["retrieval_parity"])

    def test_prompt_grounding_variants_append_only_the_documented_suffix(self):
        case = case_for(self.content_hash)
        report, rows = rag_evaluator.run_grounding_experiments(
            [case], self.runtime, FakeGenerator("alpha [SOURCE 1]")
        )
        control = rows["G0_control"][0]["trace"]
        g4 = rows["G4_explicit_facts"][0]["trace"]
        g5 = rows["G5_citation_required"][0]["trace"]
        self.assertEqual(g4.prompt.sources, control.prompt.sources)
        self.assertEqual(g5.prompt.sources, control.prompt.sources)
        self.assertEqual(g4.prompt.prompt, f"{control.prompt.prompt}\n\n{rag_evaluator.G4_EXPLICIT_FACTS_SUFFIX}")
        self.assertEqual(g5.prompt.prompt, f"{control.prompt.prompt}\n\n{rag_evaluator.G5_CITATION_REQUIRED_SUFFIX}")
        self.assertTrue(report["variants"]["G4_explicit_facts"]["retrieval_parity"])
        self.assertTrue(report["variants"]["G5_citation_required"]["retrieval_parity"])

    def test_stability_comparison_requires_same_direction_and_fingerprint(self):
        def report(answer_use):
            return {"variants": {
                "G0_control": {"metrics": {"answer_use": 0.0, "generation_timeout_count": 0}},
                "G4_explicit_facts": {"metrics": {"answer_use": answer_use, "generation_timeout_count": 0}, "retrieval_parity": True},
            }}
        first = report(1.0)
        second = report(1.0)
        comparison = rag_evaluator.compare_grounding_stability(first, second, {"corpus": "same"}, {"corpus": "same"})
        self.assertTrue(comparison["fingerprint_match"])
        self.assertTrue(comparison["variants"]["G4_explicit_facts"]["complete"])
        self.assertEqual(
            comparison["variants"]["G4_explicit_facts"]["metric_directions"]["answer_use"],
            {"first_run": "IMPROVED", "second_run": "IMPROVED", "stable": True},
        )

    def test_grounding_variant_preserves_variant_label_after_control_timeout(self):
        control = rag_evaluator.timeout_result(
            case_for(self.content_hash), rag_evaluator.StageTimeoutError("generation", 1.0), 0.0, {}, lambda: 1.0
        )
        variant = rag_evaluator.evaluate_grounding_variant(
            case_for(self.content_hash), control, self.runtime, FakeGenerator("alpha"),
            rag_evaluator.GROUNDING_EXPERIMENTS[1],
        )
        self.assertEqual(variant["experiment"], "G1_evidence_first")
        self.assertEqual(variant["status"], "NOT_RUN")
        self.assertEqual(variant["trace"].failure.code, "not_run_control_failed")
        self.assertIsNone(variant["metrics"]["latency_ms"])
        self.assertTrue(variant["grounding_context"]["retrieval_parity"])

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

    def test_grounding_runs_continue_after_generation_error(self):
        class ErrorGenerator(FakeGenerator):
            def chat(self, **kwargs):
                if kwargs.get("stream"):
                    raise RuntimeError("local generation failed")
                return super().chat(**kwargs)
        report, rows = rag_evaluator.run_grounding_experiments(
            [case_for(self.content_hash)], self.runtime, ErrorGenerator(""),
        )
        self.assertEqual(set(rows), {"G0_control", "G1_evidence_first", "G2_deduplicated_context", "G3_focused_context", "G4_explicit_facts", "G5_citation_required"})
        self.assertEqual(rows["G0_control"][0]["trace"].failure.code, "generation_error")
        self.assertEqual(rows["G4_explicit_facts"][0]["experiment"], "G4_explicit_facts")
        self.assertEqual(report["variants"]["G0_control"]["metrics"]["citation_evaluable_case_count"], 0)

    def test_grounding_runner_checkpoints_variants_and_resumes_two_cases(self):
        first = case_for(self.content_hash)
        second = case_for(self.content_hash)
        second["id"] = "case-2"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "grounding"
            partial, partial_rows = rag_evaluator.run_grounding_experiments_resumable(
                [first, second], self.runtime, FakeGenerator("alpha [SOURCE 1]"), output,
                max_new_variants=1,
            )
            self.assertEqual(len(partial_rows["G0_control"]), 1)
            self.assertEqual(len(partial_rows["G4_explicit_facts"]), 0)
            self.assertEqual(json.loads((output / "partial_summary.json").read_text(encoding="utf-8"))["status"], "PARTIAL")
            self.assertTrue((output / "grounding_checkpoint.pkl").exists())
            with patch("rag_evaluator.evaluate_case", wraps=rag_evaluator.evaluate_case) as evaluate:
                complete, rows = rag_evaluator.run_grounding_experiments_resumable(
                    [first, second], self.runtime, FakeGenerator("alpha [SOURCE 1]"), output,
                    resume=True,
                )
            self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(evaluate.call_args.args[0]["id"], "case-2")
            self.assertEqual({len(values) for values in rows.values()}, {2})
            summary = json.loads((output / "partial_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertEqual(summary["completed_variant_count"], 2 * len(rag_evaluator.PROMPT_GROUNDING_EXPERIMENTS))
            self.assertTrue(summary["fingerprint_match"])
            self.assertEqual(summary["fingerprint_before"], summary["fingerprint_after"])
            self.assertIn("G0_control", complete["variants"])

    def test_grounding_two_pass_summary_is_written_after_each_pass(self):
        first = case_for(self.content_hash)
        second = case_for(self.content_hash)
        second["id"] = "case-2"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "two-pass"
            result = rag_evaluator.run_grounding_two_passes_resumable(
                [first, second], self.runtime, FakeGenerator("alpha [SOURCE 1]"), output,
            )
            self.assertEqual(result["status"], "COMPLETE")
            summary = json.loads((output / "two_pass_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(summary["passes"]), {"run_1", "run_2"})
            self.assertEqual(summary["passes"]["run_1"]["partial_summary"]["status"], "COMPLETE")
            self.assertEqual(summary["passes"]["run_2"]["partial_summary"]["status"], "COMPLETE")
            self.assertTrue(summary["passes"]["run_1"]["partial_summary"]["fingerprint_match"])
            self.assertTrue(summary["passes"]["run_2"]["partial_summary"]["fingerprint_match"])

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
