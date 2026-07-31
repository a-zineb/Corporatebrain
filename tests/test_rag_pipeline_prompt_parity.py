"""Exact prompt, citation, and refusal parity checks against app.py."""

from __future__ import annotations

import ast
import copy
import os
import re
import unittest

import rag_pipeline
from baseline_app_reference import assignment_nodes as baseline_assignment_nodes
from baseline_app_reference import execute_assignment
from baseline_app_reference import tree as baseline_tree

def app_tree():
    return baseline_tree()


def assignment_nodes(name):
    return baseline_assignment_nodes(name)


def load_legacy_source_builder():
    function = next(
        node
        for node in ast.walk(app_tree())
        if isinstance(node, ast.FunctionDef) and node.name == "build_source_list"
    )
    function = copy.deepcopy(function)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"os": os, "STORAGE_DIR": "doc_storage_v2"}
    exec(compile(module, "app.py", "exec"), namespace)
    return namespace["build_source_list"]


def legacy_citations(response):
    namespace = {"re": re, "full_stream_response": response}
    execute_assignment("cited_ids", namespace, occurrence=0)
    execute_assignment("cited_ids", namespace, occurrence=1)
    return tuple(namespace["cited_ids"])


def legacy_no_coverage(response):
    namespace = {"re": re, "full_stream_response": response}
    execute_assignment("response_lower", namespace)
    execute_assignment("no_coverage_patterns", namespace)
    execute_assignment("no_documentary_answer", namespace)
    return namespace["no_documentary_answer"]


class PromptParityTests(unittest.TestCase):
    """Compare shared prompt/source behavior directly with the legacy source."""

    def build_sources(self):
        chunks = ["Premier extrait", "Second excerpt"]
        metas = [
            {"source_file": "OCM.docx", "location": "Page 1"},
            {"source_file": "MZ.docx", "location": "Page 2"},
        ]
        legacy_builder = load_legacy_source_builder()
        legacy_filtered = legacy_builder(chunks[:1], metas[:1])
        legacy_relaxed = legacy_builder(chunks[1:], metas[1:], relaxed_flag=True, start_id=2)
        shared_filtered = rag_pipeline.build_source_list(chunks[:1], metas[:1], "doc_storage_v2")
        shared_relaxed = rag_pipeline.build_source_list(chunks[1:], metas[1:], "doc_storage_v2", relaxed_flag=True, start_id=2)
        return chunks, metas, legacy_filtered + legacy_relaxed, shared_filtered + shared_relaxed

    def test_source_context_numbering_and_french_prompt_are_byte_identical(self):
        _chunks, _metas, legacy_sources, shared_sources = self.build_sources()
        history = [
            {"role": "user", "content": "Question précédente"},
            {"role": "assistant", "content": "Réponse précédente"},
        ]
        legacy_context = "\n---\n".join(
            f"[SOURCE {source['id']}]{' (hors des filtres actifs — piste proche)' if source.get('relaxed') else ''}\n{source['text']}"
            for source in legacy_sources
        )
        legacy_history = "".join(
            f"{'Utilisateur' if message['role'] == 'user' else 'Assistant'}: {message['content']}\n"
            for message in history[-4:]
        )
        legacy_relaxed_note = execute_assignment("relaxed_note", {"was_relaxed": True})
        legacy_prompt = execute_assignment(
            "prompt_instructions",
            {
                "filter_ent": "OCM",
                "filter_application": "MZ",
                "recent_chat_history": legacy_history,
                "context_str": legacy_context,
                "relaxed_note": legacy_relaxed_note,
                "user_query": "Explique le flux.",
                "current_lang": "French",
            },
        )
        shared = rag_pipeline.build_production_prompt(
            user_query="Explique le flux.",
            filter_ent="OCM",
            filter_application="MZ",
            history=history,
            sources=shared_sources,
            current_lang="French",
            was_relaxed=True,
        )

        self.assertEqual(
            [
                {"id": source.source_id, "file": source.file_name, "loc": source.location, "text": source.text,
                 "path": source.path, "relaxed": source.relaxed}
                for source in shared.sources
            ],
            legacy_sources,
        )
        self.assertEqual(shared.context, legacy_context)
        self.assertEqual(shared.prompt, legacy_prompt)

    def test_english_and_multilingual_prompt_inputs_match_legacy(self):
        _chunks, _metas, legacy_sources, shared_sources = self.build_sources()
        history = [{"role": "user", "content": "¿Qué significa OCM?"}]
        legacy_context = "\n---\n".join(
            f"[SOURCE {source['id']}]{' (hors des filtres actifs — piste proche)' if source.get('relaxed') else ''}\n{source['text']}"
            for source in legacy_sources
        )
        legacy_history = "Utilisateur: ¿Qué significa OCM?\n"
        legacy_prompt = execute_assignment(
            "prompt_instructions",
            {
                "filter_ent": "Tous",
                "filter_application": "Tous",
                "recent_chat_history": legacy_history,
                "context_str": legacy_context,
                "relaxed_note": execute_assignment("relaxed_note", {"was_relaxed": False}),
                "user_query": "What does OCM mean?",
                "current_lang": "English",
            },
        )
        shared = rag_pipeline.build_production_prompt(
            user_query="What does OCM mean?",
            filter_ent="Tous",
            filter_application="Tous",
            history=history,
            sources=shared_sources,
            current_lang="English",
            was_relaxed=False,
        )
        self.assertEqual(shared.prompt, legacy_prompt)

    def test_empty_context_no_match_messages_match_legacy(self):
        for language in ("French", "English"):
            legacy = execute_assignment("no_match_msg", {"current_lang": language})
            self.assertEqual(rag_pipeline.build_no_match_message(language), legacy)
            self.assertEqual(rag_pipeline.build_context(()), "")

    def test_citations_duplicates_invalid_and_malformed_cases_match_legacy(self):
        sources = rag_pipeline.build_source_list(
            ["A", "B"],
            [
                {"source_file": "one.pdf", "location": "Page 1"},
                {"source_file": "two.pdf", "location": "Page 2"},
            ],
            "doc_storage_v2",
        )
        valid_response = "Fait [SOURCE 1], encore [SOURCE 2], doublon [SOURCE 1]."
        result = rag_pipeline.select_display_sources(valid_response, sources)
        self.assertEqual(result.cited_source_ids, legacy_citations(valid_response))
        self.assertEqual([source.source_id for source in result.display_sources], [1, 2])
        self.assertEqual(result.invalid_source_ids, ())

        invalid_response = "Invalide [SOURCE 99] et malformé [SOURCE x] [source 1] [SOURCE -1]."
        invalid = rag_pipeline.select_display_sources(invalid_response, sources)
        self.assertEqual(invalid.cited_source_ids, legacy_citations(invalid_response))
        self.assertEqual(invalid.invalid_source_ids, (99,))
        self.assertEqual(invalid.display_sources, ())

    def test_confirmed_refusals_suppress_sources_and_qualified_answers_preserve_them(self):
        sources = rag_pipeline.build_source_list(
            ["A"],
            [{"source_file": "one.pdf", "location": "Page 1"}],
            "doc_storage_v2",
        )
        responses = [
            "Je ne trouve pas cette information dans les documents. [SOURCE 1]",
            "Cette information n'est pas dans le contexte. [SOURCE 1]",
            "Information absente du corpus. [SOURCE 1]",
            "Le contexte fourni ne contient aucune information. [SOURCE 1]",
            "Le contexte fourni ne contient pas cette réponse. [SOURCE 1]",
            "Les documents ne contiennent aucune information. [SOURCE 1]",
            "Je ne trouve pas de réponse dans le contexte fourni. [SOURCE 1]",
            "La question posée ne trouve pas de réponse dans le contexte fourni. [SOURCE 1]",
            "Je ne trouve pas de réponse dans les documents. [SOURCE 1]",
            "Ce point n'est pas mentionné dans les documents. [SOURCE 1]",
            "Ce point n'est pas abordé dans les documents. [SOURCE 1]",
            "L'information n'est pas présente dans les documents. [SOURCE 1]",
            "I cannot find this in the document context. [SOURCE 1]",
            "Not mentioned in the source. [SOURCE 1]",
        ]
        for response in (
            "Désolé, je n'ai pas trouvé d'information sur ce sujet.",
            "Je n’ai pas trouvé d'informations dans le contexte fourni.",
            "Je n'ai trouvé aucune mention de numéro de sécurité sociale dans le contexte fourni.",
            "Désolé, mais aucun des documents fournis ne contient d'informations relatives à un numéro de sécurité sociale.",
            "I cannot find this in the document context.",
        ):
            result = rag_pipeline.select_display_sources(response, sources)
            self.assertTrue(result.no_coverage_detected)
            self.assertEqual(result.display_sources, ())
        qualified = "Ce point n'est pas explicitement mentionné. Cependant, [SOURCE 1] apporte un élément pertinent."
        result = rag_pipeline.select_display_sources(qualified, sources)
        self.assertFalse(result.no_coverage_detected)
        self.assertEqual([source.source_id for source in result.display_sources], [1])
        clarification = rag_pipeline.select_display_sources(
            "Pouvez-vous préciser l'application, le document ou le contexte concerné ?", sources
        )
        self.assertFalse(clarification.no_coverage_detected)

    def test_display_deduplication_keeps_first_matching_legacy_ui_order(self):
        sources = (
            rag_pipeline.PromptSource(1, "one.pdf", "Page 1", "A", "same-path"),
            rag_pipeline.PromptSource(2, "one.pdf", "Page 2", "B", "same-path"),
            rag_pipeline.PromptSource(3, "two.pdf", "Page 1", "C", "other-path"),
        )
        self.assertEqual(
            [source.source_id for source in rag_pipeline.deduplicate_sources_by_path(sources)],
            [1, 3],
        )


if __name__ == "__main__":
    unittest.main()
