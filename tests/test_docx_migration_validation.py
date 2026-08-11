import unittest

from docx_migration import BatchMigrationState, simulate_batch


class BatchMigrationValidationTests(unittest.TestCase):
    def test_two_sequential_migrations_use_accepted_baseline(self):
        state = BatchMigrationState({"a.docx": 2, "b.docx": 3, "other.pdf": 5})
        before_a = state.before_document()
        after_a = {"a.docx": 10, "b.docx": 3, "other.pdf": 5}
        self.assertTrue(state.validate_document(before_a, after_a, "a.docx", 10))
        state.accept_document("a.docx", 2, 10, after_a)
        before_b = state.before_document()
        after_b = {"a.docx": 10, "b.docx": 20, "other.pdf": 5}
        self.assertTrue(state.validate_document(before_b, after_b, "b.docx", 20))
        state.accept_document("b.docx", 3, 20, after_b)
        self.assertEqual(state.final_expected_counts(), after_b)

    def test_failure_after_two_successes_does_not_approve_third(self):
        state = BatchMigrationState({"a.docx": 1, "b.docx": 1, "c.docx": 1})
        first = {"a.docx": 2, "b.docx": 1, "c.docx": 1}
        second = {"a.docx": 2, "b.docx": 3, "c.docx": 1}
        self.assertTrue(state.validate_document(state.before_document(), first, "a.docx", 2))
        state.accept_document("a.docx", 1, 2, first)
        self.assertTrue(state.validate_document(state.before_document(), second, "b.docx", 3))
        state.accept_document("b.docx", 1, 3, second)
        failed = {"a.docx": 2, "b.docx": 3, "c.docx": 4, "other.pdf": 1}
        self.assertFalse(state.validate_document(state.before_document(), failed, "c.docx", 4))
        self.assertEqual(state.before_document(), second)

    def test_rollback_baseline_is_immediately_pre_document_state(self):
        state = BatchMigrationState({"a.docx": 1, "b.docx": 1})
        accepted = {"a.docx": 2, "b.docx": 1}
        state.accept_document("a.docx", 1, 2, accepted)
        before_b = state.before_document()
        self.assertEqual(before_b, accepted)
        self.assertEqual(state.final_expected_counts(), accepted)

    def test_final_cumulative_delta_uses_initial_snapshot(self):
        state = BatchMigrationState({"a.docx": 2, "b.docx": 3})
        state.accept_document("a.docx", 2, 5, {"a.docx": 5, "b.docx": 3})
        state.accept_document("b.docx", 3, 7, {"a.docx": 5, "b.docx": 7})
        self.assertEqual(state.final_expected_counts(), {"a.docx": 5, "b.docx": 7})

    def test_simulate_batch_tracks_accepted_baseline(self):
        plan = simulate_batch(
            {"crbt.docx": 162, "p2p.docx": 6, "other.pdf": 4},
            [("p2p.docx", 6, 143)],
        )
        self.assertEqual(plan[0]["status"], "READY")
        self.assertEqual(plan[0]["accepted_baseline_before"]["crbt.docx"], 162)
        self.assertEqual(plan[0]["expected_baseline_after"]["p2p.docx"], 143)


if __name__ == "__main__":
    unittest.main()
