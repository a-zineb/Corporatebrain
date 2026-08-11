"""Pure validation state for transactional DOCX batch migrations.

This module does not access ChromaDB.  Callers supply source-file counts before
and after each document transaction and apply the actual mutation/rollback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


Counts = dict[str, int]


@dataclass
class BatchMigrationState:
    """Track immutable initial and mutable accepted corpus-count baselines."""

    initial_snapshot: Counts
    accepted_snapshot: Counts | None = None
    approved_deltas: Counts = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.initial_snapshot = dict(self.initial_snapshot)
        self.accepted_snapshot = dict(self.initial_snapshot)

    def before_document(self) -> Counts:
        """Return the state immediately before the next document transaction."""
        return dict(self.accepted_snapshot or {})

    def validate_document(
        self,
        before: Mapping[str, int],
        after: Mapping[str, int],
        target_source: str,
        expected_new_count: int,
    ) -> bool:
        """Validate that only the target changed to its expected new count."""
        before_counts, after_counts = dict(before), dict(after)
        if after_counts.get(target_source) != expected_new_count:
            return False
        keys = set(before_counts) | set(after_counts)
        return all(
            source == target_source or after_counts.get(source, 0) == before_counts.get(source, 0)
            for source in keys
        )

    def accept_document(self, source: str, old_count: int, new_count: int, after: Mapping[str, int]) -> None:
        """Advance the accepted baseline after successful validation."""
        self.accepted_snapshot = dict(after)
        self.approved_deltas[source] = self.approved_deltas.get(source, 0) + (new_count - old_count)

    def final_expected_counts(self) -> Counts:
        """Compute final counts from the immutable initial snapshot and deltas."""
        result = dict(self.initial_snapshot)
        for source, delta in self.approved_deltas.items():
            result[source] = result.get(source, 0) + delta
        return result


def simulate_batch(initial_counts: Mapping[str, int], documents: list[tuple[str, int, int]]) -> list[dict[str, object]]:
    """Simulate sequential migrations without accessing ChromaDB.

    Each tuple is ``(source_file, current_count, proposed_count)``.  The
    returned plan records the accepted baseline before and after every step.
    """
    state = BatchMigrationState(dict(initial_counts))
    plan: list[dict[str, object]] = []
    for source, old_count, new_count in documents:
        before = state.before_document()
        after = dict(before)
        after[source] = new_count
        valid = state.validate_document(before, after, source, new_count)
        row = {
            "source_file": source,
            "current_legacy_count": old_count,
            "proposed_structured_count": new_count,
            "expected_delta": new_count - old_count,
            "accepted_baseline_before": before,
            "expected_baseline_after": after,
            "status": "READY" if valid else "INVALID",
        }
        plan.append(row)
        if not valid:
            break
        state.accept_document(source, old_count, new_count, after)
    return plan
