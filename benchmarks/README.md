# Corporate Brain benchmark contract

`corporatebrain.v1.jsonl` is a reviewed, versioned benchmark for the certified
`rag_pipeline.py` runtime. It is data, not a production input. A run is comparable
only when its active collection matches `corporatebrain.v1.manifest.json`.

## Annotation identity

Every answerable relevance annotation uses the active Chroma `chunk_id` plus the
SHA-256 of that chunk's exact text. `source_file` and `location` make reviews
readable. If a future corpus has no stable IDs, annotators must use
`anchor_type: content_metadata_fallback` and identify the chunk by content hash,
source file, and location. The fallback must be recorded in the benchmark version
notes before use.

Labels are `2` for direct support, `1` for partial/supporting context, and `0` for
known non-relevant context. This seed uses only labels 1 and 2.

## Behavior labels

`answerable` cases use `expected_behavior.mode: answer` and contain a corpus-backed
reference answer. `unanswerable` cases use `refuse_no_coverage` with no relevance or
acceptable citations. `ambiguous` cases use `request_clarification`; they may retain
supporting anchors to explain why a question is ambiguous, but do not prescribe a
single answer.

## Metric boundary

Deterministic evaluation may calculate retrieval metrics, expected-source matching,
and citation validity: a displayed citation must resolve to an annotated acceptable
source. It must not claim claim-level citation coverage unless a later benchmark
version adds explicit claim annotations.

Faithfulness, answer relevancy, context precision/recall, and model-judged
hallucination remain future, non-deterministic metrics. They are never combined with
the deterministic baseline score.

## Review and versioning

An author proposes a record; a second reviewer verifies its chunk anchor, exact
content hash, answerability, expected behavior, filter, and relevance label. An
adjudication is required for disagreements. Any changed record, corpus fingerprint,
or annotation policy creates a new benchmark version.

Generated reports belong under `evaluation_runs/` or `evaluation_reports/`; both are
ignored by Git. Benchmark definitions, manifests, and schema files remain versioned.
