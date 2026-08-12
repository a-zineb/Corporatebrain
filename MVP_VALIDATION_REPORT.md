# Corporate Brain MVP — Validation report

Date: 2026-08-11

## Architecture before and after

Before, canonical normalization and deterministic QA existed, but Streamlit
mixed upload state, source display, answer formatting and mode routing inside a
single large script. Sources opened files rather than an exact internal target,
uploads reran after every successful ingestion, and response contracts did not
carry per-message language or suggestions.

After, `mvp_services.py` owns idempotent preparation, all-document deterministic
search, source targets, clarification candidates and local metrics.
`ui_components.py` owns typography, chat answer types, tables, multi-values,
source cards/viewer, suggestions, status, See more and Find Me. The existing
canonical engine remains the selected-document security boundary.

## Implemented behavior

- Per-message English/French detection is stored in `AnswerResult.query_language`.
- Clarifications are derived only from actual field labels and section titles.
- Find Me searches every prepared canonical document and never mutates active selection.
- Provenance includes page/bbox, paragraph, table/row, section and sheet.
- PDF targets render the exact page and rectangle when coordinates exist; all formats have a highlighted canonical fallback.
- Section/table/multi-value answer types use reusable renderers and consistent app typography.
- Response mode/scope and UI language use segmented controls; documents use pills.
- Upload is keyed by content hash plus schema version and has terminal success/warning/failure states.
- Existing files are prepared once in a cached startup registry.
- Direct metrics remain local; debug details remain behind existing environment flags.

## Automated validation

- Full suite after changes: 314 passing tests (including 7 new MVP tests).
- Two pre-existing environment-contract failures remain: active Chroma has 3,165 chunks while the certified manifest expects 1,072.
- Focused app, startup, PDF, DOCX and canonical regression run: 128/128 passed.
- Certified Direct Answer benchmark: 96/100 correct, 0 wrong, 4 false no-evidence, 0 cross-document leakage, 0 secret leakage.
- Per format: DOCX 24/25, PDF 24/25, XLSX 24/25, CSV 24/25.
- Warm latency: p50 0.185 ms, p95 0.515 ms, max 6.043 ms.
- Cold preparation: DOCX 20.315 ms, PDF 10.225 ms, XLSX 3.224 ms, CSV 2.001 ms.
- Direct Answer calls: Ollama 0, Chroma 0.
- Knowledge Catalog over 26 unique documents: p50 0.227 ms, p95 0.330 ms.

## Requested feature summary

English/French switching, global Find Me, typo suggestions, source navigation,
multi-page PDF table reconstruction, multi-value aggregation, section answers,
table rendering, button controls, modern chat styling, thinking/generation
labels, compact See more metadata, blank-container prevention, terminal upload
states, success notification and selected-document isolation are implemented.
Existing ZEBRA PDF and DOCX regressions pass. XLSX/CSV are covered by the
certified synthetic benchmark because the local storage corpus has no CSV file.

## Known limitations and UAT decision

- Four certified queries still return false no-evidence, so supported recall is 96% (above the 95% MVP target, below the 98% stretch target).
- The stored Chroma collection does not match the committed 1,072-chunk benchmark manifest; it must be reconciled or regenerated before calling that collection certified.
- Legacy `.doc` visual fidelity depends on Microsoft Word conversion; the canonical viewer remains the reliable fallback.
- Browser-level visual checks and a live Ollama generation session remain manual UAT items.

The duplicate is ready for user acceptance testing, with the Chroma manifest mismatch explicitly excluded from certification until reconciled.
