# Corporate Brain MVP — User acceptance checklist

Use one document whose facts you know, then repeat the core checks with a PDF,
DOCX, XLSX and CSV. Never use a real password as a test value.

## Ingestion and document state

- [ ] Upload a supported file from **Admin · Ingestion manuelle**.
- [ ] Confirm the preparation spinner stops without refreshing the browser.
- [ ] Confirm a success toast and **Ready** or **Ready with warnings** badge appears.
- [ ] Leave the uploader visible and interact with another control; confirm the file is not ingested twice.
- [ ] Upload an unreadable file; confirm the spinner stops and an actionable error appears.

## Selection and isolation

- [ ] Select document A with the document pills and ask for a distinctive fact.
- [ ] Switch to document B and ask the same question.
- [ ] Confirm no source or value from A appears in B's answer.
- [ ] Switch back to A and confirm its answer remains stable.

## Language and query quality

- [ ] Ask an English question and confirm answer labels and answer are English.
- [ ] Ask the next question in French and confirm answer labels and answer switch to French.
- [ ] Test accents: `é è à ç œ` and straight/curly apostrophes.
- [ ] Ask a misspelled question such as `cdr formar of BI ?`.
- [ ] If confidence is insufficient, click one **Did you mean? / Voulez-vous dire :** chip and confirm it reruns.

## Answer structures

- [ ] Ask a single-value question and confirm only the value is prominent.
- [ ] Ask `host?`, `directory?`, or another multi-answer question and confirm all distinct values are shown.
- [ ] Ask a section title such as `Historique des modifications` or `Glossaire`.
- [ ] Confirm the section contents appear, not only its heading.
- [ ] Confirm a history/matrix answer uses a real table with ordered columns and horizontal scrolling when needed.
- [ ] Confirm repeated identical values are deduplicated.
- [ ] Ask for a password/token and confirm it is blocked while safe fields in the same row remain answerable.

## Sources and global search

- [ ] Open a source from a PDF answer; confirm the correct page is rendered and evidence is highlighted.
- [ ] Open DOCX/DOC evidence; confirm the canonical paragraph/table row is highlighted.
- [ ] Open XLSX/CSV evidence; confirm sheet/row information and highlighted row are correct.
- [ ] Use previous/next controls and close the viewer; confirm chat state is preserved.
- [ ] Open **Find me: where / Trouver : où**, search a term found in several documents, and inspect grouped results.
- [ ] Close it and confirm the active document and normal scope did not change.

## Modes and operational UX

- [ ] Browse Knowledge Catalog and filter by filename/title/application/zone/version.
- [ ] Confirm Knowledge Catalog responds without starting Ollama.
- [ ] Use AI Answer and confirm its evidence respects the selected-document scope.
- [ ] Confirm the generating indicator appears and the streamed answer finishes cleanly.
- [ ] Expand **See more / Voir plus** and verify method, pages, latency, language and state.
- [ ] Confirm no empty black response block appears after any answer type.

