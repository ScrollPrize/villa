# R4 implementation delta review

Reviewed the uncommitted implementation against `2b5d1e01f`, using the R4 adjudications, per-area fix notes, `reading.md`, and the coherent-store contract. This is a source-level review of the fixes, not a fresh broad audit. No builds, tests, installations, or production/test edits were performed by this reviewer; the coordinating agent owns runtime verification.

## Confirmed delta issue, corrected during review

**D-001 — misplaced AFV test assertions (P2, test-suite regression).** At the reviewed state, `tests/python/test_import_afv.py:54` begins a block using `verify`, `config`, `path`, and `tool` inside `test_import_keeps_exact_bytes_metadata_and_refuses_existing_output`, which defines none of those names. Its first new assertion raises `NameError` after the import. The block belongs after the result assertions in `test_native_batch_command_readback_is_opt_in`, where those variables are defined. Its accidental relocation also removes that test's readback-option and removed-legacy-mode checks. Restore the block to that native test, retaining the new pre-existing-report replacement fixture. Reported to the coordinator immediately; source-confirmed, not independently executed.

The coordinator restored this block to the native test. Follow-up source inspection confirmed the misplaced names are absent from the first test and the native readback/legacy-mode assertions are restored. No remaining defect is recorded for D-001; runtime verification remains with the coordinator.

## Production delta disposition

No additional confirmed production defect was found in the inspected fixes:

- C-002 uses the shared retained allocation ledger, rejects unrelated foreign IDs already allocated by the target, and leaves same-target historical slice restoration available. The surrounding operation savepoint still owns failure rollback.
- C-003 and N-002 retain the loaded full object reference, write generation, and mapping guard. Artifact publication shares manifest/upload/submission code with numeric convenience publication without changing that convenience API's current-head semantics. Mapping operations reach the ordinary checked executor; the geometry writer uses the same remembered tool state.
- C-004 refills only object-limited pages lost to precedence; backend exclusions run before object admission. Spent output/processing/candidate/SQL work is retained across pages, and visible intersection counts are recomputed. Limits remain per attachment/draft group. Cancellation receives prior-page work plus the current callback counters. No unbounded prelisting was introduced.
- N-001 retries correction saving before destination preparation or correction reload. The existing failed-save return therefore vetoes restart; the reviewed regression covers persistence/retry, not an interactive GUI toggle.
- P-001 carries canonical source validity through stored reads, mesh loading, approval loading, and scaling, including a valid transformed x-coordinate of -1. Local tifxyz loading also delegates validity to the existing reader.
- S-002 wraps both direct-copy writes in the shared savepoint. That helper restores `StatementPool::currentWrite` bookkeeping on rollback as well as SQL state; it is not merely a database-only rollback.
- S-004 checks integer range before narrowing, including unsigned JSON values. S-005 only computes a successor when renumbering requires one and checks exhaustion before incrementing.
- S-006 protects input/config aliases and future output members before import, uses the shared dependency-closure destination validator for linked/completed stores, and publishes through the existing atomic-file helper. Ordinary report replacement remains intentional. No consumer-specific dependency resolver was added.
- S-007 provisions the exported server target's Boost dependency conditionally. No unjustified zstd alias workaround was introduced.

C-006 is correctly withdrawn: `src/internal/Database.hpp:83` declares an `unordered_multimap`, whose duplicate-key insertions retain both statement handles. The added ownership regression does not justify a production fix.

## Limits

This review does not claim independent runtime, real-volume, full-GUI, macOS, or performance validation. The new native restart regression tests correction persistence/retry and the caller is checked by source/compilation, as the native fix notes explicitly acknowledge. Existing volume-frame traversal passes frame-local callback counters within each page; the refill fix should not be described as changing that pre-existing behavior. Test results belong in the coordinated validation record.
