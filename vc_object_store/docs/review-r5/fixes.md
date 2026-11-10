# R5 implementation and verification

Baseline: `bea32d5d8`. The [review contract](../review-contract.md) governs closure. No store schema change.

| IDs | Shared boundary and affected callers | Regression coverage |
| --- | --- | --- |
| P-003 | One native anchor codec for Python, stored records, requests and hosted wire; Python only adapts types and return presentation. | Python ndarray/tuple/None round trip; native and hosted relationship suites. |
| C-001, C-007, N-001 | Preserve source validity through transformed read and inverse edit; explicit point dependencies consult selected and prospective masks. | Transformed negative-coordinate edits, sentinel deletion, masked anchor creation and mask updates. No interactive GUI run. |
| C-002, S-010 | Normalize UUIDs at in-memory paging boundaries, consistently with binary SQL lookup. | Local length and volume cursor tests; hosted suite. No dedicated hosted uppercase-plus-concurrent-update probe. |
| C-003, S-004 | Extension protection follows the logical store. Copy uses the existing sealed-source overlay path for single immutable shards. | Referenced typed extension refusal and plain/packed one-shard copies. |
| C-004, C-005 | Checkpoints name only staged requests. Checked edits use normal dependency validation for local history-off edges and retain inherited observations through their owner. | Durable and mixed-policy suites; exact checkpoint interleaving is source-reviewed. |
| C-006, S-002, S-003 | Catalog updates patch owned fields; remote readers share geometry-only loading and checked file publication. | Catalog metadata preservation and loopback legacy identity; no disk-close fault injection. |
| N-002 | A failed correction save vetoes close/switch before teardown. | Consumer compilation checks, not an interactive GUI conflict test. |
| P-001, P-004, P-005 | Frozen fiber evidence stays pinned across UV preparation, subset and evaluation. Spiral refuses historical replay with unprovable inherited graph membership. | Frozen-source and inherited replay checks. No production datasets. |
| P-002 | Shared fiber writer owns atomic replacement and unique staging. | Existing tracing re-export path; no forced simultaneous export interleaving. |
| S-001, S-005, S-006 | Refuse writable hardlink aliases; publish conversion sidecars before replacement; guard caller-owned candidate batches with the shared savepoint. | Ownership, invalid conversion report directory, and catch-and-commit rollback tests. |
| S-007, S-008, S-009 | Validate port before narrowing, normalize project resource times, and use unambiguous resource keys with legacy lookup. | Resource collision regression and existing project/server suites. |

Validation commands: `bash local/scripts/run_vcobj_pr_tests`, `bash local/scripts/check_vcobj_consumers`, and `bash local/scripts/run_vcobj_consumer_checks`. The focused fiber pytest run covered H/V preparation, UV layout, reference subsets, reference comparison, and trace-store publication (41 passed). The native consumer fixture also verified valid negative coordinates through the actual SurfaceCatalog adapter.

Final PR fixture: 53 native tests, 217 Python store tests, 122 Spiral tests and 14 Lasagna tests passed. Consumer syntax and runtime fixtures and `git diff --check` passed.

The existing optional SSH and real-data checks are not required by these fixtures. This is a Linux validation; no full GUI run, full-volume solve, or macOS test was run. The pre-existing Spiral nanobind shutdown diagnostic is not part of the anchor binding.
