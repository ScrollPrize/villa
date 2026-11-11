# Review after receipt admission fix

Frozen production/test commit: `371961148` (`Centralize vcobj container and receipt admission`). The worktree was clean at the start. This pass is review-only: no production, test, schema or dependency changes.

## Review scope

Use the [review contract](../../review-contract.md), not a finding quota. The native and consumer lanes enumerate fresh inspection separately from reused prior evidence. Independent Astra adjudication checks candidate validity and the shared API contract. Discovery counts do not establish exhaustive line-by-line review or prove absence of defects.

- [Native store lane](native.md): routing, visibility, exact references, history, geometry, extensions, transport, commands, imports and exports.
- [Consumer lane](consumers.md): native/Python bindings and repository consumers, including deferred edits and saved evidence.
- [Independent adjudication](astra.md).

Coordinator fresh inspection covers receipt decoding/encoding/reference projection and saved replay (`Versions.cpp`); transaction header admission, command decoding and receipt exception wrapping (`Commands.cpp`); hosted submit/recovery and session pending-request handling; selection-record encoding/decoding; `StoreReader` selection/reference/endpoint APIs; the shared record and reference resolvers; geometry reference resolution and surface-grid materialization; source-path/export-destination protection; database-copy/locator relocation and override preparation; artifact closure preservation; and versioned extension snapshot decoding. Deliberate current discovery and physical storage SQL inside the native storage layer are not consumer-specific fallbacks. Non-vcobj DBM/fiber databases are outside this store review.

## Validation

The unchanged, approved, argument-free runner is `bash /home/hendrik/villa3/local/scripts/run_vcobj_review_checks`. It uses automatically cleaned disposable stores; no external service or user dataset is modified. This run completed with exit 0: eleven focused native targets, two repeat/umbilicus checks, 25 source-import tests, ten consumer runtime executables, 102 selected Python tests, all configured consumer syntax checks and 85 combined consumer/position tests passed. Existing Qt environment and PyTorch deprecation warnings remain. `git diff --check` passed. The prior full-suite result belongs to the committed fix record; it is not a fresh full-suite run for this review.

## Verdict

No additional confirmed defect or regression in the reviewed invariants. Astra independently agrees with the native and consumer dispositions. PCS-01 is confirmed fixed at the shared receipt decoder; hosted, saved replay, Python projection and session recovery delegate to it. Potential duplicate endpoint and tombstone-copy concerns were source-disproved and recorded in the native lane rather than promoted to findings. No fixes were needed or made during this review.

The reports explicitly preserve limitations: unchanged numerical kernels and whole-file ledgers are partly reused evidence; interactive GUI behavior, injected remote responses, crash scheduling, macOS/arm64 and full-volume performance were not newly tested. This is a broad store/consumer contract review, not a guarantee that the repository contains no further bugs.
