# Single-reference review and simplification

Baseline: `38c72d972`. The [specification](../versioned-reference-contract.md) requires one complete reference throughout internal calls, with extra application state beside it. This pass reviews that representation invariant only; it does not reopen unrelated solver or storage features.

Discovery ledgers: [native](native.md), [Python consumers](python.md). [Astra adjudication](adjudication.md) distinguishes internal alternate formats from necessary external/schema serialization. Candidate files retain their original evidence; this tracker records implementation and actual verification.

| Candidate | Shared fix | Status |
| --- | --- | --- |
| P-001 | Removed Lasagna tuple APIs and reference-to-text round trips. Service/config/geometry/artifact paths carry the same dictionary; native external decoder handles current UUID or saved textual/JSON input | Implemented; source-approved |
| P-002, P-003 | Spiral pins and project umbilicus retain complete references. Old numeric pins decode once with their recorded owner; genuinely ownerless point snapshots cannot select objects | Implemented; source-approved |
| N-001, N-003 | Native pins embed `ObjectVersion`; GUI history rows keep typed selected states | Implemented; source-approved |
| N-002 | Typed archive-key/reference variant at the storage layer; selected graph paths carry `ObjectRevision` directly. Removed unused graph adapters and the unused public `versionReference` conversion method | Implemented; source-approved |
| N-004, N-005 | Shared native scalar codec; new receipts use standard dictionaries and decode saved array-form receipts only at ingress. Commands, drafts, selections, bindings and GUI exports use the shared codec | Implemented; source-approved |
| N-006 | Overlap cache key contains `ObjectVersion` beside its geometry revision; comparison belongs to the shared value type | Implemented; source-approved |

Compatibility includes current UUID discovery, existing legacy CLI/JSON selectors, retained exact references after overlay/deletion, and history-on/off. No schema change or extra persistent identity structures. Use the already approved runners; do not install dependencies or invent a new fixture command.

The obsolete **internal APIs** are deleted, not retained as fallback paths. Compatibility decoding applies only to existing saved inputs or explicit external selectors. A legacy point-only umbilicus has no recoverable owning reference: it remains labeled provenance of supplied points, never a lookup identity.

The full suite exposed two existing Spiral race tests when complete caller references replaced numeric read guards. The shared native request/preparation merger now reports differing snapshots as `TransactionConflict`, matching changes after preparation; it still rejects the commit and preserves all complete-reference guards. No consumer-specific exception-string workaround was added. The native link regression exercises a change between caller selection and detector preparation.

Final caller-closure grep found a remaining N-005 producer in `SegmentationLasagnaPanel`: the active selected surface was still emitted as flattened fields. It now emits the shared serialized reference as one component. UUID-only current-discovery requests in that panel and the context menu remain intentional discovery, not another versioned-reference format.

[Validation and exact production/test line counts](validation.md) record the approved-runner results separately from [Astra's final source review](final-review.md).
