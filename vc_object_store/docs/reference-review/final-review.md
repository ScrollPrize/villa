# Final independent source review

Historical report: partial-identity compatibility decisions below are superseded by [legacy removal](legacy-removal.md) and the current [reference contract](../versioned-reference-contract.md).

Baseline: `38c72d972`. Reviewed the working-tree implementation separately from the baseline findings. This review concerns the single-reference contract, not a new schema-21 or general solver audit. No tests were executed by this reviewer; runtime results must be reported separately by the implementation lanes.

## Verdict

Source review approved for the nine identified representation problems. Both final boundary follow-ups below are now resolved in the inspected source. Runtime completion remains a separate gate. This is not a claim that every repository caller or platform has been exhaustively tested.

| Finding | Source-review disposition |
| --- | --- |
| N-001 | `PinnedObject` retains one `ObjectVersion`; geometry, attachment and active write generation remain distinct. Current/receipt pins, unsaved constructors, fiber, surface and growth consumers migrate together. Version-zero unsaved state is not parsed as a published reference. |
| N-002 | History uses a typed archive-UUID/`ObjectVersion` choice at the storage adapter. Exact graph reads carry selected revisions; endpoint checks use complete references. Internal JSON-array serialization and unused archive-selection graph/relationship wrappers are removed. Archive UUIDs remain genuine storage identities, not alternate exact-reference transport. Binary 16/56-byte encodings remain unchanged. Legacy array parsing remains only at old receipt/revision input boundaries. |
| N-003 | Surface-history items retain whole references in a typed backing map. Both View and Restore consume the same selected value; no per-role reconstruction remains. |
| N-004 | New receipts emit standard reference objects. Old array entries still decode; missing complete receipt identities retain the reload requirement. Positive-version encoding checks and unchanged inherited identities are preserved. |
| N-005 | Native scalar codecs serve wire, restore commands, drafts, selection transform entries, Python extension reads and the GUI export response, with enclosing-format validation kept explicit. |
| N-006 | Overlap caches key on `ObjectVersion` plus geometry, using shared value comparison. The selected-state public API requires published identity; this is not a new cache-specific flattened comparator. |
| P-001 | Lasagna tuple parsing/formatting and tuple selector adapters are removed. Canonical dictionaries survive service/config, base/external surface loading, grid and artifact calls. Selected surfaces retain the same reference. CLI/old JSON decode uses the native boundary; complete exact inputs do not require a current row. |
| P-002 | Spiral pins/selections retain one reference map, not synchronized numeric maps. Object UUID/version accessors derive from references. Legacy manifest/provenance decoding is confined to ingress and retains the original recorded single-store lineage. Internal `refs()` does not normalize or reconstruct. Link read guards reuse existing command-v2 complete expected references; no new identity schema or link-guard API is added. |
| P-003 | Project/service/provenance records retain a reference component beside materialized points. Ownerless legacy point data is explicitly marked `legacy_source`, never fabricated into an exact reference. Materialized source JSON preserves the complete identity, and filename identity includes selected data. GUI producer and summary consumers migrate with Python. |

## Final boundary follow-ups

1. `agent_bridge/AgentBridgeHandlers_stores.cpp` still manually serialized the four `result.version` fields, a sibling explicitly named by N-005. Requested migration to the shared scalar encoder plus Qt container adaptation. This is necessary serialization, but it should not retain a duplicate reference codec.
2. `objectVersionFromJson` initially converted the JSON version to `int64_t` before checking integer type. Requested integer-type and unsigned-range checks before conversion, avoiding out-of-range floating-to-integer conversion for malformed external input. This concerns the newly added strict decoder, not a broader legacy-validation redesign.

Both follow-ups were subsequently inspected and are resolved: the Qt response uses `objectVersionToJson`; strict scalar decoding checks object/key/type/unsigned range before converting the reference.

Earlier fix-review issues were addressed in source: leading-zero locator versions use checked decimal parsing, receipt encoding again rejects nonpositive versions, and the now-unused JSON-array serializer was deleted rather than retained for validation.

## Compatibility and simplification assessment

One-time extraction of legacy application envelope fields followed immediately by the shared native reference codec is legitimate boundary adaptation. It does not justify internal fallback rebuilding. Original single-store pins and ownerless materialized umbilicus points have different legacy meanings and are not incorrectly collapsed into current-owner lookup.

The patch genuinely removes representations and work: Lasagna tuple/text conversion stages, parallel Spiral numeric/reference state, native array-string round trips, redundant endpoint counters, and dead private wrappers. It also adds shared external decoding and explicit legacy handling. Therefore report final production additions/removals including new files separately from tests/docs; do not claim a net code-size reduction without the actual final count. Schema changes, per-object directories and payload copies were not introduced by this diff.

## Runtime evidence boundary

At review request time the root reported a successful native build, three shared input-decoder tests, and 33 focused Lasagna/config/approval tests with one optional hosted skip. Full repository runner and additional Spiral/service checks were still running. Those are implementation-lane reports, not independent runtime verification by this reviewer. Final test totals, skips, GUI syntax/runtime coverage and platform limitations belong in the handoff after completion.

## Final scoped follow-up

Source-approved: removal of `ObjectStore::versionReference(Txn, ObjectRevision)` eliminates an unused production reconstruction/revalidation adapter. Remaining repository fixtures obtain `reference()` directly from the selected revision; repository search found no remaining calls. This is source-API removal, not an assertion of compatibility for unknown external C++ consumers. Selected revisions already contain the complete reference, and no live-head fallback is introduced.

Source-approved: `Commands.cpp::mergeVersions` now reports `TransactionConflict` when caller guards and preparation snapshots disagree. Migrating Spiral from numeric read dependencies to command-v2 exact expected references exposed this pre-detection race at the shared merge boundary; classification as a version conflict restores retry behavior without weakening any guard or adding consumer string matching. The helper also merges multiple preparations, where disagreeing snapshots likewise cannot commit. Conflict details retain expected/observed numbers, and complete-reference ownership checks remain in ordinary checked-write validation. The new native regression pins a caller reference, edits before preparation, and checks a typed conflict with no commit advancement. This is an in-scope integration fix, not an unrelated transaction redesign.

The root subsequently reported 53 native and 222 Python store tests passing before this follow-up, with two Spiral race tests identifying the conflict-classification issue. A complete rerun was started after the fix. These remain reported runtime results, not tests run by this reviewer; final rerun completion must be recorded separately.

Source-approved late producer closure: `SegmentationLasagnaPanel::startOptimizationWithOverrides` now emits the active selected surface as `{type, uuid, name, reference}` using `objectVersionToJson`, instead of adding flat store/generation/version fields. Its `linkedSurfacesOf` inputs are freshly constructed current-discovery envelopes, so no stale legacy identity fields remain alongside the nested reference. The existing selection timing is unchanged. This is a direct N-005/P-001 GUI-producer closure, not a separate policy change. UUID-only helpers/model/context-menu requests remain intentional current discovery, not incomplete historical references. GUI syntax verification is a separate implementation-lane check.
