# Independent follow-up implementation review

Reviewed the uncommitted implementation over `5920625cc`, including the shared transaction representation, command codec, validation/receipt boundary, migrated producers and F-01/F-02/F-04 consumers. This is source inspection only: no tests, builds or production edits were performed by this review. Findings below describe the inspected snapshot, not later fixes.

## Blockers reported to implementation owner

Final source recheck: IR-01 and IR-02 are resolved. The descriptions below retain the original findings and intermediate recheck evidence. No source-review blocker remains; the latest regression additions await the final approved-runner rerun.

### IR-01: command-2 full references must canonicalize before map lookup

In `Commands.cpp::transactionRequestFromJson`, `selectionFrom` canonicalizes numeric-map UUID keys, but the `expected_object_references` loop uses the raw `ref.objectUuid` for both `seen.insert` and `expectedObjects.find`. `objectVersionsRequestFromJson` uses `objectVersionFrom`, which does not canonicalize. `expectObject` then canonicalizes and replaces the map entry.

Consequently, a mixed/upper-case reference UUID can bypass a conflicting numeric expectation, and two differently cased references to the same object can bypass duplicate detection. This regresses the previous explicit canonical lookup. Canonicalize each full reference before duplicate detection, numeric consistency checking and insertion. Regression cases must include a canonical numeric key plus differently cased full-reference key with another counter, and two differently cased full references to the same object. No runtime reproduction is claimed.

**Recheck:** the loop now calls `canonicalObjectVersion(value)` before `seen`, lookup and insertion. The reported bypass is closed in source. Approved-runner results for the new mixed-case regressions remain pending.

### IR-02: current-record helper cannot select the deletion state it advertises

`StoreReader.hpp` documents `currentObjectExpectation` as retaining exact tombstones, but the implementation calls `reader.currentObject`. `LocalStoreReader::currentObject` explicitly removes deleted revisions. The helper therefore yields `NoLive` instead of an exact deleted-record guard; setting `expected.deleted = current->deleted` cannot recover that state.

The underlying new validator is appropriately tombstone-aware (`objectSource`/`edgeSource` with `includeDeleted=true`). Selection must match its capability. Supply a shared reader path that selects the winning record, including deletion, independently of history retention, and use it where an exact deleted guard is intended. Alternatively, explicitly document the weaker live-or-no-live contract and avoid claiming exact tombstone selection through this helper; do not use a history lookup to manufacture history-off deletion identity.

Related fixture defect: `unifiedExpectations(HistoryMode::Off)` dereferences `store.currentObjectRevision` after deletion. `History.cpp::head` has no local archive branch with history disabled and uses visible `objectSource`, which excludes tombstones. The optional is empty on that path. The history-on local fixture can work through `object_heads`, concealing this policy difference. Obtain a deletion-aware selected state through the shared API before dereferencing; test both policies and inherited deletion.

**Recheck:** `StoreReader::currentObject(..., includeDeleted)` now has local/hosted implementations, and `currentObjectExpectation` explicitly requests deletion. `currentObjectRevision` calls `head(..., true)`; history-off exact resolution checks the selected current row's owner/generation/counter and sets its deletion flag. This closes the history-off selection gap without treating old history-off versions as retained.

However, the history-on branch of `head` still returns null when `object_heads` has no entry. Committed deletion explicitly removes `object_heads` after publishing a tombstone into `object_versions`. Therefore history-on tombstones (local or inherited) still fail selection; the prior suggestion that the local history-on fixture could work was limited to an existing head and does not hold after that deletion commits. For `includeDeleted=true`, fall back to the winning current record's full reference and resolve its retained tombstone, while preserving the default publication-head behavior. This remaining gap is a blocker.

**Final recheck — resolved:** the history-on branch now returns early on a missing archive head only when `includeDeleted` is false. Explicit deletion-aware selection falls through to the winning owner's current row, constructs its complete `ObjectVersion`, and resolves the corresponding retained tombstone through `resolveObjectVersion`. Default publication-head calls retain their previous behavior. This completes local/referenced selection for both history policies without changing history-off retention or fabricating geometry. The implementation owner reports 53 native tests passed plus the optional SSH skip, including `unifiedExpectations` with both history policies; this reviewer did not execute those tests. Newer mixed-case, replay and explicit-reader assertions require the next approved rerun.

The final source also fixes `expectObject` assigning a moved reference while using its UUID as the subscript key: assignment now copies the canonical reference, so implicit conversion to `RecordExpectation` cannot empty the key before evaluation.

The inspected geometry closure does not resurrect deleted geometry: deletion clears storage/geometry fields; `geometryHead` excludes deleted records; exact old history-off references still fail the current-row version comparison. `objectAtGeometry` can continue through retained nondeleted historical metadata, and `endpointApplies` rejects deleted endpoint revisions. Default `StoreReader::currentObject` still filters deletion, including on hosted readers. Hosted current-record transport already serializes the revision's deletion flag.

## Accepted design and inspected paths

- One `RecordSelection` with `RecordExpectation` values replaces the independently synchronized object-number/full-reference maps. Existing `ObjectVersion` and `RelationshipVersion` remain intact; `deleted` adds state, not another identity encoding. `NoLive` retains the old absent-or-deleted meaning, not permission to recreate a reserved identity.
- `mergeExpectations` retains a full handle when a matching legacy number is encountered and rejects different complete handles at the same counter. Read/write coalescing compares complete expectation values rather than counters. Separate membership/commit and mapping guards remain necessary and are retained in the inspected consumers.
- The validator rejects wrong-kind full references, mismatched key/reference UUIDs, invalid counters and non-complete deleted expectations. It checks complete owner, lineage, UUID, counter and deletion state. Read-dependency checks preserve their distinct conflict category.
- Command 3 carries full read/relationship/deletion guards directly; commands 1 and 2 are derived compatibility formats. This appropriately prevents old readers from silently accepting a stronger new guard. The source shape preserves old emitted command-1/2 serialization, but new-code round-trip assertions alone are not an independent baseline digest/replay test. IR-01 is now source-resolved.
- Receipt replay still precedes head checks. Numeric projections are confined to the transaction/receipt bookkeeping boundary; live unchanged inherited receipt inputs continue to obtain their owning record references. Deleted outcomes remain null receipt entries rather than fabricated live references. No-op publication, history-off bookkeeping and retained historical reads are not intentionally redesigned here.
- `StoredGeometry`, transform dependencies, fetch callbacks and artifact publication now carry complete selections. The artifact request keeps its selected identity across upload; pinned artifact publication retains the loaded generation and mapping guard. Attached-document loaded references still override current discovery, and growth annotations retain the editor's loaded guard without a redundant head read.
- F-02 scalar codec replacements preserve scalar wire shape and leave genuine batch/two-endpoint conversion intact. F-01 dependencies contain the full optional object reference beside availability status and preserve generation/volume/scale guards. F-04 uses the relationship's own native reference, not endpoint ownership, while the content digest deliberately continues to exclude publication pins. Legacy relationship manifests retain the explicitly weaker previous comparison, without invented owner information.

## Verification required before closure

Use only the approved runners recorded in `validation.md`. In addition to IR-01/IR-02 regressions, cover wrong-kind/key wire references, selected deleted versus no-live state, history-on/off and referenced tombstones, equal-counter owner/lineage merges, full read/relationship conflict checks, no-op inherited receipts, durable old request replay, and deferred/upload guards. The new fiber-map dependency test must actually be included in an approved runner or reported as unrun. Public Python artifact publication changed signature; document the selected-reference call form rather than leaving examples that pass UUID plus number.

The implementation remains within reference unification: no schema changes, new identity tables or referenced-payload copies were found in the reviewed diff. The identified source blockers are resolved; closure remains conditional on reporting the final approved-runner results, including the latest assertions. Source inspection is not a passing test result.

Implementation-owner closure evidence is now recorded in [follow-up validation](follow-up-validation.md): final approved-runner results and the latest coherent-store/fiber-map regressions completed. This does not change the independent review's source-only evidence boundary.
