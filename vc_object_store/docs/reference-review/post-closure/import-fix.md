# PC-A01 metadata-import identity reservation

Baseline: `b73a31f33`. Astra's accepted counterexample and scope are in `astra.md` and the post-closure README. Preserve schema21, binary UUIDs, explicit restore semantics, live same-content idempotence and conflicting-content refusal. Metadata re-import is not an implicit restore operation.

- [x] Route same-kind and cross-kind import identity admission through existing tombstone-aware `objectSource`/`edgeSource` resolution, including referenced owners; add no SQL directory or proxy rows.
- [x] Keep the operation-wide savepoint; add a test for caught import error followed by unrelated outer-transaction mutation/commit (runtime pending).
- [x] Add object and edge deleted-identity regressions, local history on/off and all referenced source/overlay policy pairs; cover same-kind and cross-kind collisions (runtime pending).
- [x] Preserve live idempotence/conflict behavior and existing explicit low-level recreation/restore tests; extend idempotence checks to inherited live rows (runtime pending).
- [x] Main agent ran the approved fixed runners; final native generation/identity regressions pass. See [final verification](fixes.md#commits-and-verification). No competing build or commit in this lane.

`Snapshots.cpp` now uses existing deleted-aware scope resolution before same-kind low-level insertion and for cross-kind admission. Live content comparison is unchanged. There is no extra local SQL check, identity directory, history-off branch, schema change or alternate restore API. The existing operation savepoint is unchanged.

`test_object_store_generations.cpp::metadataImportReservations` covers two local policies and all four source/overlay policy pairs. Each rejects a live content conflict, a deleted object, a deleted edge and both cross-kind deleted UUID substitutions. A new object precedes the rejected member, and a new edge precedes edge rejection; after catching the exception the caller creates unrelated data and commits. Assertions check that prior imported rows are absent, unrelated work is present, tombstones keep state/version/original owner, and live same-content import preserves owner and commit sequence. Existing `metadataImport` and explicit deleted-object/edge recreation tests remain intact.

No build/test execution is claimed by this lane; main integration owns the fixed approved runner. `git diff --check` passes.

Integration follow-up: `56dc9c792` commits this fix. The parent completed the local/referenced matrix in the generations executable and the full native suite; runtime-pending annotations above describe the earlier handoff, not the final status. The shared import rejection preserves both the outer-transaction unrelated work and original tombstone identity/state/version in those regressions.
