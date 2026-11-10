# R7 independent adjudication

Reviewed snapshot: `85874dcfd`; rechecked against `da50b2f4f` (the additional checked Python restore fix). Comparison baseline: `1b32f031b`. This is source/diff adjudication, not runtime reproduction. The restore fix does not close any candidate below. Final rescan after both review lanes completed found C-001, C-002, S-001, X-001 and X-002; no additional candidate files. Subsequent in-progress fixes are not adjudicated here.

The review follows `review-contract.md` and the shared resolution, checked-write and evidence-provenance rules in `coherent-store-api-cleanup.md`. “Pre-existing” is attribution, not an automatic exclusion from v21 release scope. No production changes were made here.

| ID | Decision | Attribution and release relevance |
| --- | --- | --- |
| C-001 | Confirm compatibility regression; retain explicit acceptance/fix decision | R6 changes legacy history-on ordering. Same ordering-policy family as S-001, but a distinct legacy compatibility case. |
| C-002 | Confirm, P2 | New Python/direct-store regression; pre-existing session sibling. Shared resource no-op validation is incomplete. |
| S-001 | Confirm, P2 | R6 ordering change exposes an unmigrated current initial-publication writer. Same shared ordering contract as C-001. |
| X-001 | Confirm, P2 | Pre-existing selected-workflow provenance omission, incompletely closed by R6. Relevant v21 consumer boundary, not an unrelated solver issue. |
| X-002 | Confirm, P2 | Pre-existing joint-workflow provenance omission, incompletely closed by R6. Relevant v21 consumer boundary, not a new numerical/solver regression. |

## C-001: legacy ordering compatibility

At the baseline, `Project.cpp::addedAt` selects the oldest retained publication timestamp before falling back to `createdAt`. Current `sortByAdded` compares only creation strings and UUIDs. Two legacy local history-on resources created in separate same-second transactions therefore have distinct baseline order evidence but identical current sort keys except UUID. Reverse UUID order proves a changed result without an intervening write. `projectResources` exposes the order and `projectRecord` selects its first result; `setProjectName` and `copyProject` consume that selection.

Challenge: R6 explicitly documents its whole-second/UUID policy. This is not an accidental deviation from that implementation policy, and adding a history lookup naively would restore the override-order bug R6 fixed. Nevertheless, documented implementation intent alone is not user acceptance of changing existing first-resource/project selection. Retain this as a compatibility decision, not as a claim that all coarse timestamps ever had recoverable ordering. Confirmed regression is specifically the retained-history case. Fix/accept this together with the ordering contract, with separate legacy and current-write tests.

## C-002: incomplete resource no-op validation

`resolveProjectResourceIdentity` returns a currently occupied new deterministic UUID without checking its tuple. `prepareResourceWrite` selects that occupant; `ResourceWrite::unchanged` checks only retired state, tags and JSON metadata. Public generic object creation permits explicit UUIDs, so no hash collision is required. A wrong-kind occupant with matching tags/metadata returns success while remaining absent from `projectResources(kind)`. This applies to both native session and direct-store helpers, hence Python's current binding.

Baseline Python used `upsertObject`, whose shared `RecordSql.cpp::upsertObjectImpl` rejects an immutable kind change before its no-op comparison. The baseline session already used the incomplete early comparison. `da50b2f4f` changes direct writes to a checked request but retains this early return, so it does not repair the defect. Validate resource kind/storage location/owner identity in shared preparation before deciding no-op, covering both entry points and inherited occupants. Do not fix only the Python adapter. The candidate's old `WriteTxn` helper signature describes the frozen snapshot, not current HEAD.

## S-001: current initial publication still writes coarse stamps

`InitialShardImport::appendObjects` delegates through `appendObjectRefs` to `importInitialObjectRefs`; `internal/InitialObjects.cpp` prepares arbitrary object records and passes one whole-second `nowIsoUtc()` batch stamp to `insertObjectRow`. Neither this entry point nor its normalization limits records to geometry kinds. Ordinary `RecordSql.cpp` upserts now request fractional stamps specifically for projects/resources. Consequently a first initial resource and a later ordinary resource in the same second sort in reverse, because `.` precedes `Z` lexicographically. History-off initial storage does not invalidate the case: the baseline used the coarse fallback before the later ordinary history publication.

This is not merely legacy ambiguity: a currently supported creation boundary manufactures the legacy form. Share creation-stamp policy at both insertion boundaries; test projects as well as resources, and both temporal orders. Plain/packed/reference routing preserves the record's stamp and does not independently fix the ordering. C-001 and S-001 share an ordering-policy work item, but fixing writer precision alone does not close C-001.

## X-001: staged selected evaluation loses its input pin

`fiber_selected_dataset_eval.run` captures a source version and supplies it to selected-coordinate verification, splitting and direct diagnostic sampling. It writes `tensors.npz`, `inputs.json`, and `init/config.json`, not `tensors.json`. Its optional soft stage calls `soft.run_prepared(stage_config)`; `initialization_config` substitutes the selected tensor path and removes sample/provenance overrides. `fiber_soft_winding.run_prepared` looks only for the adjacent tensor JSON stamp and otherwise passes `source_version=None` to `prepare_samples`, which live-reads polylines. Thus an edit after direct sampling but before a configured soft stage is reachable and is not rejected by subsequent source-ID/arc-array equality checks.

The same missing transport exists at the baseline, but R6 explicitly migrated these sibling reads and the shared prepared entry point. The outstanding issue is loss of v21 source evidence across that boundary. Consolidate the shared prepared-input provenance transport, retaining fail-closed checks for every staged caller; do not treat a fresh pin taken in the stage as validation of old tensors.

## X-002: joint assembly validates only its already-combined output

`fiber_joint_gt_eval.run` loads saved baseline/GT tensors, hashes those files plus a live GT digest, then live-reads both parent stores to create `fibers.vcobj`. It neither includes the baseline store in the signature nor supplies selected source versions to either read. An existing combined file is reused without verifying that its geometry is the content represented by the input signature. R6 then captures a complete source version for that combined file and correctly passes it to split/sampling; this protects only later changes, not initial assembly from mismatched parent evidence.

Same-UUID geometry changes can preserve IDs and valid arc bounds, so later array identity and bounds checks do not prove equivalence. This omission exists at `1b32f031b`; it is not introduced by the new downstream pin. It shares the evidence-chain invariant with X-001, but has a distinct parent-to-combined assembly boundary and must retain its own fresh/reused-output tests. Validate each parent against the source evidence underlying its tensors, then carry those selections through assembly and persisted reuse verification. Merely hashing current parents on first entry does not prove old tensors used those versions.

## Closure requirements and limitations

Three root-cause work groups remain: project/resource ordering (C-001/S-001), resource identity before no-op (C-002), and complete workflow evidence chains (X-001/X-002). There are five separately testable boundary cases; grouping must not erase their acceptance criteria.

No candidates above are rejected as genuinely unrelated, and none are claimed reproduced here. Required runtime regressions are the candidate fixtures at the listed sibling boundaries, not an inference from a broad passing suite. No build, install, real-volume run, macOS check or performance measurement was performed for this documentation-only adjudication. Source checks used `git diff 1b32f031b HEAD -- <affected paths>`, `git show 1b32f031b:<path>`, and direct inspection of the named implementations.
