# R9 independent finalization gate

Frozen commit: `8859fc945`. The R8 fixes challenged are `e0ca7f04a` and `76924b4c8`; source was inspected with `git show` at the frozen commit, not inferred from a later worktree. The vcobj review contract and its evidence-provenance invariant govern this gate. This is a source-backed verdict, not a runtime reproduction or a claim that all repository behavior was tested.

## Verdict: STOP finalization

R9 X-001 is an actionable v21 consumer provenance defect in the immediate supported expansion → paired preparation → joint solve path. The new manifest exists, but its digest need not identify the source used to derive the tensors. Do not finalize the release audit as clean at `8859fc945`. Fix this evidence boundary and verify the affected caller chain; unrelated missing test coverage and unverifiable old artifacts are not independent blockers.

Final lane reconciliation: `core.md` and `storage.md` report no additional actionable findings; `consumers.md` records X-001 as its sole candidate. The final candidate rescan contains only X-001. The stop verdict is specifically for that confirmed consumer finding, not for the ledgers' disclosed test gaps.

## R8 fix challenges

### R8 C-001: closed for the reported supported transition

`internal/RecordOverrides.cpp::createRecordOverride` now refines a legacy project/resource creation stamp while the source owner is still selected. `detail::retainedCreationPublication` reads that owner's retained version-1 publication; `refineLegacyCreationStamp` accepts it only in the original creation second. The override row receives the refined stamp, and ordinary `RecordSql.cpp` updates do not replace `created_at`. `Project.cpp::sortByAdded` uses the same refinement for an unoverridden legacy row. Therefore A's key before and after the first override is the same in the R8 fixture, whether the destination retains history or not. Subsequent overrides carry the already-precise stamp unchanged.

This is storage-owned preservation at the ownership transition, not consumer SQL or a change to `objectHistory`'s winning-owner contract. Projects and all resource kinds share `preciseObjectTimestamp`; native, Python and session listings share the sorter. Unavailable historical creation evidence remains a documented deterministic fallback, not guessed chronology. Previously materialized coarse overrides without recoverable local creation evidence are not a new regression established by this delta.

The added native test checks same-second refinement, different-second rejection and missing evidence. It is not an end-to-end runtime proof of the overlay transition; no such execution is claimed by this adjudication. No new actionable defect was established in this fix.

### R8 X-001: missing-manifest boundary closed, provenance closure incomplete

`fiber_neighbor_pair.run` now reads the expansion dataset's `inputs.json`, verifies its `fibers` digest, and writes that digest plus a source pin into `prepared/inputs.json`. `fiber_joint_gt_eval.run` consumes those fields and supplies the selected source to geometry reads. The new mocked producer test checks that the joint request sees the manifest. This fixes the R8 absent-file failure for the active producer.

The direct documentation now distinguishes new solve evidence from report-only use and tells users to regenerate unverifiable preparation artifacts. Missing baseline/GT evidence has actionable errors. `report_only` still branches before the new solve requirements. Requiring valid provenance for a new geometry-consuming solve is correct; refusing unsupported old evidence is not itself a release defect.

However, transporting and verifying a digest is insufficient if its producer did not tie it to the tensors. That remaining defect is R9 X-001, below; the mock's synthetic `saved` digest cannot establish that relationship.

## R9 X-001: confirmed, P2, release-relevant

At the frozen commit, `fiber_neighbor_expansion.run` hashes `settings['parent_tensors']` and separately computes a **current** `fibers_digest(settings['fibers'])` for `inputs.json`. It never validates the saved source signature accompanying those parent tensors. Its final `tensors.npz` is an induced subgraph of the loaded parent, not tensors recomputed from the geometry at the newly recorded digest. Catalog/polyline reads also lack a carried source pin.

Choose a valid, currently supported parent tensor artifact produced by `fiber_uv_layout.prepare`, then publish a same-UUID fiber edit before starting a fresh expansion. Preserve usable arc ranges and selection/support so bounds/ID checks do not fail. Expansion loads old graph evidence and records the new geometry digest. Paired preparation verifies that new digest, promotes it to the prepared baseline manifest, and joint assembly reads new geometry under the resulting pin. No race, missing legacy artifact, hash collision or malformed input is required. The free arm's selected-coordinate comparison can also pass because both compared polylines were materialized after the edit.

Independent source evidence: `fiber_uv_layout.prepare` emits the original digest and `fiber_source` in its tensor signature; expansion ignores that signature at input, retains parent-derived graph arrays at output; paired checks the expansion digest only; joint checks that same digest/pin and uses it for reads. Hashing the parent tensor file and current store separately does not prove that the tensors were derived from that store state.

Attribution: the unverified expansion parent existed before `76924b4c8`. That fix newly accepts its input digest as tensor provenance through the paired manifest, replacing the prior missing-file failure. This is a residual evidence-chain omission exposed by the supported producer migration, not an unrelated solver/numerical issue. It is the same broad root cause as R7's incomplete provenance chain, at the immediate upstream boundary now being relied upon; do not count the digest capture, pair promotion and joint consumption as three independent bugs.

Required closure: validate the saved parent tensor source before expansion, carry its selected state into all relevant geometry reads and exported evidence, and retain that evidence through paired preparation and joint assembly. Do not cure it with another fresh digest. Test unchanged input, an edit before expansion, and an edit after source selection through the producer/consumer boundary. A fail-closed error for genuinely absent original evidence is acceptable and must give regeneration guidance; report-only behavior should remain separate.

## Limitations

No tests, builds, real-volume or GPU workloads, GUI sessions, macOS runs or performance measurements were executed by this adjudicator. Added tests were inspected as source only. Independent lane runtime claims, if any, belong to their own ledgers. The gate is evidence-based and bounded to the reviewed APIs/callers, not a zero-bug guarantee or a demand for endless open-ended review. A new frozen fix must reopen the affected producer/consumer chain before changing this stop verdict.
