# R10 independent finalization decision

Final frozen commit: `2748bd5c1`. Reviewed `433eed807` against R9's `8859fc945`, then separately challenged the paired-boundary guard in `2748bd5c1`. Reconciled the R10 consumer and storage ledgers; core/store runtime code is unchanged from the R9 review. All implementation evidence was inspected at the named commits with `git show`/`git diff`.

## Verdict: NO STOP — conclude this bounded review

No unresolved actionable supported-v21 correctness defect is established at `2748bd5c1` in the reviewed scope. R9's supported parent-source mismatch is repaired for newly produced expansion datasets. R10's acceptance of uncertified older expansion evidence is now rejected before a new paired solve can publish a misleading manifest. Neither unavailable historic provenance nor unexecuted test combinations are being represented as new correctness defects.

This is a review-gate disposition, not a substitute for the separately recorded build/test gate and not a claim that all research workflows or arbitrary repository code are bug-free. Do not launch another unrestricted review cycle merely because this finite audit cannot prove the absence of all defects. Reopen the affected inventory if a concrete supported counterexample or a later code change appears.

## Per-finding disposition

| Finding | Final decision | Evidence |
| --- | --- | --- |
| R8 C-001, legacy creation order after override | Closed for the reported supported transition; carry R9 disposition forward | No store/runtime delta since the R9 source review. Original-owner same-second creation evidence refines the stamp before override; reader and writer use the same key. |
| R8 X-001, paired producer lacks baseline manifest | Closed | Active producer emits the digest and source position expected by joint assembly. Report-only remains separate from new solve validation. |
| R9 X-001, expansion stamps current source over saved parent tensors | Closed by `433eed807` | Expansion validates the original parent tensor sidecar signature before loading source geometry, stores its verified digest/pin, and supplies the pin to catalog and ordinary polyline reads. |
| R10 X-001, paired producer accepts uncertified legacy expansion stamp | Closed by `2748bd5c1` | Pairing requires a nonempty saved `fiber_source`, rejects its absence with regeneration guidance, verifies the saved digest at that position, and forwards the same position to the prepared baseline. No current-position fallback remains at this boundary. |

## Independent challenge of the final evidence chain

The supported parent producer `fiber_uv_layout.prepare` records `signature.fibers` and `signature.fiber_source` alongside its tensor output. At the final commit, `fiber_neighbor_expansion.run` requires that sidecar and calls the shared `verify_fiber_signature` before catalog or source-curve access. An edit made before fresh expansion cannot be blessed by hashing the new current store: the comparison starts from the saved parent evidence. The selected position is then supplied to `fiber_catalog` and `load_curves`; a later publication is rejected by the shared source guard rather than silently switching geometry. The exported `inputs.json` retains that verified digest and pin with hashes of the parent tensor and sidecar.

Paired preparation now requires that pin rather than interpreting its absence as permission to capture a new one. It validates the saved digest at the pin, derives the prepared graph from the expanded tensor arrays, and writes both fields into `prepared/inputs.json`. Joint assembly validates and uses those fields for ordinary parent geometry, keeps its separate GT source verification, and validates retained combined-store evidence on reuse. Direct joint and split-search joint/selectable arms share that consumer. These steps close the actual stale-parent example used in R9 and its uncertified-old-expansion continuation from R10.

The final regression source exercises both rejection of a digest-only expansion manifest and forwarding of an existing certified pin to the joint request. The expansion regression source verifies that parent evidence rejection happens before sampling. These are focused mocked boundary tests, not a full GPU expansion-to-joint replay; this adjudicator did not execute them.

## Explicit boundaries and accepted limitations

- Historical preparation that never recorded original source evidence is unsupported for a **new** solve and must be regenerated. Failing closed is the intended contract, not a new data-conversion feature request. An artificially forged sidecar or manually relabeled tensors is not a supported producer counterexample.
- Explicit joint `report_only` branches before new parent evidence checks. This review establishes no report-only legacy compatibility regression; new preparation/solve paths are not substitutes for report-only dispatch.
- Research seed quality, GT-assisted selection strategy, solver convergence, and provenance of independent exploratory workflows beyond the reviewed source-geometry chain are not newly established v21 correctness regressions. They were not silently certified by fixing the ordinary parent source boundary, and do not expand this gate into an unlimited numerical/research audit.
- Legacy creation chronology that was never retained, or was discarded before the supported override transition, remains a deterministic documented fallback. The final workflow-only commits do not change storage placement, schema, exact identities, checked writes, history policy or transport.
- Local/referenced, history-on/off, packed/sharded, hosted/GUI and failure-injection runtime combinations remain distinct verification cases. The R9/R10 storage and consumer ledgers disclose which were only source-traced. Missing runtime coverage alone is not an observed defect.

## Verification attribution

This adjudication performed frozen-source/diff inspection and a documentation whitespace check only. No installation, build, test suite, GPU workload, GUI session, real-scroll run, macOS run or performance measurement was executed by this reviewer. Runtime results must be attributed to the separate final validation record. Production changes were made by the implementation lane; this file only records the independent gate decision.
