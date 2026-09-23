# 2048 winding 8 / 8.5 link audit

Read-only inspection of saved bp-beam-new2048-perpsign-p1low-zero-fixed artifacts. No solve rerun.

Whole-source matching: reference-local20/80 ellipse,25degree tangent agreement,4base-voxel arc sampling; native normal sampler, all valid. Approximate sampled geometry, not continuous guarantee.

- Reference8.5: source92 and1683 both100% reference coverage/trace purity;95% ellipse scales0.234/0.247. All12 pieces467–472 and7412–7417 survive in *_oracle_w_28_h.obj.
- Reference8: no complete95%/95% source equivalent. Partial candidates1610:73.64% reference coverage/75.47% purity;3656:65.09%/34.08%;1747:50.36%/48.66%. The larger crop generates different traces;1024 matching does not transfer.
- Initial consistency.csv marks6/12 reference8.5 pieces defective, but conditioned oracle output retains all12. Do not confuse initial and conditioned stages.

The reported reference8.5 estimate7.5 corresponds to published raw layer29. Actual matching source traces are horizontal layer28. Same output offset and global sign−1 means their calibrated winding is8.5: they are correctly assigned, not reversed. Verified shared publishing mapping in fiberTraceReferenceOutputWinding / publishReferencePruneArtifacts / direct reference report call.

28 hard perpendicular signs link these two complete8.5 traces to the three partial8 candidates above. All have the expected sign (canonical±0.5, global−1).16 are disabled by removed reference8-side input_defect endpoints;12 survive and preserve correct order.

Examples from source92/piece467 (raw28 horizontal, calibrated8.5):

| Factor | Other source/piece | Conditioned state | Effect |
|---|---|---|---|
|76811|1610/7096|input_defect|Removed link|
|76812|1610/7097|raw28 vertical, calibrated8|Correct +0.5 order retained|
|76821|1747/7711|raw29 vertical, calibrated7|Correct sign, stretched1.5 separation retained|

The printed reference ladder is NOT the winding of geometrically matched source fibers. calibrateFiberTraceReferenceWindings independently selects weighted[source].all.preferredWinding from reference-to-BP observations. The actual references are clamped in the conditioned solve; the report subsequently re-estimates preferred positions from cross-constraints. Same-winding parallel magnitude/sign terms are disabled, so a perfect geometric match does not automatically bind this diagnostic estimate to its source winding.

Scope: complete8.5 matches and links to reference8 candidates with>=40%coverage, not every marginal candidate. This establishes the diagnostic/source mismatch and exact retained/removed links. Individual reference-cross loss contributions tipping the diagnostic toward7.5 have not been reconstructed; no claim that these removals alone caused that preference. The prior table must be read as independently fitted reference estimates, not source-fiber assignments.
