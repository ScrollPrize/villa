# Stage2 H/V assignment and whole-fiber filtering
- User approved axial prior + H/V/rejection logits + late ternary entropy. Full plan
  hv_stage_plan.md independently reviewed before implementation. See hv_stage.md.
- Added native reusable segment quadrature helper, extraction-time sidecar and
  feature-only backfill through existing AFV loader/frame checks. Source512698
  fibers;31.26s wall;0.084sfeature computation. No constraint-search rerun.
- Stage1 penalty0.1 inward saved solution reused (2402fixed exclusions).
  Active510296fibers/7277943events; preparation37.08s.21147isolated active fibers.
- Eight managed-CUDA10000step runs: uniform/degree_mix rejection x0.1/1penalty
  xentropy0/.01. Fixed active denominator, positiveV axial prior. Shared manager
  launch factored from stage1, not copied. Native143tests;Python15tests passed.
- Independent implementation review confirmed objective and frozen exclusions;
  requested stronger source binding and sorted-ID validation, now added.
  Verified AFV hash and ALL512698fiber lengths against constraint DB provenance.
  Empty edges handled, all-excluded dataset rejects clearly.
- Axial features are an additive sidecar, not mutation of the immutable schema2DB.
  Native feature-only and ordinary extraction call the same implementation.
  Old DB lacks a historical source hash; current source hash matches original
  recordedAFV05fa4606... and lengths agree. Source files remain unchanged.
- GPU sweep completed under manager task c0ed1140-0740-47e1-b713-048b01a695c7,
  shared4GiB allocation. Approx90s/variant. All output under data/hv_stage_v1.
  Initial run:6.8587%axis-only relation disagreements reduced to0.0847%;
  stage2reject3.6091%,retain87.8366%events. Not anatomical accuracy.
- Final report/audit command: approved crop-benchmark runner with run_hv_report.json.
  Report complete:hv_stage_results.md; historical output/1000fiber visualization preserved.
- Eight10000step runs finished88.85–89.68s each,662.4MiBpeak tensor memory.
  Manager terminal/finished with no heldGPU/CPU resources. Independent saved-array
  audit passed all8runs (source IDs, frozen broken probs/labels, retained/error
  counts). Final143native/15Python tests; git diff --check clean.
- Recommended inspection candidate degree_mix_b0.1_e0.01:251658H/237800V,
  20838new broken,23240total (4.5329%). Active arc retained95.2522%, events90.9436%,
  contradictions0.05826% (parallel0.00248%,perpendicular0.31714%). Ambiguous0%
  under max-probability<.9 criterion. No ground-truth cross-family validation.
- User expected3–10%rejections based on casual inspection; recorded after these
  low-penalty runs, not used to force a rejection fraction. No further tuning.
