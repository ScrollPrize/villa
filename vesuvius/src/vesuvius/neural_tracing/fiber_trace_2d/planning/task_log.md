# N2 ordinary-only winding evaluation

User explicitly accepts reference links as special post-split attachments. Do not change those semantics. Earlier conditioned results are in docs/fiber_benchmark_runs/2026-09-28-reference-neighborhood.*. This task measures ordinary-only correctness and shifts for the same N2 cohort. Existing uncommitted changes belong to earlier work and are preserved, including unrelated pytest scratch. No new model inference, no component filter, no iteration-limit extensions.

## Implementation/validation
- Added explicit ordinary-only neighborhood flag with independent readout. Combined graph is selection-only; solver subsets the original ordinary report, excludes references/cross factors/fixed states/conditioned initialization. Existing topology, solver, source/piece mapping, cross-subset and readout reused.
- Ordinary graph identical across trials:548 sources,29597 pieces,42102 factors. Exact original source/piece ID sets match conditioned N2 outputs; conditioned artifacts hashed in report provenance.
- Independent plan and code reviewer found no wiring blocker. Native build succeeds;123 native tests and27 Python tests pass (including new identity/leakage checks, metric definitions, missing references, multiple gauges and ambiguity flags). Reference-driven weight searches explicitly rejected in this mode; guard added after completed trials, with no change to their commands or solve logic.
- All three approved-runner evaluations complete,109–117s command wall with three64-worker runs concurrent,500-message cap unchanged. No GPU/model inference or trace regeneration.

## Results and metric limitation
- Removing reference anchors yields28 integer gauges with readout evidence. Gauge0 uniquely covers all26 references and2826/2861 observations (98.78%);27 partial gauges retained and exported. No global aggregate fabricated from incompatible readouts, and no component removed from the solver.
- Full-coverage gauge0 deterministic rank1: D3/C10 correctness4/26=15.3846%,21offset changes,shift score19.2308%; D10/C3 correctness2/26=7.6923%,22changes,15.3846%; D10/C10 correctness4/26=15.3846%,17changes,34.6154%.
- Tied/unbounded-best reference rows in gauge0:10,14,7 respectively. Full candidate alternatives and per-gauge tables preserved. All26 gauge0 references have nonzero best-assignment hard-sign violations for each setting, so poor results are not merely a global offset.
- Correctness/shift use existing independent_metrics and ladder_shift_metrics definitions. Partial ladders have no shift score and retain denominator26 for correctness. Readout gauge sign/offset calibration uses references only after the solve.
- Reference-filtering/oracle removal percentage was not measured in these ordinary-only runs. Earlier conditioned disablement percentages are explicitly not relabeled as that metric.
- Results, ladders and commands: volume-cartographer/docs/fiber_benchmark_runs/2026-09-28-n2-unconditioned.md and .json. Raw outputs /ephemeral/hendrik/fiber_crop_bench/20260928_n2_unconditioned_v1/. No outstanding evaluations and no default tuning promotion.
