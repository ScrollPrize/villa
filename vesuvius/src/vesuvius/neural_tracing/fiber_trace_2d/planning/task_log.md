# Task log: separate winding value and sign-hardness constraints

## Starting point

- Dominant signed observations already have a separately configured hard or
  finite sign penalty. The signed `abs(delta-target)` residual is intentional;
  the extra sign term is the separately weighted high/hard reversal rule.
- Final agreement collapses H/V relation, magnitude, and sign into one
  infringement bit per physical observation.
- Reference benchmarking collapses magnitude and sign into one candidate match
  and changes behavior when the tested magnitude weight is zero, so the recent
  five-weight tuning objective is not comparable across scenarios.
- Current defaults are magnitude `0,4,2,2,1`, finite sign cost `44`, and aligned
  hard signs for both dominant relation types within 30 degrees.

## Deviations

- None.

## Plan review

- Separate structural evidence presence from weight-dependent BP enablement;
  benchmark denominators use the former and BP connectivity uses the latter.
- Retain a dominant-parallel winding value when a signed estimate is absent,
  using the available unsigned target without adding sign hardness.
- Use exact finite sign coefficient `cost * relation_weight * decision_conf *
  normal_conf`; zero disables both finite and promoted-hard sign evidence.
- Audit all solver paths, not only final pairwise energy, for signed winding
  value plus separately weighted sign hardness.
- Generate the BP-consistent reference candidate from the signed winding target
  and list the extra sign-hardness judgment independently.
- Keep degree-scaled Defect incidence at one physical measurement.
- Extend deterministic local search from five to seven tagged coordinates and
  report the corrected baseline before tuning because old totals are not
  directly comparable.

## User clarification

- A sign flip must remain an error in the ordinary winding-value constraint.
- The separate sign constraint is the additional high/hard loss that makes a
  reversal forbidden or substantially more expensive.

## Implementation

- Preserved the signed winding-value residual in fixed, adaptive, alternating,
  decoded-energy, projection, gauge, and reference-inference paths.
- Added independent perpendicular and parallel sign-hardness multipliers and
  structural-presence flags. Zero sign weight disables only the additional
  finite/hard term, not the ordinary signed winding residual.
- Split solver agreement, evidence summaries, reference accuracy, and factor
  CSV output into winding-value and sign-hardness items.
- Extended exhaustive and zero-aware local weight search from five to seven
  dimensions.

## Validation

- Release build: `vc_fiber_trace_chunk` and
  `test_fiber_trace_winding_bp` completed successfully.
- Focused winding BP tests: 70 cases passed.
- The focused tests cover signed reversal with independent sign weight, zero
  sign weight, finite sign scaling, hard promotion, unsigned parallel fallback,
  split agreement, and fixed benchmark denominators.

## 1024 tuning result

- Release command:

  ```bash
  volume-cartographer/build/bin/vc_fiber_trace_chunk direction-ablation /home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/crop_traces.zarr --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --output /tmp/winding-sign-split-tune/fibers --direction-dominance 0.9 --piece-length 512 --bp-only --bp-inference sum-product-mixed --quality-fraction 0.25 --winding-fixed-phase 0.5 --winding-fixed-scale 0.822 --winding-fixed-orientation --bp-message-iterations 500 --reference-fiber-dir /home/hendrik/business/aiconsulting/vesuviuschallenge/data/test_datasets/2026-08-28_fiber_stack --reference-fiber-tag hendrik_crop1 --parallel-winding-cutoff 0.5 --split-continuity hard --winding-hard-signs both --winding-hard-sign-angle 30 --winding-normal-confidence linear --winding-decision-confidence cosine --winding-weights 0,4,2,2,1 --winding-sign-weights 1,1 --winding-weight-search-local --winding-sign-cost 44 --winding-defect-cost 100 --bp-temperature 1.25
  ```

- Dataset: `data/workdir3/crop_traces.zarr`, retained quality fraction `0.25`,
  reference tag `hendrik_crop1`, fixed phase `0.5`, fixed scale `0.822`, hard
  split continuity, 30-degree aligned hard-sign gate, cosine decision
  confidence, linear normal confidence, sign cost `44`, Defect cost `100`, and
  temperature `1.25`.
- Corrected starting tuple: winding `0,4,2,2,1`, sign `1,1`; result `6/8`
  exact reference windings and `3141/3953 = 79.4586%` reference items.
- Zero-aware local search evaluated 111 scenarios and accepted five moves:
  sign perpendicular `1 -> 0`, perpendicular next `0 -> 1`, perpendicular far
  `4 -> 2`, perpendicular next `1 -> 0.5`, and parallel same `2 -> 1`.
- Selected defaults: winding `0.5,2,1,2,1`, sign hardness `0,1`; result `8/8`
  exact reference windings and `3562/4061 = 87.7124%` reference items.
- Final class accuracy was perpendicular winding `1456/1875 = 77.653%`,
  perpendicular sign `1828/1875 = 97.493%`, and parallel-same winding
  `278/311 = 89.389%`. The `0.5` parallel cutoff admitted no parallel-other or
  parallel-sign reference items, so the parallel sign default remains `1` and
  was not identified by this run.
- The selected perpendicular sign weight `0` removes only its additional
  high/hard penalty. Perpendicular winding weights remain positive, so the
  ordinary signed residual still penalizes every reversal.
