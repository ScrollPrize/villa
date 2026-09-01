# Task Log

- Starting revision: `c375b1977` plus the uncommitted completed winding-sweep
  implementation and correspondence diagnostics in the working tree.
- This task continues the fixed 1024-crop hyperparameter evaluation. The
  unchanged baseline is class weights `8,1,2,2,1`, perpendicular hard sign,
  piece-break cost `0`, Defect cost `50`, orientation BP temperature `2.5`,
  fixed winding phase `0.5`, fixed scale `1.0`, fixed orientation, and 500
  messages.
- Previous baseline result: converged, 8/8 exact reference windings,
  1,784/2,839 matched admitted reference constraints, 1,313 active pieces, and
  47 Defect pieces. It repeated exactly.
- Previous interaction result motivating this task: adding parallel hard signs
  consistently worsened distant-magnitude variants. Audit found that selected
  hypothesis scores remain in `[0.5,1]`, normal alignment changes target
  magnitude but not confidence, and hard signs are infinite even for barely
  dominant or weakly aligned evidence.
- Independent plan review required explicit transient parallel-alignment data
  flow, missing-alignment behavior, exact finite-sign energy and zero-delta
  semantics, hard-sign behavior at zero confidence, discrete incidence rules,
  fixed-denominator reporting, a mandatory experiment matrix, and broader
  default/extraction/reference tests. The plan was updated before implementation.

## Implementation

- Added post-decision confidence modes `legacy`, `linear`, and `cosine`.
  `legacy` retains the selected normalized hypothesis score `s`; `linear` uses
  `2s-1`; `cosine` applies `(1-cos(pi*(2s-1)))/2`.
- Added normal-confidence modes `none`, `linear`, and `cosine`. `linear` maps
  alignment angle to `[0,1]`; `cosine` uses the absolute normal dot directly.
  Missing alignment is neutral under `none` and zero-confidence under either
  weighted mode.
- Perpendicular constraints retain the closest connector's absolute normal
  alignment. Parallel constraints use the deterministic median alignment over
  all accepted connector samples, including an even-count central-pair mean.
- Added optional finite sign-infringement energy. A wrong or zero predicted
  sign costs `sign_cost * decision_confidence * normal_confidence`; absent cost
  retains the legacy hard incompatibility. Zero cost removes the sign factor.
- Applied identical coefficient/sign semantics to winding BP, decoded energy,
  component connectivity, Defect incidence, diagnostics, and reference-fiber
  inference. Orientation/prepass energy and hard continuity are unchanged.
- Added CLI controls, documentation, and focused extraction, solver,
  reference-inference, missing-alignment, zero-confidence, and default
  compatibility tests.

## Fixed benchmark

All rows use the Release `vc_fiber_trace_chunk`, the fixed 1024 crop, 500
quality-filtered fibers split at 512 base voxels, eight `hendrik_crop1`
references, fixed phase `0.5`, fixed scale `1`, and fixed orientation. The
approved `/tmp/vc_direction_ablation_runner.sh` records complete logs under
`/tmp/winding_sweep`. Columns are solver status, exact reference windings,
matched/evaluated constraints, active pieces, and Defect pieces.

Baseline: `converged`, `8/8`, `1784/2839`, `1313`, `47`.

### Confidence matrix with legacy hard perpendicular sign

| Decision | Normal | Status | Exact | Right/total | Active | Defect |
| --- | --- | --- | --- | --- | --- | --- |
| legacy | none | converged | 8/8 | 1784/2839 | 1313 | 47 |
| legacy | linear | converged | 6/8 | 1559/2856 | 1310 | 50 |
| legacy | cosine | converged | 8/8 | 1637/2847 | 1309 | 51 |
| linear | none | converged | 8/8 | 1641/2858 | 1313 | 47 |
| linear | linear | converged | 8/8 | 1630/2847 | 1311 | 49 |
| linear | cosine | converged | 8/8 | 1637/2847 | 1309 | 51 |
| cosine | none | converged | 8/8 | 1641/2858 | 1313 | 47 |
| cosine | linear | converged | 8/8 | 1630/2847 | 1310 | 50 |
| cosine | cosine | converged | 8/8 | 1637/2847 | 1309 | 51 |

No confidence remapping beat the baseline. Normal attenuation was especially
harmful on this crop, so both controls remain neutral by default.

### Sign matrix at legacy/no-confidence weighting

| Signs | Cost | Status | Exact | Right/total | Active | Defect |
| --- | ---: | --- | --- | --- | ---: | ---: |
| perpendicular | 0 | converged | 6/8 | 1731/2992 | 1360 | 0 |
| perpendicular | 0.25 | converged | 7/8 | 1758/2992 | 1360 | 0 |
| perpendicular | 1 | converged | 7/8 | 1758/2992 | 1360 | 0 |
| perpendicular | 4 | converged | 7/8 | 1758/2992 | 1360 | 0 |
| perpendicular | 16 | converged | 7/8 | 1832/2992 | 1360 | 0 |
| perpendicular | 64 | converged | 8/8 | 1863/2977 | 1352 | 8 |
| parallel | 0 | converged | 6/8 | 1731/2992 | 1360 | 0 |
| parallel | 0.25 | converged | 6/8 | 1731/2992 | 1360 | 0 |
| parallel | 1 | converged | 6/8 | 1734/2992 | 1360 | 0 |
| parallel | 4 | converged | 6/8 | 1739/2992 | 1360 | 0 |
| parallel | 16 | converged | 6/8 | 1812/2992 | 1360 | 0 |
| parallel | 64 | converged | 5/8 | 1790/2983 | 1354 | 6 |
| both | 0 | converged | 6/8 | 1731/2992 | 1360 | 0 |
| both | 0.25 | converged | 6/8 | 1731/2992 | 1360 | 0 |
| both | 1 | converged | 7/8 | 1762/2992 | 1360 | 0 |
| both | 4 | converged | 8/8 | 1815/2992 | 1360 | 0 |
| both | 16 | converged | 8/8 | 1874/2992 | 1360 | 0 |
| both | 64 | converged | 8/8 | 1826/2899 | 1328 | 32 |
| parallel | hard | converged | 4/8 | 1725/2930 | 1328 | 32 |
| both | hard | message_limit | 7/8 | 1568/2502 | 1226 | 134 |

### Coordinate refinement

- Finite perpendicular costs `32,128,256,512` gave respectively
  `1865/2992`, `1828/2932`, `1801/2874`, and `1792/2850`, all at 8/8 exact.
- Finite-both costs `32,128,256,512` gave `1867/2989`, `1774/2808`,
  `1665/2647`, and `1634/2599`; the last two reached the message limit.
- Around finite-both cost 32, halving/doubling each class weight, Defect cost,
  temperature, both confidence modes, and the parallel-only sign mode produced
  no better authoritative row. The only nontrivial improvements were Defect
  cost 100 or temperature 1.25, each `1870/2992` at 8/8.
- With Defect cost 100 and temperature 1.25, sign costs
  `16,24,32,40,44,48,52,56,64` produced matched counts
  `1874,1870,1870,1875,1875,1875,1873,1873,1873` from the same 2,992 evaluated
  constraints, all converged at 8/8 exact. Defect costs `75,100,150` were
  identical on this plateau. Temperatures `0.625,0.9375,1.25,1.875,2.5` were
  also identical at sign cost 48.

Selected experimental row: decision `legacy`, normal `none`, finite signs
`both`, sign cost `44`, weights `8,1,2,2,1`, piece-break cost `0`, Defect cost
`100`, and temperature `1.25`. It converged with 8/8 exact references,
`1875/2992` matched constraints, 1,360 active pieces, and zero Defect pieces.
Two repeat runs reproduced the same solver state counts, residual, objective,
reference estimates, and benchmark totals. Legacy CLI defaults remain unchanged;
the selected row must be requested explicitly.

## Validation

- Release build:
  `cmake --build volume-cartographer/build --target vc_fiber_trace_chunk test_fiber_trace_winding_bp test_fiberlet_crop_trace test_lasagna_normal_alignment -j 16`
- `volume-cartographer/build/bin/test_fiber_trace_winding_bp`: 55 cases passed.
- `volume-cartographer/build/bin/test_fiberlet_crop_trace`: 81 cases passed.
- `volume-cartographer/build/bin/test_lasagna_normal_alignment`: 8 cases passed.
- `volume-cartographer/build/bin/vc_fiber_trace_chunk --help`: new controls
  advertised with the intended neutral defaults.
- Final-build baseline and selected benchmark reruns reproduced
  `1784/2839` and `1875/2992`, respectively.
- `git diff --check`: clean.

No implementation requirement was deferred. The confidence modes remain
experimental and neutral by default because neither attenuation variant beat
legacy confidence on this single fixed crop.

## Follow-up default promotion

- At user request, promoted the selected row to the shared/CLI defaults: both
  sign classes, finite sign cost `44`, winding Defect cost `100`, and
  orientation BP temperature `1.25`. Decision `legacy`, normal confidence
  `none`, class weights `8,1,2,2,1`, and piece-break cost `0` remain unchanged.
- Added `--winding-sign-cost hard` so the former strict sign behavior remains
  directly selectable after finite cost `44` becomes the omission default.
- A Release benchmark with all promoted parameters omitted reproduced the
  selected row: converged, 8/8 exact reference windings, `1875/2992` matched
  constraints, 1,360 active pieces, and zero Defect pieces. One of the original
  1,361 pieces was removed as disconnected before inference; it was not marked
  Defect.
- The explicit legacy override (`perpendicular`, `hard`, Defect cost `50`,
  temperature `2.5`) reproduced `1784/2839`, confirming the old behavior
  remains selectable.
- Rebuilt the Release app and focused tests after promotion. Winding BP passed
  55 cases, crop tracing passed 81, and normal alignment passed 8. CLI help
  reports the promoted values.
