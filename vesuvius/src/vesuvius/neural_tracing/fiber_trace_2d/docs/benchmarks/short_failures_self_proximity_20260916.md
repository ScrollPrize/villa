# Short failure benchmark: self-proximity prefilter

Full50originalGTfibers, eligible CP-bounded/non-Lasagna spans only. Compare
different eligible pieces of the same original fiber too. Exclude entire original
fiber on any hit, before failure selection. Normal manifests from frozen best
u22000 evaluation; current40normal/80tangential acceptance ellipsoid in basevoxels.
Ignore original arclength separations<500basevoxels; sample spacing<=4basevoxels.

Native C++ uniform-grid segment broad phase, sphere80bounds, shared normal-aware
threshold metric. Minimize metric along allowed candidate segment intervals,
including500separation cutoff. This is sampled screening, not certified continuous
curve intersection. A flag establishes geometric self-proximity, not annotation error.

| Scroll | Excluded original fiber | Original GT arcs, base voxels | Ellipsoid error ratio |
|---|---|---:|---:|
| PHerc0139 | kb_20260828T155910844_000043.json | 29479.721 / 43437.365 | 0.999309 |
| PHerc0139 | kb_20260905T035141671_000093.json | 33699.821 / 34290.240 | 0.983570 |
| PHerc0332 | dj_20260810T113807458_000001.json | 33411.180 / 47274.486 | 0.998775 |
| PHercParis4_fibers | kb_20260729T173847941_000210.json | 2745.910 / 3797.999 | 0.975746 |

All four hits have valid normals. Reports stop at the first hit per original
fiber; these are not deepest-overlap measurements.50fibers,359265indexed segments,
25581narrow-phase checks,0.570935seconds summed native index/query time (one run;
normal dataset opening/JSON orchestration excluded). RelWithDebInfo workspace
build,GCC13,x86-64-v3. No model inference or GPU work.

Machine-readable geometry/report:
`/ephemeral/hendrik/fiber3d_train2/evaluations/short_failures_200_20260916_v2/self_proximity.json`.
Run via approved `run_classic_fiber_evaluation.py short-failures --config`
`fiber_trace_3d/configs/eval_short_failures_200_20260916_v2.json`.

Remaining candidate observations:64best,213Classic,403other, before dedup or fresh
reproduction. Greedy location-only dedup with other quota50 leaves255potential
locations (51best,50other,154Classic); actual reproducibility can reduce this.
Final benchmark extraction is ongoing, not complete.

Validation:243Python passed/8skipped;23native replay,84trace,11reference cases.
