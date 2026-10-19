# Integer fiber winding by minimum-cost circulation

Implemented on branch `fiber-min-flow`, based on requested `global-fiber-sgd-surface` commit `db07e60fd`. Worktree: `/home/hendrik/villa_super/villa-min-flow`. Original worktrees and datasets are unchanged. Current task and design contract live in [planning](../../planning/fiber-min-flow/task_plan.md).

## Formulation

One integer anchor winding `k_i` per original whole fiber. No optimized geometry, widths, probability distributions, or fiber enablement. H/V family labels do not add half-integers to positions.

Let `T_i(s)` be signed seam crossings along fiber i from its original first point to endpoint s. Let `C_ab` be signed crossings along the straight constraint connector a→b. The physical relative winding is

```
(k_b + T_b) - (k_a + T_a) - C_ab.
```

For measured integer target m, optimize `sum_e w_e * abs(k_b-k_a-d_e)` where `d_e = m_e+C_ab-T_b+T_a`.

The cut extends from the umbilicus toward image-up (negative Y), along every Z; its axis is linearly interpolated between control points. Positive crossing follows increasing atan2(Y-cY,X-cX). Paths are split at axis Z knots, so intersections are computed on piecewise-linear XYZ rather than interpolated angles. Half-open endpoint handling makes reversing a path negate its crossing count. Paths intersecting the axis are rejected as undefined.

Targets use the existing measurement convention: outward H→V permits same winding, outward V→H requires at least the next winding. Distances select larger integer jumps when warranted. Subtract `.5*(family_b-family_a)` from signed measurement calibration only; round to nearest integer with ties toward smaller outward separation, and enforce the same/next minimum. Thus the half-step is in interpreting the measurement, never a winding variable. Lasagna measurement scale is inherited (1 in these cases).

Dual circulation: minimize `-sum_e d_e*y_e`, conserve signed flow at every node, and bound `-w_e <= y_e <= w_e`. SciPy HiGHS dual simplex solves this network LP. Floating capacities preserve weights without quantization; weights are 1 for parallel and the saved `perpendicular_relative_weight` for perpendicular constraints. Geometry-dependent and uncertainty-dependent weights are absent. Integer arc costs yield integer optimal node potentials. Each solve checks the recovered integer primal value against the flow dual value, as well as flow conservation and capacity bounds. Disconnected components each have an arbitrary additive gauge.

Sign-only tests use `w*max(0, minimum - signed_physical_difference)`, the same circulation with one-sided flow bounds `[0,w]`. They do not invent distance evidence or import GT phase/Z supervision. Full solves have only the requested L1 target penalties, so a violated sign is possible at the global optimum.

## Unsigned observations

The old SGD solver allows sign=0: it fits absolute separation and masks out the ordering penalty. This is not an unresolved H/V label or tracing direction. The two large caches each have 834 unsigned events. Of these, 814 have the same integer target (zero) under either orientation and remain in the solve. Twenty genuinely ambiguous target sets are omitted: 16 parallel ±1 cases and four perpendicular same/next ambiguities. Solving their choice jointly is not a single fixed-target convex circulation. No sign has been guessed from ground truth or a fitted solution. The certificate applies to the retained objective, not those omitted terms. `solution.npz` records the full validity mask and all transport values.

## Reproduction

From this worktree:

```bash
/home/hendrik/.venv_las/bin/python vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config global-fiber-min-flow/run_seven.json
PYTHONPATH=vesuvius/src /home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=vesuvius/src vesuvius/tests/test_fiber_min_flow.py vesuvius/tests/test_fiber_uv_bucket_split.py vesuvius/tests/test_fiber_uv_layout.py
```

CPU only; no installation or GPU reservation required. Uses existing NumPy/SciPy. Source inputs remain under `/home/hendrik/villa2/global-fiber-sgd`. The surface branch predates the newer benchmark runner, so an additive adapter consumes its saved manifests and caches. `run_seven.json` sets absolute input/output paths; change these for another checkout. The runner overwrites its own output directory when rerun.

[Input manifest](../../global-fiber-min-flow/data/seven_v1/manifest.json) records resolved configs and SHA-256 hashes of tensors, raw fiber geometry, umbilicus, cross events and reference. Both large tensor hashes match those recorded by the original SGD solves. The historical source AFC database path is missing, so raw AFC sign recomputation is unavailable; this does not prevent replaying the exact hashed SGD tensor inputs. No preprocessing was regenerated.

## Seven-case results

All seven solves completed and have zero primal–dual gap. Each full small test scores 41 reference relations (35 perpendicular, six parallel); source fiber 7 is excluded from accuracy by the frozen reference definition but retained in solving. Width initialization variants collapse to the same integer problem. Dense and sparse geometry supervision are both absent, so the two sign-only cases likewise coincide. These are winding results, not passes of the old continuous geometry benchmark.

| Case | Fibers | Used/input constraints | Objective | Perp reference correct | Perp L1 | Flow seconds |
|---|---:|---:|---:|---:|---:|---:|
| full | 7 | 46/46 | 15 | 100.00% | 0.000000 | 0.0112 |
| half | 7 | 46/46 | 15 | 100.00% | 0.000000 | 0.0031 |
| double | 7 | 46/46 | 15 | 100.00% | 0.000000 | 0.0029 |
| sign | 7 | 46/46 | 0 | 100.00% | 0.000000 | 0.0027 |
| sign_sparse_control | 7 | 46/46 | 0 | 100.00% | 0.000000 | 0.0026 |
| free | 2053 | 15869/15889 | 886 | 85.39% | 0.147826 | 0.1300 |
| joint | 2060 | 21166/21186 | 1099 | 86.09% | 0.140870 | 0.1586 |

Large reference scoring uses the frozen integer offsets of GT fibers 1–6 and the existing cross-event measurements, with exact physical seam transport. It scores 4,918/5,251 cross events (575 perpendicular, 4,343 parallel); the other 333 involve reference fiber 7. This is a measurement-based reference consistency metric, not direct labeling of every ordinary fiber. The table uses a separately reported median integer gauge per connected component. Free has five components (four touch scored reference events); joint has two (one touches reference events). With a single global gauge, free perpendicular accuracy is 84.70%, L1 .179130; joint is unchanged. All fibers are retained, including isolated components.

Free parallel reference accuracy: 96.22%, L1 .037992; joint: 98.07%, L1 .019341. Reference sign errors: free 4/575 perpendicular and 9/4343 parallel; joint 3/575 and 0/4343. Internal measurement sign errors: free 22/4283 perpendicular and 0/10772 parallel; joint 24/4924 and 0/15428. These denominators include only known signs.

Raw-polyline seam transport differs from flooring cached interpolated angles at 11 free and 13 joint endpoints. Inspection reproduced the cached angular interpolation exactly; the difference comes from interpolating angles versus XYZ across long segments. The solver and reference scoring consistently use physical XYZ crossings. There are 54/56 crossing connectors in free/joint, demonstrating that connector transport is active as well as fiber transport.

The free dataset membership was previously selected with GT assistance; its solve excludes GT but the dataset is not independent held-out evaluation. Joint includes GT fibers as ordinary variable nodes and their cross constraints; it does not fix their integer labels. Frozen reference offsets are used only after solving.

### Historical SGD scores (different metric)

The saved free SGD report has perpendicular correctness 50.18%, continuous L1 .600548; joint 78.57%, .325191. These are historical context only: that scorer uses continuous fitted coordinates, a fitted GT layout, different coverage/enablement, and includes reference fiber 7. They are **not an apples-to-apples improvement estimate** against the integer table above. Width/geometry errors are inapplicable to this solver. No speedup claim against the much larger SGD optimization problem is made.

## Files and validation

- `fiber_min_flow.py`: target conversion, exact polyline seam transport, circulation and certificate.
- `fiber_min_flow_benchmark.py`: immutable dataset adapter, provenance, reference scoring, seven-case execution.
- `crop_benchmark.py`: additive dispatch through the existing runner.
- `test_fiber_min_flow.py`: brute-force random graph checks, direction reversal, gauges, self-edges, one-sided terms, H/V and unsigned targets, multiple seam crossings, seam movement, curved axis, endpoint conventions and XYZ/angle interpolation distinction.
- Each case writes `solution.npz` and `metrics.json`; consolidated [results](../../global-fiber-min-flow/data/seven_v1/results.json) and manifest are retained.

Optimality certifies this fixed-target objective. It does not establish physical correctness of measurements, uniqueness of labels, or constrain relative gauges of disconnected components. Twenty ambiguous terms remain an explicit limitation of the fixed-target model; no uncertainty model, arbitrary sign choice, radial tie-break prior or geometry optimizer was introduced.

Follow-up: [detailed signs versus GT audit](gt-sign-audit.md), including affected fibers, one-step errors, component alignment and reference coverage.
