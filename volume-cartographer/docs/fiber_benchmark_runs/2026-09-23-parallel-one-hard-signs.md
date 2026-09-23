# Hard parallel-one signs — 2026-09-23

Config-only ablation, independently reviewed. Same saved485 beam8 source traces /1204 BP pieces, same1024 crop, normals and reference protocol. No retracing/inference or native code changes.

Command:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_beam_bp_p1hardsign_20260923.json
```

Added sign weights1,1, hard signs both, parallel cutoff1.5. Cutoff is applied after quantization; zero cannot have a sign, >=2 excluded. Those >=2 magnitudes already had zero weight. Parallel-one magnitude0.1 and perpendicular signs/continuity unchanged. Sign means strict positive signed difference; defect endpoints still bypass hard constraints.

| Metric | Previous | Hard parallel-one signs |
|---|---:|---:|
| Reference exact/all26 | 88.46% (23/26) | 84.62% (22/26) |
| Exact on24 intersecting references | 91.67% (22/24) | 91.67% (22/24) |
| Wrong / missing, all26 | 3 /0 | 3 /1 |
| Constraint score (higher better) | -118.7336% | -402.0611% |
| Problematic / retained fulfilled | 23007 /19377 | 33942 /8442 |
| Retained BP pieces | 1015/1204 (84.30%) | 460/1204 (38.21%) |
| Initial perpendicular signs neutralized by defects | 8475/24766 (34.22%) | 12586/24766 (50.82%) |
| Initial continuation links neutralized | 180/719 (25.03%) | 309/719 (42.98%) |
| Whole runner wall time | 53.0352s | 42.1107s |

Actual CSV audit:5629 hard parallel signs, all target±1;24766 perpendicular hard signs unchanged. Initial parallel signs2344active/satisfied,3285defect-neutralized (58.36%). All initial active hard signs and continuation constraints satisfied. This is feasible through defects, not a hard infeasibility exit. Initial, conditioned and fresh solvers reached500-message cap; not converged/optimality proof.

Conditioned reference errors:4→1.5(outside crop),8→7(marginal124base-voxel coverage),11.5→12.5(interior);12missing(outside crop). Previous errors8→7,11.5→12.5,12→8. Thus no improvement on covered references, and markedly worse retention. Hard signs can induce more defects under contradictory constraints; this run alone does not prove which constraints/finite-iteration effects caused the increase. Faster wall time is not a speed improvement claim: later graph is much smaller.

Artifacts: /ephemeral/hendrik/fiber_crop_bench/20260923_v1/bp-beam-new-perpsign-p1hardsign{.log,.run.json,_winding_factors.csv,_consistency.csv}. No defaults changed; retain previous policy as baseline.
