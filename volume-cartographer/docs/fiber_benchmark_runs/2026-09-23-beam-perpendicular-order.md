# New beam8: hard perpendicular order + weak parallel-one (2026-09-23)

Reuses `/ephemeral/hendrik/fiber_crop_bench/20260923_v1/beam-crop-new.zarr`
(485 candidates), same Paris4 crop,26 references and Lasagna normals as the
[baseline](2026-09-23-dense-beam-bp.md). No retracing/inference.

Winding measurement ablation:

| Factor | Baseline | Experiment |
|---|---:|---:|
| Perpendicular winding magnitude | 0 | 0 |
| Perpendicular sign/order | sign weight1, penalty44 with angle-based hard promotion | hard at every enabled factor |
| Parallel same-winding magnitude | 0.5 | 0 |
| Parallel one-winding-apart magnitude | 4 | 0.1 |
| Parallel >=2 magnitude | 1 | 0 |
| Parallel sign/order | 0.5 | disabled |

Hard order requires strictly correct sign, not equality. The0.1 soft coefficient
still receives existing normal/decision confidence weighting. Orientation
prepass, hard split continuity, phase0.5, measurement scale0.822, temperature0.25,
500-message limit and reference-assisted oracle-inlier closure remain unchanged.
This is not a change to production defaults. No automatic fallback to finite
sign costs is authorized if hard constraints are infeasible.

## Reproduction

```sh
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_beam_bp_perpsign_20260923.json
```

New flags: `--winding-weights 0,0,0,0.1,0 --winding-sign-weights 1,0
--winding-hard-signs perpendicular --winding-sign-cost hard
--winding-hard-sign-angle off`. Independent config review passed.
Artifacts prefix: `/ephemeral/hendrik/fiber_crop_bench/20260923_v1/bp-beam-new-perpsign-p1low`.
Runner captures command/source/binary/reference identities and wall/CPU/RSS.

## Results

| Metric | New beam8 baseline | Hard perpendicular + parallel-one0.1 |
|---|---:|---:|
| Constraint score (closer0 better) | -178.90% | -119.57% |
| Oracle round0 exact references | 88.46% | 88.46% |
| Exact/wrong/missing | 23/3/0 | 23/3/0 |
| Problematic / retained fulfilled | 27187/15197 | 23081/19303 |
| BP wall time | 45.73s | 68.69s |
| Conditioned round0 status | converged | message_limit (accepted by unchanged protocol) |

The same three winding disagreements remain:8.0->7.0,11.5->12.5,12.0->8.0.
All other23 references match. No reference accuracy gain; constraint ratio is
better but this is a changed objective and conditioned retention set.

Factor CSV verified24766 perpendicular signs, all hard; zero parallel signs
and zero perpendicular magnitudes.5629 parallel target1 factors remain,
plus719 target0 split-continuity factors (1204pieces minus485traces), not
same-winding measurement factors.

Important hard-constraint caveat: existing solver defect handling remains
enabled at its default cost100. The decoded report evaluates16291 hard signs,
all satisfied, and neutralizes8475 via the existing defect mechanism. Thus
"hard" does not force all24766 input relationships to survive. No finite
sign-cost fallback occurred. Initial and conditioned solves hit the500-message
cap; this is not a converged-optimum claim.
