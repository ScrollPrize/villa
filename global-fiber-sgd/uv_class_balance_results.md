# Perpendicular-dominant balanced losses with fixed sign charge

## Objective

Use graph-wide class counts, not minibatch counts. With E total events, P
parallel and Q perpendicular, weights are E/(11P) and 10E/(11Q). Thus mean
weight is one and aggregate class coefficients are1/11 and10/11. Actual loss
contributions can differ from10:1 because residuals and probabilities differ.
This weights distance, vertical equality, sign hinge, fixed charge and endpoint
phase priors. Uniform node priors/rejection and continuity are unchanged.

The existing calibrated fixed sign charge is enabled with multiplier32:
C =32*(E/N)*rejection_cost, since fixed rejection weights are one. It charges
C*max(expected violating pair mass, retained mass when selected pair violates),
then receives the same class weight. Hinge remains40*relu(-ordered).
This adds probability/rejection gradients for arbitrarily small violations;
the hinge supplies the coordinate gradient. No infinite barrier or guarantee.
Selected ordering is recomputed in float64; all other math/precision unchanged.

Multiplier32 chosen because16 loses the local rejection-dominance property
afterparallel downweighting at50% retention (16*.117*.5<1;32*.117*.5>1).
Hard-readout protocol remains the previous argmax of winding+broken categories;
no new50% threshold or post-hoc deletion. Diagnostics remain unweighted and
count signed separation below-1e-4. Unknown-sign events have no sign charge.

## Runs

Same subset IDs14–20 (positions5–11), Z30k–50k,2192sources,2199pieces,
24771events (19222parallel,5549perpendicular). Same151hypotheses,CCW, six
exclusive GPUs, globalbatch24576, stateLR0.0073484692283495336,widthLR2,
seed0,allvariablesjoint fromstart,10080updates (~10000.649sampledpasses),
diagnostics every1008updates. Fresh and reference/width original initializations.
New options: constraint_class_balance=true,perpendicular_relative_weight=10,
sign_charge_multiplier=32. Old configurations/objectives remain unchanged.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_balanced10000_suite.json
```

Outputs: data/uv_balanced10000_w14_20_{fresh,reference}_v1.
Compare geometric metrics with uv_reference_5_11_10000_results.md; scalar
losses are different objectives and cannot be compared directly.

## Validation

Independent plan/code/config reviews approved.80 CPU regression tests passed:
class balancing, fixed sign charge, distance objectives, pair backend, layout,
anchor, local window and distributed configuration. Production preflight now
exercises charged/class-weighted compiled gradients against eager concatenated
six-rank gradients. Both starts passed production preflight.

## Original LR trial (interrupted at user request)

Fresh completed10k; reference was stopped after its2k diagnostic. No final
reference snapshot exists. Stopped suite released its GPU lease. Production
charged compiled gradient checks passed. Tables include all recorded checkpoints.

### fresh

| Pass | Loss | L1 | Within2 % | Retained % | Perp sign % | Parallel sign % | Width factor | Fit range | Ref range |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.7750 | 37.6269 | 0.9829 | 11/23 | 15/20 |
| 1000.06 | 503.8846 | 2.1456 | 67.162 | 65.174 | 0.1288 | 1.5198 | 1.0072 | 12/23 | 14/20 |
| 2000.13 | 17.7521 | 1.8900 | 72.489 | 62.104 | 0.1401 | 0.8001 | 1.0205 | 13/22 | 14/20 |
| 3000.19 | 7.8758 | 1.7133 | 75.112 | 61.591 | 0.0480 | 0.3073 | 1.0279 | 13/22 | 14/20 |
| 4000.26 | 5.3305 | 1.6185 | 79.682 | 62.036 | 0.0000 | 0.3772 | 1.1170 | 12/21 | 14/20 |
| 5000.32 | 4.2249 | 1.4911 | 81.115 | 62.310 | 0.0000 | 0.4765 | 1.1144 | 13/20 | 14/20 |
| 6000.39 | 3.6401 | 1.3859 | 83.092 | 62.783 | 0.0000 | 0.6126 | 1.1003 | 13/20 | 14/20 |
| 7000.45 | 3.3373 | 1.3099 | 83.945 | 63.435 | 0.0000 | 0.1648 | 1.0758 | 13/20 | 14/20 |
| 8000.52 | 3.1441 | 1.3044 | 84.535 | 64.001 | 0.0000 | 0.4778 | 1.0440 | 13/20 | 14/20 |
| 9000.58 | 3.0210 | 1.3133 | 85.811 | 64.166 | 0.0000 | 0.0543 | 0.9237 | 14/21 | 14/20 |
| 10000.65 | 2.8635 | 1.2706 | 86.898 | 64.727 | 0.0000 | 0.2540 | 0.9016 | 14/21 | 14/20 |

### reference

| Pass | Loss | L1 | Within2 % | Retained % | Perp sign % | Parallel sign % | Width factor | Fit range | Ref range |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0009 | 100.000 | 100.000 | 9.2860 | 11.5138 | 1.0000 | 15/20 | 15/20 |
| 1000.06 | 330.7773 | 0.6466 | 97.828 | 90.777 | 0.1445 | 0.9586 | 1.0086 | 14/20 | 15/20 |
| 2000.13 | 19.3627 | 0.4570 | 98.258 | 86.593 | 0.0223 | 0.7278 | 1.0014 | 14/20 | 15/20 |
