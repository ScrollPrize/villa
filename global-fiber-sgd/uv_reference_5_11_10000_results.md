# Positions 5–11: 10,000 total passes

Both fresh and reference-initialized joint solves use the unchanged audited
IDs 14–20, base Z=30,000–50,000 subset: 2,192 source fibers, 2,199 pieces,
24,771 constraints and one continuation. Earlier sign/HV/direction stages
are reused. These are sampled dataset-pass equivalents, not exhaustive epochs.

Same six exclusive GPUs, global batch 24,576, state LR 0.0073484692283495336,
width LR 2, seed 0, 151 hypotheses, CCW, compiled quadratic loss, combined
coordinate mapping. All parameters are jointly optimized from the start.
10,080 updates give 10,000.649 passes; reporting every 1,008 updates gives
1,000.065-pass intervals. This is 10k total, not 10k additional passes.

Existing checkpoints lack Adam state, so both runs reproduce their original
initializations instead of warm-starting with reset optimizer state. Check
the first 1,008-update metrics against the completed 1k runs. Loss averages
the preceding reporting interval, so its first value is not comparable to
the previous 100-pass loss windows. Reference L1 and within-two metrics use
global offset correction, equal original-fiber weighting and retained fibers;
rejection changes the cohort. Width factor is mean fitted/reference width for
the seven supported reference intervals after that offset correction.

## Reproduction

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_window10000_suite.json
```

Child configs: `run_uv_window10000_fresh.json` and
`run_uv_window10000_reference.json`. Outputs:
`data/uv_window10000_w14_20_{fresh,reference}_v1`; suite metadata:
`data/uv_window10000_suite_v1`. Completed output roots are protected from
overwriting; choose new output roots for a rerun. No solver code changes.

## Results

Both runs completed and validated. Full checkpoint tables follow.
Previous 1k visualizations remain unchanged; no new rendering requested.

## fresh: every 1,000 passes

| Passes | Loss | Ref L1 | Within ±2 % | Retained % | Sign violations % | Width factor | Offset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.662 | 0.9829 | -1 | 11/23 | 15/20 |
| 1000.06 | 18.3763 | 1.5227 | 80.937 | 95.227 | 1.549 | 1.0420 | 0 | 13/22 | 15/20 |
| 2000.13 | 1.2650 | 1.2027 | 87.436 | 95.344 | 0.770 | 1.0752 | 0 | 14/21 | 15/20 |
| 3000.19 | 0.5835 | 1.0410 | 90.210 | 95.675 | 0.749 | 1.0851 | 0 | 14/21 | 15/20 |
| 4000.26 | 0.4024 | 0.9140 | 92.419 | 96.029 | 0.755 | 1.0815 | 0 | 14/21 | 15/20 |
| 5000.32 | 0.3210 | 0.8530 | 93.738 | 96.530 | 0.612 | 1.0567 | 0 | 14/21 | 15/20 |
| 6000.39 | 0.2870 | 0.8319 | 94.303 | 96.810 | 0.719 | 1.0095 | 0 | 15/21 | 15/20 |
| 7000.45 | 0.2567 | 0.8345 | 94.406 | 97.297 | 0.587 | 0.9488 | 0 | 15/21 | 15/20 |
| 8000.52 | 0.2372 | 0.8379 | 94.457 | 97.526 | 0.584 | 0.8877 | 0 | 15/21 | 15/20 |
| 9000.58 | 0.2208 | 0.8482 | 94.555 | 97.612 | 0.683 | 0.8394 | 0 | 15/21 | 15/20 |
| 10000.65 | 0.2139 | 0.8452 | 94.931 | 97.866 | 0.649 | 0.9215 | -1 | 14/20 | 15/20 |

## reference: every 1,000 passes

| Passes | Loss | Ref L1 | Within ±2 % | Retained % | Sign violations % | Width factor | Offset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0009 | 100.000 | 100.000 | 10.981 | 1.0000 | 0 | 15/20 | 15/20 |
| 1000.06 | 12.6305 | 0.5807 | 98.175 | 100.000 | 0.945 | 1.0046 | 0 | 14/20 | 15/20 |
| 2000.13 | 0.7284 | 0.5078 | 98.403 | 100.000 | 0.880 | 0.9883 | 0 | 14/20 | 15/20 |
| 3000.19 | 0.2849 | 0.5014 | 98.495 | 100.000 | 0.656 | 0.9660 | 0 | 14/20 | 15/20 |
| 4000.26 | 0.1845 | 0.5205 | 98.586 | 100.000 | 0.673 | 0.9371 | 0 | 14/20 | 15/20 |
| 5000.32 | 0.1493 | 0.5420 | 98.586 | 100.000 | 0.509 | 0.9024 | 0 | 14/20 | 15/20 |
| 6000.39 | 0.1342 | 0.5584 | 98.631 | 100.000 | 0.643 | 0.8753 | 0 | 14/20 | 15/20 |
| 7000.45 | 0.1269 | 0.5666 | 98.631 | 100.000 | 0.626 | 0.8618 | 0 | 14/20 | 15/20 |
| 8000.52 | 0.1228 | 0.5798 | 98.631 | 100.000 | 0.669 | 0.8546 | 0 | 14/20 | 15/20 |
| 9000.58 | 0.1205 | 0.5839 | 98.631 | 100.000 | 0.695 | 0.8514 | 0 | 14/20 | 15/20 |
| 10000.65 | 0.1190 | 0.5803 | 98.677 | 100.000 | 0.682 | 0.8496 | 0 | 14/20 | 15/20 |

## Validation and timing

Both completed successfully, finite final states, exact reproduction of the prior
1k reference metric dictionaries at update 1008. Six-rank gradient equivalence
and coordinate value/gradient/bitwise-repeat preflights passed. GPU manager
lease c703ce04-317a-40c7-a7b6-32386d32ed37 finished and released all resources.

| Start | Solve seconds | Update mean ms | p50 ms | p95 ms | Peak GiB/GPU |
|---|---:|---:|---:|---:|---:|
| fresh | 158.358 | 15.438 | 15.408 | 15.618 | 1.161 |
| reference | 172.771 | 16.872 | 16.860 | 17.333 | 1.161 |

Single trial per initialization; first five updates excluded from update timings,
per-update maximum over six ranks. Solve timing includes diagnostics/save but
not worker startup/manager wait. No change in math or solver implementation.

Fresh improves strongly from 1k but reference L1 plateaus around 6k. Reference
L1 is best near 3k among nonzero reported checkpoints, then worsens while loss
continues falling. Reference mean width factor drops to 0.8496; smaller objective
is not synonymous with improved reference geometry. Fresh global alignment
changes from zero to -1 at 10k, changing its compared width intervals as well.
Reference sign violations persist; no claim of sign feasibility or convergence.

## Exact loss weights used

Parallel and perpendicular events have identical per-event weights and are
sampled uniformly together, not class-balanced (19,222 parallel, 5,549
perpendicular). For hypothesis pair (a,b), define corrected winding separation
delta = q_b - q_a - phase_delta + 0.5*(family_b-family_a), and ordered = sign*delta.
Known-sign distance cost is 0.5*((ordered-measured_distance)/0.5)^2, weight1.
Unknown-sign events use abs(delta) for that distance term.
Sign cost is sign_weight10 * 4*relu(-ordered): a linear hinge with effective
slope40 per winding, with no constant cost for merely crossing zero.
Every cost is summed over all151x151 pairs with joint probability p_a*p_b,
then averaged over events. Probabilities include broken in their softmax, so
raising broken probability reduces retained pair mass and hence these costs.
Rejection costs10*mean(p_broken), averaged uniformly over pieces, not events.
The optional sign_charge_multiplier fixed violation charge is absent/default0
in these runs; no hard barrier or selected-assignment safeguard is enabled.
Vertical pair equality costs0.01*min(((v_b-v_a)/128)^2,100), same for both classes.
Reported sign violations instead use retained argmax assignments, known signs,
and ordered < -1e-4, not the probability-weighted training hinge.

