# Balanced objective: state-learning-rate comparison

User requested stop original balanced suite and test10x/0.1xLR. Interrupted
baseline reference run after2k saved diagnostic; baseline fresh had already
finished10k. Preserved both. See uv_class_balance_results.md for all recorded
baseline metrics; original unbalanced10k results are in
uv_reference_5_11_10000_results.md.

Four new original-initialized trials: high/low times fresh/reference,10k total
sampled passes each. High stateLR0.07348469228349534; low0.0007348469228349534.
Width LR2 unchanged, explicitly communicated. This is not scaling both optimizers.
Same151hypotheses,subsetpositions5–11/IDs14–20,Z30k–50k,seed0,globalbatch24576,
sixexclusiveGPUs,10080updates,diagnostics1008updates. Same class-balanced
aggregate perpendicular:parallel10:1 and fixed sign-charge multiplier32.
No warm starts, altered loss terms, new prior targets or hard-readout changes.

## Command

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_balanced_lr_suite.json
```

Child configs: run_uv_balanced_lr_{high,low}_{fresh,reference}.json.
Outputs: data/uv_balanced_lr_{high,low}_{fresh,reference}_v1.
Suite manager task b697d6ed-6481-4190-86c9-351967569b8e.
Independently reviewed configurations. Existing80regressiontests passed;
production validates charged/class-weighted compiled vs eager6GPUgradients.

## Metrics

Reference L1 and within-two use global median offset correction. Rejection
changes the evaluated cohort, so retained length and fitted/reference q10/q90
ranges are both reported. Width factor is mean fitted/reference over7supported
intervals afterglobaloffset; offset changes can change the compared intervals.
Loss is online average overpreceding1kpasses, not a frozenstateobjective.
Loss comparable among these LR trials but not with old uncharged/unbalanced loss.

All four trials finished and validated. Full tables follow.

## high fresh: all checkpoints

| Pass | Loss | Ref L1 | Within2 % | Retained % | Perp sign % | Parallel sign % | Width factor | Width abs error % | Offset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.7750 | 37.6269 | 0.9829 | 2.766 | -1 | 11/23 | 15/20 |
| 1000.06 | 64.6566 | 1.4921 | 80.628 | 58.497 | 0.0556 | 0.1381 | 0.9280 | 7.204 | 0 | 13/21 | 14/20 |
| 2000.13 | 3.5778 | 1.3897 | 84.056 | 60.030 | 0.0000 | 0.1480 | 0.9288 | 7.120 | 0 | 14/21 | 14/20 |
| 3000.19 | 3.3220 | 1.3237 | 85.292 | 60.941 | 0.0533 | 0.0800 | 0.9265 | 7.354 | 0 | 14/21 | 14/20 |
| 4000.26 | 3.1883 | 1.2938 | 86.025 | 61.881 | 0.0000 | 0.0467 | 0.9996 | 3.692 | -1 | 13/20 | 14/20 |
| 5000.32 | 3.1427 | 1.2835 | 86.160 | 62.911 | 0.0000 | 0.1215 | 0.9093 | 9.072 | 0 | 14/21 | 14/20 |
| 6000.39 | 2.9858 | 1.2645 | 85.809 | 63.965 | 0.0000 | 0.0145 | 0.8960 | 10.398 | 0 | 14/21 | 14/20 |
| 7000.45 | 2.8238 | 1.2617 | 86.108 | 64.426 | 0.0000 | 0.0435 | 0.8824 | 11.756 | 0 | 14/21 | 14/20 |
| 8000.52 | 2.7998 | 1.2446 | 86.250 | 64.944 | 0.0000 | 0.0722 | 0.8714 | 12.856 | 0 | 14/21 | 14/20 |
| 9000.58 | 2.8962 | 1.2215 | 86.342 | 65.127 | 0.0000 | 0.0435 | 0.8618 | 13.821 | 0 | 14/21 | 14/20 |
| 10000.65 | 2.7149 | 1.2058 | 86.635 | 65.508 | 0.0000 | 0.1432 | 0.8514 | 14.856 | 0 | 14/21 | 14/20 |

## high reference: all checkpoints

| Pass | Loss | Ref L1 | Within2 % | Retained % | Perp sign % | Parallel sign % | Width factor | Width abs error % | Offset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0009 | 100.000 | 100.000 | 9.2860 | 11.5138 | 1.0000 | 0.000 | 0 | 15/20 | 15/20 |
| 1000.06 | 44.5435 | 0.5076 | 97.991 | 78.880 | 0.0823 | 0.1525 | 0.9913 | 1.241 | 0 | 14/20 | 15/20 |
| 2000.13 | 3.0748 | 0.5597 | 97.875 | 78.151 | 0.0000 | 0.8679 | 0.9672 | 4.539 | 0 | 14/20 | 15/20 |
| 3000.19 | 2.9857 | 0.5670 | 97.767 | 78.265 | 0.0285 | 0.2428 | 0.9432 | 7.610 | 0 | 14/20 | 15/20 |
| 4000.26 | 3.1272 | 0.5742 | 97.879 | 78.260 | 0.0000 | 0.0373 | 0.9297 | 9.282 | 0 | 14/20 | 15/20 |
| 5000.32 | 2.8284 | 0.5870 | 97.995 | 78.444 | 0.0285 | 0.3402 | 0.9208 | 10.434 | 0 | 14/20 | 15/20 |
| 6000.39 | 2.6807 | 0.5873 | 97.945 | 78.494 | 0.0856 | 0.1710 | 0.9173 | 11.048 | 0 | 14/20 | 15/20 |
| 7000.45 | 2.5824 | 0.5996 | 97.781 | 78.461 | 0.0287 | 0.0864 | 0.9149 | 11.301 | 0 | 15/20 | 15/20 |
| 8000.52 | 2.7687 | 0.6071 | 97.781 | 78.530 | 0.0000 | 0.1244 | 0.9128 | 11.202 | 0 | 15/20 | 15/20 |
| 9000.58 | 2.5516 | 0.6188 | 97.777 | 78.148 | 0.0876 | 0.0000 | 0.9109 | 11.133 | 0 | 15/20 | 15/20 |
| 10000.65 | 2.5073 | 0.6331 | 97.615 | 78.247 | 0.0584 | 0.0776 | 0.9082 | 10.966 | 0 | 15/20 | 15/20 |

## low fresh: all checkpoints

| Pass | Loss | Ref L1 | Within2 % | Retained % | Perp sign % | Parallel sign % | Width factor | Width abs error % | Offset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 3.1957 | 45.119 | 100.000 | 37.7750 | 37.6269 | 0.9829 | 2.766 | -1 | 11/23 | 15/20 |
| 1000.06 | 1713.9177 | 2.8203 | 52.372 | 100.000 | 23.4583 | 22.7100 | 1.1139 | 13.446 | -1 | 11/22 | 15/20 |
| 2000.13 | 1217.4697 | 2.6552 | 55.601 | 99.800 | 17.7183 | 18.4369 | 1.2447 | 24.833 | -1 | 11/22 | 15/20 |
| 3000.19 | 862.7227 | 2.4895 | 58.539 | 98.020 | 12.1856 | 13.7305 | 1.3513 | 35.126 | 0 | 12/23 | 15/20 |
| 4000.26 | 549.0717 | 2.3439 | 61.770 | 92.104 | 6.0367 | 8.9824 | 1.4839 | 48.394 | 0 | 13/23 | 15/20 |
| 5000.32 | 308.5963 | 2.2666 | 64.096 | 85.769 | 2.6178 | 5.3936 | 1.5574 | 55.741 | 0 | 13/23 | 14/20 |
| 6000.39 | 168.6143 | 2.1638 | 67.243 | 79.167 | 0.9211 | 3.3148 | 1.5892 | 58.918 | 0 | 13/23 | 14/20 |
| 7000.45 | 97.4555 | 2.0875 | 69.018 | 73.831 | 0.2326 | 2.2620 | 1.5873 | 58.734 | 0 | 13/23 | 14/20 |
| 8000.52 | 61.7954 | 2.0116 | 70.472 | 70.078 | 0.5113 | 1.3572 | 1.5454 | 54.540 | 0 | 13/23 | 14/20 |
| 9000.58 | 42.7420 | 1.9516 | 71.608 | 67.650 | 0.3113 | 1.2060 | 1.4686 | 46.861 | 0 | 13/23 | 14/20 |
| 10000.65 | 30.7049 | 1.9164 | 72.342 | 64.956 | 0.2122 | 0.6353 | 1.3612 | 36.116 | 0 | 13/23 | 14/20 |

## low reference: all checkpoints

| Pass | Loss | Ref L1 | Within2 % | Retained % | Perp sign % | Parallel sign % | Width factor | Width abs error % | Offset | Fit q10/q90 | Ref q10/q90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | — | 0.0009 | 100.000 | 100.000 | 9.2860 | 11.5138 | 1.0000 | 0.000 | 0 | 15/20 | 15/20 |
| 1000.06 | 1232.7867 | 0.5949 | 99.635 | 100.000 | 1.7310 | 4.7870 | 1.0251 | 18.592 | 0 | 14/21 | 15/20 |
| 2000.13 | 742.4468 | 0.7623 | 97.217 | 100.000 | 1.3523 | 4.6112 | 1.0186 | 32.596 | 0 | 14/21 | 15/20 |
| 3000.19 | 445.7088 | 0.8059 | 97.078 | 99.957 | 1.3704 | 3.5862 | 1.0078 | 40.896 | 0 | 14/21 | 15/20 |
| 4000.26 | 274.0091 | 0.8136 | 96.658 | 99.740 | 0.8667 | 2.6406 | 1.0046 | 43.690 | 0 | 14/21 | 15/20 |
| 5000.32 | 175.9364 | 0.7775 | 97.144 | 99.288 | 0.5067 | 2.3421 | 0.9964 | 41.307 | 0 | 14/21 | 15/20 |
| 6000.39 | 121.7072 | 0.7211 | 97.958 | 98.613 | 0.3283 | 2.3870 | 0.9817 | 35.291 | 0 | 14/21 | 15/20 |
| 7000.45 | 87.1005 | 0.6757 | 97.801 | 97.627 | 0.2768 | 1.6273 | 0.9593 | 27.804 | 0 | 14/21 | 15/20 |
| 8000.52 | 63.7232 | 0.5999 | 98.303 | 96.487 | 0.2816 | 1.4227 | 0.9336 | 23.613 | 0 | 14/20 | 15/20 |
| 9000.58 | 49.0010 | 0.5462 | 98.381 | 95.167 | 0.2115 | 1.0146 | 0.9232 | 18.341 | 0 | 14/20 | 15/20 |
| 10000.65 | 38.2828 | 0.4993 | 98.403 | 93.011 | 0.2604 | 0.6955 | 0.9176 | 15.612 | 0 | 14/20 | 15/20 |

## Timing and validation

| Run | Solve seconds | Update mean ms | p50 ms | p95 ms | Peak GiB/GPU |
|---|---:|---:|---:|---:|---:|
| high_fresh | 185.607 | 18.224 | 18.121 | 18.624 | 1.573 |
| high_reference | 185.917 | 18.264 | 18.247 | 18.474 | 1.573 |
| low_fresh | 182.172 | 17.875 | 17.949 | 18.174 | 1.573 |
| low_reference | 184.589 | 18.128 | 18.099 | 18.405 | 1.573 |

Allfour runs completed10k, finite finalarrays, sixrankcharged compiledgradient
checks andcoordinatepreflights passed. Suiteexit0; managerleasefinished/released.
Singletrial each, first5updates excluded; perupdatemax across6ranks. Solvewall
includesdiagnostics/save,notstartup/managerhandoff. Total solverwall738.285s.
Classweights ineachsummary: parallel0.1171527,perpendicular4.0582251,
aggregate shares0.0909091/0.9090909. No globalfeasibility claim.

## Interpretation

HighLR improvesfreshL1 to1.2058 versus1.2706balancedoriginalLR, butlowLRfresh
onlyreaches1.9164. Retainedlengthhigh65.51%,low64.96%. Reference startdiffers:
lowLR givesL1.4993 andretains93.01%,versushighL1.6331 and78.25%retained,
withhigherremaining signerrors. OriginalLRreference wasstoppedat2k andcannot
becompared as a10kresult. Unbalanced/unchargedfreshbaseline stillhadlowerL1
.8452 and97.87%retention, butmoreperpendicularsignerrors; objectives differ.
Widthmeanfactor canhideopposingerrors: reportmeanabsolute relativewidtherror
aswell. HighLRfresh finalfactor.8514/abs14.856%;lowLRfresh1.3612/abs36.116%.
No widthdecouplingwasimplemented. UserfavoredhighLRprovisionally; notchanged
historicalconfigs orotheroptimizerdefaults.
