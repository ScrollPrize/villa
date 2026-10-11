# Global distance objective comparison

Historical results: these runs included the pairwise wrapped-phase metric.
The user subsequently selected squared and requested removing that metric.
Current code keeps the vertical metric only; the tables below were not rerun.
Subsequently the V term was corrected to equal-V constraints (no CT-Z pair target)
and separate weak per-fiber V≈Z calibration. The results below also predate that
correction and are not measurements of the current loss.

## Protocol

Four matched CCW solves, fresh radial initialization,151 winding hypotheses,
seed0,4096constraints per random minibatch,16130optimizer updates. Sampling is
uniform with replacement, not exhaustive epochs. One report every1613updates
(~one event-pass equivalent). All geometry, widths, winding and broken
probabilities train from the first update; no warm start or reference
initialization. Cached earlier normal/HV/direction solutions are unchanged.
Reference data are read only by diagnostic metrics, not optimization.

Fused shared objective (`pair_backend=compiled`), FP32; all-to-all pairs retained.
24GiB shared GPU-manager grants,8CPU cores each, GPUs0–3. Concurrent training
may share GPU compute, so wall times are not isolated performance benchmarks.

| GPU | Objective | Distance term |
|---|---|---|
| 0 | likelihood | Existing probability-mixture likelihood |
| 1 | squared | Expected .5r² |
| 2 | Huber | Expected Huber(r), transition at |r|=1 |
| 3 | Huber + likelihood | Expected Huber plus existing likelihood |

Here r=(predicted signed separation−measured separation)/0.5. Unknown-sign
observations use absolute separation. Huber is .5r² inside the transition and
|r|−.5 outside. Expected costs use unconditional pair mass, so broken probability
gates them without renormalizing away its rejection cost. All distance weights1;
sign10, rejection10, learning rate.03, width learning rate2. Squared/Huber use
expected existing metric costs; hybrid includes metric only through likelihood.
All other priors, continuation and fixed entropy regularization unchanged.
No band pruning, moment approximation, or global gradient accumulation.

## Validation

49focused regression tests pass. Tests cover farther wrong assignments, gradient
directions, partial/full rejection, Huber transition, sign±1/unknown, wrong-sign
distances and invalid sigma. Managed real4096×151 CUDA loss and gradient checks
passed for all three new objectives. Evidence:
`data/uv_distance_objectives_validation_v1/summary.json`.
Fused FP32 can change rounding and optimizer trajectories (see fusion report).

## Commands and artifacts

Each command was launched concurrently through the approved runner:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_distance_likelihood.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_distance_squared.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_distance_huber.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_distance_huber_likelihood.json
```

Outputs: `data/uv_distance_objectives_v1/{likelihood,squared,huber,huber_likelihood}`.
Each contains frozen configuration, history.json and upon completion solution.npz
and summary.json. Source constraint cache: `data/uv_anchor_v1/ccw_tensors.npz`.
Projection metrics: `data/reference_comparison100_v1/projection.npz`.

## Results

All four completed16130updates (~10.001event-pass equivalents), exited0 and
saved solutions. [Full per-interval tables](uv_distance_objectives_progress.md).

| Objective | Reference L1 ↓ | Within2 ↑ | Retained arc ↑ | Sign violations ↓ | Fit q10/q90 | Ref q10/q90 | Seconds |
|---|---:|---:|---:|---:|---|---|---:|
| Initial (all) | 12.24234 | 12.9912% | 100.0000% | 40.9860%* | 49/122 | 42/124 | — |
| Likelihood baseline | 11.91029 | 13.1291% | 68.3658% | 3.0784% | 49/122 | 42/124 | 822.1 |
| Expected squared | 11.72362 | 13.2096% | 66.0228% | 3.3311% | 49/122 | 42/125 | 712.6 |
| Expected Huber | 11.86019 | 13.2250% | 70.2530% | 3.3357% | 49/122 | 42/124 | 694.3 |
| Huber + likelihood | 11.87166 | 13.2166% | 64.6978% | 3.2167% | 49/122 | 42/125 | 741.8 |

Reference metrics use equal-weight original source fibers, retained curve-sample
median assignments, and global offset correction (+12 for every final run).
The evaluated retained populations differ; these are not common-subset metrics.
Reference q90 shifts to125 in squared/hybrid because the retained subset changes,
not because reference data changed. Initial whole matched reference q90 is124.
*Initial/intermediate sign statistics use the same fixed100k sample; final rows
use the full audit over all6,606,100events, counting violations among retained
known-sign events. Run seconds exclude manager queue wait and initial setup.

| Objective | Last interval mean objective | Pair | Regularization | Mean broken probability |
|---|---:|---:|---:|---:|
| Likelihood | 2.55224 | 1.04647 | 1.50577 | 14.4263% |
| Squared | 2.61608 | 0.92991 | 1.68617 | 16.2653% |
| Huber | 2.27046 | 0.87079 | 1.39966 | 13.4275% |
| Hybrid | 2.77401 | 1.03560 | 1.73841 | 16.8379% |

Raw objective scales differ across variants and cannot rank model quality.
These are online interval means, not frozen-state full-dataset objectives.

## Interpretation

No large global winding recovery in this initial parameter comparison. Squared
improves retained-fiber L1 by0.18667 versus baseline but retains2.3431percentage
points less length. Huber improves L1 by0.05010 while retaining1.8872points more
length, but its final sign violation rate is0.2573points higher. Hybrid loses
3.6681points retained length with only0.03863L1 improvement. All robust fitted
ranges remain49–122 and only about13%match the reference within two windings.

Huber is the most promising retention/error tradeoff of these initial settings,
not a demonstrated solution to the global-resolution problem. None is promoted
as a new default. No additional parameter tuning or longer solve was silently
performed. Next experiments should distinguish actual winding movement from
selection by rejection, and examine optimization/init constraints rather than
assuming the mixture likelihood was the sole bottleneck.
