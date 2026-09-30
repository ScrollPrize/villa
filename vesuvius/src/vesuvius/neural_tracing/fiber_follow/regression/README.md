# Observation-memory fiber regression

Regression has one model: the continuous follower with main-encoder observation
memory, shared-decoder refinement, and causal segment survival confidence.
See [architecture and supervision](TRAJECTORY_MEMORY.md).

## Training

From `fiber_follow`, using the existing project environment:

```bash
bash scripts/launch_memory.sh
tail -F output/logs/axial_survival_memory_v6_run1.log
```

`launch_trajectory_memory.sh` delegates to the same launcher. The launcher uses
CT/presence plus six unsigned local-frame direction channels, batch 8,
microbatch 4, eight workers, 16-point maximum commit, history spacing 8,
16 memory slots, 64 cached observations, two-decision gradient chunks,
and one shared-decoder refinement pass. It starts random weights and refuses
an existing destination. Set `RUN_NAME` for another fresh run; trailing trainer
arguments override launcher settings. No training is launched by editing code.

The loss combines geometry, generated-path survival likelihood, and candidate-path
survival likelihood. Survival likelihood sums the supervised intervals per path;
it does not average prefixes or reweight the commit window. The existing geometry
loss still weights the commit window and supervises initial/final curves 25%/75%.
The numerical scale of confidence losses has therefore changed; their weights
and confidence thresholds require calibration on freshly trained weights.

Memory writing runs in FP32; CUDA forward/backward otherwise uses BF16 autocast.
Gradient clipping remains separate for memory (5) and the rest of the model (20).
Compiled training retains Inductor `emulate_precision_casts`. CPU full-graph
capture checks do not establish production CUDA throughput or numerical parity.

## Inputs and sampling

All physical coordinates are trace-grid voxels. The main crop is 120x101x101 at
spacing 0.5. The output has 16 points on forward planes 1 through 16.
Every generator decoder layer reads the complete deep lattice and the dense
features at every lateral pixel on all 16 output planes, during both initial
prediction and refinement. Each plane retains the full 101x101 resolution:
163,216 fine tokens join 39,015 deep tokens plus references and observation
memory. Fine tokens have learned channel projections and physical XYZ position
embeddings. Output planes between input slices interpolate only along depth.
The scorer reads segment-local dense samples plus deep/reference/memory tokens.
Observed history and the seed condition the crop; annotations only define losses.
Cold starts with a remote seed encode that seed crop once. Warm tracing encodes
only each new head crop and carries observation memory forward.

Direction inputs are `CT, presence, uu, vv, ff, uv, uf, vf`: the six independent
components of an unsigned direction second moment, expressed in each crop's own
frame. Missing directions are zero. Interpolate second moments, not encoded bytes
or signed vectors. CT/presence blur, brightness, noise and dropout do not change
the direction channels. Direction fields come from the selected presence grid.

Matched identity pairs and constructed wrong-fiber histories supply difficult
choice/departure states. Remote observations can establish identity even when the
seed is outside the current crop. Training streams retain observed geometry,
including wrong turns, without using membership labels to filter memory writes.
Unknown identity and censored annotation mask supervision.

Sampling revision 4 budgets supervision per stream, independently of its length.
With `--feature-history-loss-fraction .25`, a stream reserves .75 of its loss
weight for the selected endpoint and shares .25 across earlier observations.
A single-observation stream has weight one. Unknown labels contribute zero;
their budget is not redistributed. All observations still enter memory in order.
Weights are applied before the effective crop-batch denominator, including during
endpoint replay; they are never renormalized within a gradient chunk. Loss scales
and gradient norms therefore differ from earlier runs that weighted every crop
equally; compare tracing metrics rather than raw historical losses.

The launcher requests matched pairs for .3 of endpoint proposals, with .75 of
pair proposals being recoverable choices (`--decision-choice-fraction`). Both
sides of a choice receive geometry targets; an already-departed side receives
only stopping supervision. Unavailable or holdout-rejected pairs fall back to
ordinary draws. The launcher also caps whole synthetic-switch streams at .15 of
admitted crops. Logs report actual endpoint counts, choice counts, and supervision
weights separately from crop-source percentages, so fallback and stream expansion
are visible.

Bank following has its own endpoint-proposal budget: the launcher's
`--bank-following-probability .2` reserves 20% independently of the 30% matched
decisions. The remaining 50% uses `--fresh-fraction .7`: 35% annotation-fresh
and 15% recent replay overall, before fallback and rejection. Fresh targets come
from actual annotations; covered-parent resampling and synthetic switch histories
still use annotated targets. Generated bank paths enter through the dedicated
following budget or matched decisions. Missing or unsafe bank proposals fall back
to the annotation/replay mix; empty replay strata fall back to annotations.
These are requested endpoint shares, not guaranteed crop percentages.
Sampling revision 5 changes `--bank-following-probability` from a fraction of fresh
draws to an independent endpoint fraction, including when resuming old runs.
Its sum with `--decision-fraction` must not exceed one.

Each matched endpoint scores four shuffled paths: the two bank/annotation
continuations and two smooth transitions between them, with varied transition
onsets. Paired observations share candidate geometry and order. Candidate labels
use the same dense tolerance, recovery limit, foreign masks, endpoint semantics
and censoring as generated paths. They are not assigned all-positive/all-negative
labels by candidate identity. Logs count first-segment and later first failures.
The transitions are rejected-path proposals, never geometry targets or certified
bank following paths.

The loader no longer samples contrastive points or emits embedding query tensors.
Lateral resampling uses visible forward bank coverage directly. The obsolete
`--negative-near-fraction`, `--negative-near-distance` and `--negative-lateral-max`
options are removed; bank mining and crop support determine available foreign
geometry. Separate negative, following and continuation banks remain supported.

## Validation

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=../../.. \
../../../../.venv/bin/python -m pytest \
  tests/test_survival_confidence.py tests/test_detailed_memory.py \
  tests/test_trajectory_memory.py tests/test_recurrent_refinement.py \
  tests/test_feature_correctness.py tests/test_direction_inputs.py \
  tests/test_decision_training.py tests/test_sampling_balance.py \
  tests/test_output_plane_features.py \
  -q -o cache_dir=/tmp/fiber-survival-pytest
```

Checks cover suffix independence, segment evidence, full observation access,
first-failure/censoring losses, generator gradient isolation, memory retention,
chronological replay, optimizer updates, checkpoint round trips and tracing.
Output-plane checks cover full lateral coverage, physical coordinates, fractional
depth interpolation, access through every generator layer and gradient isolation.
They do not establish trained tracing accuracy or confidence calibration.
