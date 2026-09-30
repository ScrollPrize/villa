# Observation-memory fiber regression

Regression has one model: the continuous follower with main-encoder observation
memory, shared-decoder refinement, and causal segment survival confidence.
See [architecture and supervision](TRAJECTORY_MEMORY.md).

## Training

From `fiber_follow`, using the existing project environment:

```bash
bash scripts/launch_memory.sh
tail -F output/logs/axial_survival_memory_v5_run1.log
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

## Validation

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=../../.. \
../../../../.venv/bin/python -m pytest \
  tests/test_survival_confidence.py tests/test_detailed_memory.py \
  tests/test_trajectory_memory.py tests/test_recurrent_refinement.py \
  tests/test_feature_correctness.py tests/test_direction_inputs.py \
  -q -o cache_dir=/tmp/fiber-survival-pytest
```

Checks cover suffix independence, segment evidence, full observation access,
first-failure/censoring losses, generator gradient isolation, memory retention,
chronological replay, optimizer updates, checkpoint round trips and tracing.
They do not establish trained tracing accuracy or confidence calibration.
