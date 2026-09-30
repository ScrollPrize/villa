# Observation memory for the continuous follower (v4)

The trajectory launcher selects feature-memory revision 2 with one recurrent
refinement pass. Memory retains observations from outside the current crop;
path prediction and confidence decide how to use them.

This implementation replaces the former admission-gated revision 2. Start a
fresh training run. Old revision-2 model and optimizer checkpoints are not
compatible, and no migration is provided. Revision 1 and the other architecture
modes remain separate experiments.

## Information flow

1. The main encoder extracts independent local appearance features and an axial
   deep lattice from the current CT, presence, direction fields and observed
   history. Persistent memory does not enter this encoding.
2. Memory stores 32 pooled deep tokens and up to 16 detail tokens. Detail tokens
   contain 3x3 stem-feature neighborhoods plus deep features at the head and
   visible observed history. These are independent observations, not the final
   memory-conditioned fine features.
3. Spatial queries retrieve from incoming memory. The coarse retrieval is
   projected and interpolated onto the deep lattice. The fine-feature decoder
   then combines that conditioned lattice with local stem features, once per crop.
   Both fine features and visible-reference features therefore see history.
4. The path decoder reads the current conditioned spatial features, visible
   references and persistent memory directly. Four shared decoder layers predict
   16 continuous lateral coordinates. One further pass samples the proposed
   curve's features and applies bounded one-voxel lateral refinement.
5. Confidence samples the actual curve being scored and queries memory at those
   locations. Retrieved features enter the curve evidence transformer before
   prefix aggregation. There is one prefix-confidence readout, without a separate
   additive memory-logit head. The scorer never receives the generator's decoder
   tokens; supplied candidates and generated paths use the same scorer.

All coordinates use trace-grid voxels. The production crop is 120x101x101 at
spacing 0.5; output points occupy fixed forward planes 1 through 16. Point
self-attention, crop bounds and the first-point recovery limit are unchanged.

## Retention and gradients

Memory contains an immutable seed observation, the latest 64 observation grids,
and 16 persistent learned slots. Spatial tokens retain position, orientation, age
and validity, and remain readable outside the current crop. Every observation is
cached, including observations made after a wrong turn.

Slots compress observations using their existing learned update gate. There is
no fiber-membership admission classifier, admission metadata, identity-based
write multiplier, offset probe, or auxiliary memory-probe loss. The writer
receives independent observations and the previous slots; retrieved historical
interpretations are not written back as new observations.

Path and confidence objectives train memory retrieval and compression. The
visible-reference contrastive objective remains an encoder objective. Confidence
detaches scored coordinates and has no gradient path into the coordinate head,
trajectory decoder or refinement head. It still trains the shared encoder and
memory. Geometry supervises both the initial and refined curves (25% / 75%).

Training carries memory across two-decision gradient chunks. Revision-2 endpoint
replay retains its age-stratified historical re-encoding and differentiable
writer transitions. The replay objective has geometry, confidence and candidate
terms, without memory-probe terms. Other historical features remain detached,
potentially stale representations; this is not full-history encoder BPTT.
The seed persists after cache eviction, and slots can retain still older evidence.

## Fresh training

From `fiber_follow`, using the existing environment:

```bash
bash scripts/launch_trajectory_memory.sh
tail -F output/logs/axial_observation_memory_v4_run1.log
```

The launcher starts a fresh run with direction inputs, batch 8, microbatch 4,
eight workers, 16-point maximum commit, history spacing 8, revision 2 and one
refinement pass. It refuses an existing destination. Override `RUN_NAME` to
start another experiment; trailing trainer options override launcher settings.

Matched decision endpoints increase from 10% to 30% of sampling opportunities,
and synthetic memory switches increase from 15% to 30% of eligible fresh draws.
These are endpoint probabilities, not guaranteed per-crop fractions: causal
streams expand into different numbers of observations. Logs report actual
candidate-supervised crops and known departed confidence states across every
ordinary training crop. Endpoint replay is reported separately. These settings
need validation on actual branch-choice and departure outcomes.

No old checkpoint should be passed with `--resume` or `--init-tracer`.
New checkpoints from this implementation can resume normally. There is no
admission/offset loss to tune; the launcher sets `--memory-probe-weight 0`.
The shared trainer retains probe options for the other memory architectures.

## Validation

Run the CPU architectural and training checks without installing dependencies:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=../../.. \
../../../../.venv/bin/python -m pytest \
  tests/test_detailed_memory.py tests/test_trajectory_memory.py \
  tests/test_recurrent_refinement.py \
  tests/test_point_logging.py tests/test_sampling_balance.py \
  -q -o cache_dir=/tmp/fiber-observation-pytest
```

Tests check independent stored observations, memory-conditioned fine features,
candidate-specific reads, generator-independent scores and gradients,
long-range compression gradients, immutable seeds, cache masking, cold/warm
tracing, chronological replay, training updates and full-graph capture.
The CUDA test additionally compares eager and compiled BF16 gradients when
a working GPU environment is available.

These implementation tests do not establish tracing accuracy, convergence,
production GPU memory use or throughput for the revised architecture.
