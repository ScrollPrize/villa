# Spatial identity memory (v3)

`--memory-version 3` selects `SpatialMemoryFollower` in `spatial_model.py`.
Version 2 remains the default in `train.py`; existing v2 recipes, checkpoints,
optimizer resumes and `DirectFollower` calls remain supported. Version 1 remains
loadable with its existing restrictions on further training.

## Model and training behavior

- Image features query the persistent identity slots and immutable seed on a
  lateral grid at each forward plane. The default production lattice has
  25 × 25 locations on each of 16 planes, at 2 trace voxels spacing.
- A max-sum dynamic program chooses a connected route, constraining the first
  connection by the existing recovery distance. It selects a mode rather than
  averaging separate fibers. The existing refiner moves at most half a lattice
  spacing from each selected location; the first connection is projected back
  inside the distance limit if necessary.
- The route receives its own spatial cross-entropy loss with bilinear targets.
  This loss directly trains the memory reader, writer and observation encoder.
  Missing/censored annotation crossings are masked, not labeled as absence.
- The route's probability mass around the selected cells multiplies the existing
  confidence output. The resulting prefix confidence exposes competing spatial
  modes; it is not a calibrated identity guarantee. Thresholds need re-evaluation
  after training and should not be assumed interchangeable with v2 thresholds.
- `IdentityMemory` judges each observation against the incoming identity slots,
  before writing it. The learned admission probability gates the update. The
  latest observation is retained separately for local decoding, even when its
  identity is uncertain. It is not used as a persistent identity anchor.
- Writer state transitions are shared between sequence training and streaming
  inference. Older writes are a gradient-free burn-in; recent writes receive
  route and auxiliary admission/offset supervision. Labels never enter inputs.
- V3 matched decisions have identical local crops/histories but different earlier
  observed paths and seeds outside the current crop. The v2 sampler is unchanged.
- With `--route-sequence-weight 0.5`, eligible observed tracks also provide an
  earlier supervised decision. Each causal prefix is independently re-encoded
  with current weights; later observations never enter the earlier inputs.
  Main crops, memory patches, seeds and labels retain holdout footprint checks.
  One extra decision is available per eligible primary example, so this can
  substantially increase training compute and volume reads. Set the weight to
  zero to disable this additional supervision explicitly.

Localization and permission to trace are separate: off-track states can teach
where the original fiber is while retaining the existing conservative geometry
and confidence rejection labels. This version does not add a recovery controller
or authorize a jump to a visible but disconnected fiber. Unknown identity or
missing annotation is never silently made a negative localization target.

The new persistent state has an additional `recent` tensor. A v2 live recurrent
state cannot be handed to a v3 tracer; each v3 trace initializes its own state.

## Launch and checkpoint compatibility

The new launcher inherits the existing memory recipe and enables compilation:

```bash
RUN_NAME=axial_spatial_memory_v3_run1 bash scripts/launch_spatial_memory.sh \
  --init-tracer output/axial_memory_seq_run4/ckpt_032000.pt
```

Omit `--init-tracer` for a fresh model. Migration initializes a **new run**, keeps
compatible encoder/decoder/memory weights, and initializes the route head afresh;
it does not resume v2 optimizer or step state. Existing input direction channels
are preserved, and `--direction-inputs` can still expand a two-channel checkpoint.
Use the existing `--resume` mechanism with the v3 run's original options for a
true resume. The checkpoint architecture and configuration must agree.

Programmatic construction uses `build_model(cfg)`; it dispatches to v2 or v3.
`load_checkpoint()` dispatches from the saved architecture. Direct construction
of v3 uses `SpatialMemoryFollower(cfg)`.

Controls saved in the checkpoint:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--memory-version` | 2 | Select legacy recurrent or spatial identity architecture |
| `--route-grid-step` | 2 | Maximum lateral lattice spacing in trace voxels |
| `--route-transition-radius` | 1 | Allowed lattice-cell movement between planes |
| `--route-transition-cost` | 0.25 | Squared cell-displacement penalty during decoding |
| `--route-loss-weight` | 1 | Direct spatial localization loss coefficient |
| `--route-sequence-weight` | 0.5 | Additional earlier-decision loss coefficient |

## Validation and timing

`tests/test_spatial_memory_v3.py` covers mode-preserving connected routes, initial
connection bounds, bounded refinements, route-only gradients through old
observations, pre-write admission, rejected writes, padding, burn-in,
sequence/streaming equivalence, out-of-crop matched identities, causal earlier
decisions, complete batch construction/optimizer updates, checkpoint round trips,
legacy training, direction expansion, graph capture and compiled CUDA gradients.

```bash
python -m pytest tests/test_spatial_memory_v3.py tests/test_memory_sequences.py \
  tests/test_learned_memory.py tests/test_identity_decisions.py -q
```

Use the same production checkpoint configuration and compilation mode to compare
compute costs. The benchmark reports warm-up separately and measures synthetic
inputs, excluding volume I/O and dataset construction. Training timings include
host-to-device copies; inference inputs are already on the device:

```bash
python -m vesuvius.neural_tracing.fiber_follow.regression.benchmark_memory \
  --checkpoint output/axial_memory_seq_run4/ckpt_032000.pt \
  --compile --batch 1 --warmup 3 --repeats 10 \
  --out output/axial_memory_seq_run4/spatial_memory_v3_validation/compiled.json
```

Inference measures a single streaming decision. Training includes backward,
gradient clipping, AdamW and EMA; `training_sequence` adds an earlier decision
for **every** primary sample and therefore represents full auxiliary eligibility.
Real batches can have fewer eligible sequences. No quality improvement is claimed
until the new architecture has been trained and evaluated on held-out trajectories.
