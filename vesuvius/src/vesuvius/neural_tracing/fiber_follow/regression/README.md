# Axial fiber follower

## Model and memory

The direct regression model is `axial_fiber_spatial_memory_v2`, implemented by
`DirectFollower` in `model.py`. It uses a shared axial encoder, point-based curve
queries, and spatial observation memory. Memory is always enabled; its capacity
and training schedule are configurable.

Each observation retains its complete axial token grid (39,015 × 128 at the
production crop), its pose and age, and one fine descriptor at the observed
head. The descriptor preserves the supervised identity space and fine detail
from the encoder's full-resolution skip connection; the grid supplies spatial
context. Candidate queries likewise use one fine descriptor, one support flag,
and the point's coordinates. Geometry, corrections, confidence and historical
probes use this same point-based evaluator. Supported points may reach the crop
edges. The dense decoder computes the fine descriptors for every observation.

The seed grid is immutable and always readable, with the seed origin/direction
defined by its observation frame. Current image tokens, visible observed-path
tokens with explicit ages, and the two most recent historical grids are always
readable. A learned summary of each older grid supplies an archive retrieval
key. Each candidate's seed-conditioned path queries score those keys. The
highest-scoring two observation blocks are read at full spatial resolution,
with separate attention weights for each path point. A soft read over all
archive summaries trains the routing parameters, including unselected entries.
Selection is bounded per curve, not an independent top-k for every point;
stable ties prefer newer observations. Geometry, corrections, confidence,
candidate evaluation and historical probes use the same evaluator.

Defaults retain two recent observations and eight additional older grids.
The current head occupies a separate bank slot and is not duplicated among
the recent history when decoding it. Oldest archive entries are evicted first.
Sixteen learned recurrent slots read the full observation grid and preserve
compressed context after eviction. Evicted spatial detail is not recoverable:
this is a bounded archive, not an unlimited historical store. Capacities are
explicit via `--spatial-recent`, `--spatial-archive`, `--spatial-retrieve` and
`--memory-slots`. World poses remain float32; feature storage follows the
model's parameter dtype, independent of the surrounding autocast context. The default bank plus
seed alone is about 229 MiB per trace in float32, before activations or temporary
copies (about 114 MiB in BF16).

### Training windows and feature lifetime

`--trajectory-window 4` supervises up to four consecutive decisions for states
with explicitly labeled observed tracks. The original sampled state is always
included. States without suitable track labels and matched-candidate states
retain their original single-state supervision. Extra decisions use only the
observed track prefix, the original seed, and separately constructed original-
fiber targets; target offsets/departure labels never enter model inputs. Every
extra state's crop/target footprint is checked against the holdout before I/O.

All crops are sampled once into the original batch. Window images are views
of those buffers and share its photometric augmentation. The first decision
encodes the seed and earlier history; subsequent decisions encode only their
new crop and carry differentiable memory forward. Activation checkpointing
can recompute encodings during backward. Losses average over decisions within
each window, then over originally sampled states, so longer tracked windows do
not increase their source's loss weight or dilute matched-candidate examples.
Logs include `supervised_decisions`; detailed decision diagnostic groups count
the actual decisions evaluated.

**No learned encodings survive a training optimizer update.** Every window
starts with empty carried memory. All of its encodings use the current weights;
one backward pass accumulates its losses, and memory is discarded before the
optimizer steps. The next update re-encodes both seed and history. Burn-in and
stratified encoder-gradient sampling still compute fresh forward features;
`no_grad` is not a feature cache. Checkpoints contain weights/optimizer/RNG,
not observation memories. Replay stores observed geometry and labels, not old
encoder activations. Inference carries features within a trace while its loaded
model weights stay fixed; a new trace/checkpoint starts new memory.

### Running and validation

To launch a **new** experiment after reviewing the change and completing a
production-device preflight:

```bash
bash scripts/launch_spatial_memory.sh
```

Its default destination is `output/axial_spatial_memory_run1`; set `RUN_NAME` to
choose another destination. The launcher refuses an existing run directory.
`bash scripts/launch_regression.sh NAME [options]` exposes the same model with
trainer defaults. `--init-tracer output/RUN/last.pt` initializes a fresh run from
saved EMA weights. `--init-encoder output/RUN/last.pt` transfers only the encoder
and compatible identity projection; memory, path decoder and optimizer start
fresh. `--resume output/RUN/last.pt` continues a run with its optimizer, RNG state,
and matching training options. Checkpoints must use the direct architecture and
its current configuration schema.

Production preflight (also exercises the window optimizer with zero LR):

```bash
PYTHONPATH=../../.. python -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight \
  --out output/spatial_memory_preflight --device cuda --forward --compile \
  --microbatch 2 --batches 2 --memory-slots 16 \
  --memory-steps 64 --memory-grad-steps 32 --memory-encoder-grad-steps 4 \
  --spatial-recent 2 --spatial-archive 8 --spatial-retrieve 2 \
  --trajectory-window 4 --memory-switch-probability .15 --activation-checkpointing
```

Tests cover point descriptors and crop-edge support, remote spatial evidence, immutable seeds, archive
eviction, streaming/unroll parity, relative poses, candidate isolation,
padding/empty memories, gradients through retrieval, checkpoint loading,
window supervision/holdout/crop reuse, and re-encoding after parameter updates:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_spatial_memory.py \
  tests/test_memory_training.py tests/test_memory_data.py tests/test_memory_sequences.py \
  -q -o cache_dir=/tmp/spatial-memory-pytest
```

This is an architectural implementation, not a measured accuracy or speed
improvement. Production CUDA memory, throughput, compilation and tracing quality
still require validation before a long run. The encoder/decoder compile in place;
archive routing and the CPU observation/window schedule remain eager.

## Training and evaluation

The encoder consumes fine CT and presence, with a default 120 × 101 × 101 crop
at spacing .5. Axial tokens have an 8 × 2 × 2 sample stride; a full-resolution
skip connection feeds the dense feature decoder. Observed geometry and seed
identity enter the memory reader, not the image encoder. The model predicts
16 forward points and performs two bounded lateral corrections by default.
Confidence scores the resulting prefix; supplied candidate curves use the same
reader and confidence head without writing to memory.

Losses combine direct curve regression, prefix confidence, matched candidate
BCE, pointwise identity features, and per-observation departure/offset probes.
InfoNCE pools labeled original-fiber visible history references, falling back
to the immutable seed descriptor when history is insufficient. Annotation
membership selects supervised terms only; the model sees every observed crop,
including contaminated history. Unknown annotation endings are censored;
confirmed departures have no geometry target. Departure supervision requires
an observable original-fiber reference in current or historical observations.

The default launcher uses `output/neighbor_samples_r0_32_l80_160_v2`. Its miner
can publish shards independently. Contrastive query points must lie inside the
current crop and pass the presence and neighbor-validation checks. Matched
candidate examples share a crop/history but have different observed seeds.
Memory-switch examples retain an original-fiber prefix followed by a neighboring
fiber tail. Replay retains observed tracks; input crops and labels are rebuilt
with full holdout checks before I/O.

Training logs report geometry, confidence, identity and memory-probe
diagnostics. Checkpoint and evaluation entry points are:

```bash
PYTHONPATH=../../.. python -m vesuvius.neural_tracing.fiber_follow.regression.train --help
PYTHONPATH=../../.. python -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight --help
python scripts/evaluate_regression.py --help
python scripts/evaluate_regression_recovery.py --help
```

The training window never caches learned features across optimizer updates.
Inference retains them within one trace under fixed model weights. Deterministic
CPU fixture tests cover sampling, memory eviction, geometry and gradient paths;
production tracing quality and GPU throughput require measurement on real data.
