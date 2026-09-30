# Axial fiber follower

For continuous path prediction with observation-only memory, memory-conditioned
fine features, curve-specific confidence and one shared-decoder refinement pass,
see [trajectory memory v4](TRAJECTORY_MEMORY.md). Its launcher starts a fresh
revision-2 run; checkpoints from the former admission-gated revision 2 cannot
resume this architecture.

The training launcher uses `axial_fiber_memory_v2`, the model from
`output/axial_memory_seq_run1`. The model, memory writer, observation construction
and losses were restored from revision `28f69ae6a`. Its saved configuration has
4,803,129 parameters and its checkpoint loads strictly, without renamed or missing
weights. A fresh launch initializes random weights; it does not load that checkpoint.

## Model

The current 120 × 101 × 101 CT/presence crop at spacing .5 passes through the axial
encoder once. Visible observed history and the seed condition the spatial tokens.
One curve decoder predicts 16 forward points. Smaller local heads apply two bounded
lateral corrections and predict prefix confidence. Supplied candidate curves use
that confidence head without writing hypothetical observations to memory.

Historical CT uses a separate small encoder: two stride-two 3-D convolutions over
17 × 17 × 17 CT/presence patches, pooled to eight 128-channel observation tokens.
A gated recurrent writer updates 16 learned memory slots. Eight seed tokens remain
immutable for the trace. Relative motion and seed pose enter the writer, and curve
queries read the slots and seed through attention.

Training reconstructs up to 64 historical observations, plus the current head.
Reconstructed observations are spaced four trace voxels apart. Recorded replay
tracks retain their actual observation positions. The newest 32 historical writes
and the current head backpropagate; earlier writes build the state without gradients.
An auxiliary per-write head predicts departure from, and offset to, the original
fiber. Its targets never enter model inputs. The memory read initially contributes
zero; the auxiliary head trains the writer from the first update.

Each sampled state rebuilds memory with current weights. During tracing, each trace
carries its own slots and immutable seed, and subsequent decisions read only the new
head patch. Replay stores observed geometry and supervision rather than learned
features. Crop, seed and historical patch footprints are checked against holdouts.

Geometry, prefix-confidence, candidate, visible-reference identity and memory-probe
losses match the original run. CT features outside the current crop reach the model
through recurrent patch memory. The explicit identity loss uses visible references.

## Training

From `fiber_follow`, using the existing project environment:

```bash
bash scripts/launch_memory.sh
tail -F output/logs/axial_memory_seq_run2.log
```

The launcher uses effective batch 16, microbatch 4,
12 loader workers, learning rate .0003, 500 warmup updates, 100,000 total updates,
BF16 CUDA autocast outside memory observation/writing, compilation, and no
activation checkpointing. Memory patch encoding, burn-in, recurrent writes and
auxiliary probes run in FP32 during both training and tracing; memory reads remain
under the caller's autocast. After accumulation, gradient norms are clipped
independently: recurrent memory at 5, all other parameters at 20. Memory spikes
therefore cannot scale encoder/decoder gradients. `--memory-grad-clip` and
`--rest-grad-clip` override these limits; 0 disables that group's clipping while
retaining finite-gradient checks. Both options may be changed on resume and are
recorded in checkpoint training options. The terminal and JSON training logs
report `memory_grad_norm`, `rest_grad_norm` and each group's `*_grad_clip_scale`;
`grad_norm` is the combined pre-clipping norm, not a global clipping threshold.
It retains the original data paths, split, neighbor bank, sampling, recovery monitoring and online replay
settings. Existing shared sampler optimizations and interval timing logs are retained.
The live neighbor bank can grow; launches do not use an identical data snapshot
or promise bitwise reproduction.

The default destination is `output/axial_memory_seq_run2`. The launcher refuses an
existing run directory or log. Set `RUN_NAME` for another fresh run:

```bash
RUN_NAME=axial_memory_seq_run3 bash scripts/launch_memory.sh
```

Stop a named run and its workers with `bash scripts/stop.sh NAME` before launching
another. `scripts/launch_regression.sh NAME [options]` exposes the trainer directly;
its memory defaults are also 16 slots and 64 historical observations. Resuming uses
`--resume` and the original matching options. `--presence-dropout` may be changed
on resume; the sampler event and subsequent checkpoints record the new probability,
while `config.json` retains the original launch settings. Neither `--resume` nor `--init-tracer`
is used by the fresh-run recipe.

`--decision-fraction` reserves matched identity-pair slots first. For remaining
draws, `--fresh-fraction` (default 0.7) selects fresh sampling, with the remainder
allocated to current replay. Bank following, covered locations,
and memory switches are conditional on fresh sampling. Empty/rejected replay
draws can fall back to fresh sampling, so observed ratios can differ.
`--decision-fraction`, `--fresh-fraction`, and `--bank-following-probability` may
be changed on resume; sampler events and subsequent checkpoints record them.
For an older launch command, remove `--fixed-bank PATH` and use
`--fresh-fraction 0.7` for the new 70/30 base mix. Existing checkpoints remain
loadable; saved historical configuration and bank files are not rewritten.
For example, `--decision-fraction 0.1 --fresh-fraction 0.7
--bank-following-probability 0.2` requests 10% pairs, 27% current replay, and
63% fresh opportunities, including 12.6% bank following, before
rejection/fallback. Other fresh subtypes divide the remaining fresh slots.
Source allocation checks can run without pytest:
`PYTHONPATH=../../.. python -m unittest discover -s tests -p test_sampling_ratios.py -v`.

The memory launcher applies shared Gaussian blur to CT and presence on 25% of
training samples (`--blur-probability 0.25 --blur-sigma 0.5 1.25`). One sigma is
drawn uniformly per sample, in sampled crop voxels, and used for both channels,
the current crop, valid memory observations, and the seed patch. At crop spacing
0.5 this is sigma 0.25–0.625 in trace-grid voxels. Blur operates separately on
each channel with reflect padding, before CT brightness/contrast/noise and
presence dropout. Supervision and evaluation inputs are unchanged. Both blur
options may be changed on resume and are recorded in sampler events and
checkpoints. The trainer also defaults to probability 0.25; use
`--blur-probability 0` to disable blur explicitly.
Training metrics report the observed `blurred_fraction` for each logged batch.

Blur checks can run without pytest:
`PYTHONPATH=../../.. python -m unittest discover -s tests -p test_blur_augmentation.py -v`.

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q \
  -o cache_dir=/tmp/memory-pytest
```

Tests cover writer gradients, burn-in boundaries, immutable seeds, padded inputs,
coordinate transforms, streaming/reconstruction parity, active-trace ownership,
holdouts, per-write supervision, microbatch weighting and checkpoint round trips.
The full-size training launch also exercises real-data loading, compiled CUDA
forward/backward, optimizer updates and finite-loss checks.

Compiled training enables Inductor's `emulate_precision_casts`. Without it,
torch 2.12 returned encoder gradients 30-10,000 times the eager magnitude,
uncorrelated between identical repeats, whenever candidate curves were scored.
Forward values were unchanged. Small test configurations do not reproduce this;
it needs trained weights and real batches. After a torch upgrade or a model change,
compare eager and compiled gradients of one `optimizer_update` (learning rate 0) on a
saved checkpoint and captured batches. Agreement is about 1e-2 or better in bf16.

The FP32 memory path addresses a separate precision-sensitive recurrent-gradient
spike reproduced in eager BF16 on a trained checkpoint and a 65-observation replay
sample. Its memory-gradient norm dropped from about 7.6 million to 27.6 when only
observation/writing moved to FP32. This is a measured workaround, not an established
root cause or a guarantee of improved tracing quality.

Clipping defaults were calibrated at step 7,000 with FP32 memory on real effective
batches of 16. Memory still has genuine recurrent-gradient spikes. A memory cap
of 5 limits a single update's increase in the saved AdamW aggregate second-moment
energy to about 4.7%; a cap of 10 permits about 19%. The rest-of-model norms stayed
below 20 in the calibration, so that cap acts as a safeguard rather than routine
rescaling. These are measured starting values, not optimality guarantees; monitor
the per-group norms/scales as weights, batch size or loss weights change. See
`output/separate_clipping_calibration/` for the measurements and limitations.

Evaluation uses `scripts/evaluate_regression.py` and
`scripts/evaluate_regression_recovery.py` with the frozen calibration/final protocol.
Implementation checks do not establish trained tracing quality.

## Unsigned direction inputs

`--direction-inputs` adds six local-frame second-moment channels to CT/presence
in **all three observation paths**: the main crop, each causal memory patch,
and the immutable seed patch. The eight channels are
`CT, presence, uu, vv, ff, uv, uf, vf`, where `u,v,f` are the columns of the
individual patch's world-xyz frame. All subsequent encoder widths, token counts
and recurrent-state dimensions stay the same.

The loader derives `*_nx.ome.zarr/<level>` and `*_ny.ome.zarr/<level>` from the
exact selected `*_presence.ome.zarr/<level>` path. It validates shapes, uint8
encoding, zero fill, and matching OME axes/scale/origin. These fields use the
presence/trace grid, independently of the finer CT grid. Old two-channel
checkpoints and default runs never open the direction files. The model config
records the input flag; checkpoint loading and collectors reconstruct it.

Geometry contract:

- Decode source bytes as `(byte-128)/127` for world x/y and reconstruct positive
  z, then normalize to handle uint8 rounding just outside the unit disk.
  A zero byte is missing/masked data; its direction tensor is zero.
- Rotate each source direction with the **particular patch's frame**, form
  `R.T @ (n n.T) @ R`, then trilinearly interpolate its six components. This
  equals rotating the interpolated tensor because the frame is constant within
  a patch. It is invariant to replacing any source vector by its negative.
- Never interpolate encoded bytes or signed vectors. In particular, crossing
  the encoding's hemisphere seam must not invent an unrelated direction.
- Tensor off-diagonals can be negative. Do not clip them to `[0,1]`, renormalize
  interpolated mixtures, or gate them with augmented presence. Tensor trace
  retains interpolated source validity; padding remains zero.
- Blur, brightness, contrast, noise and presence dropout touch **only channels
  0 and 1**. Direction channels receive no image augmentation or dropout. An
  augmented observation pose still requires sampling at that pose and expressing
  directions in its frame; that is coordinate conversion, not direction noise.

The Lasagna checkout is not an installed dependency of this training environment.
`shared/direction_fields.py` implements its compact byte-format equation locally;
all 65,025 nonzero byte pairs were checked against that checkout's decoder.
The sampler uses an exact table of all 65,536 byte pairs (768 KiB), scalar Numba
interpolation with `fastmath=False`, and direct output buffers. This avoids
per-output-voxel array allocations/BLAS calls and a full eight-channel copy.

Start a **new run initialized from run4 checkpoint 32,000 EMA**:

```bash
bash scripts/launch_memory_directions.sh --dry-run
bash scripts/launch_memory_directions.sh
```

The default destination is `output/axial_memory_directions_run1`. Override
`RUN_NAME` or `INIT_TRACER` through the environment. Additional trainer arguments
are passed through, e.g. `--lr 0.0001`. The script refuses an existing destination.
It starts a new optimizer/schedule, retaining all old EMA weights and initializing
only the six additional input weight slices in both encoders to zero. This is
not a resume of the old optimizer. Low-precision kernels can produce small
rounding differences when the input-channel count changes, even with zero new
weights. To resume a direction-enabled checkpoint later, use the ordinary
regression launcher with `--resume`, `--direction-inputs`, and its saved options.

The wrapper matches the **checkpoint's actual training options**, which differ
from run4's original `config.json`: batch16/microbatch4, twelve workers,
commit8, presence dropout0, fresh fraction0.6, bank-following probability0.2,
decision fraction0.1, 64 memory steps with32 gradient steps, and the existing
CT/presence blur settings. No training was launched while implementing this.

Validation:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. \
  ../../../../.venv/bin/python -m unittest discover -s tests -p test_direction_inputs.py -v
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=../../.. \
  ../../../../.venv/bin/python -m unittest discover -s tests -p test_blur_augmentation.py -v
```

Tests cover sign flips, hemisphere seams, independent tensor/frame interpolation,
missing data/borders, physical grid mismatches, batched direct writes, distinct
main/history/seed frames, online/reconstructed observation parity, augmentation
isolation, weight migration, gradients and checkpoint round trips. Production-size
CUDA eager and compiled forward/backward checks also passed on real crops.
The measured eager microbatch4 peak allocated memory increased by **211 MiB**
(19.254 to19.460 GiB); this excludes optimizer state and data-loader host RAM.
Compiled direction-enabled forward/backward used15.334 GiB in the same probe.
Full training peaks can differ.

The first CPU implementation cost about230 ms per sample versus25–27 ms for
CT/presence. Profiling attributed about179 ms to the per-voxel sampling loop.
Removing tiny array allocations and BLAS dispatch, reusing exact decoded byte
values and writing channels directly into their final buffer reduced the full
observation build to about65–72 ms across six fixed real starts (one CPU thread,
warm reader caches, five timed repeats per start, main crop plus memory/seed).
The original scalar crops are unchanged. In the six-start numerical comparison,
one output component differed by7.45e-9 from the initial implementation; all
other sampled values were identical. Tests also compare against an independent
world-tensor interpolation reference. No reduced precision or approximate
interpolation was introduced. Raw timing/validation artifacts are under
`output/direction_input_validation/`; warm-cache CPU crop timings are not
end-to-end training throughput measurements.

A separate loader-plus-GPU probe used twelve workers, microbatch four, compiled
forward/backward, twelve warmup microbatches and twenty-four measured batches.
Mean time increased from **181.0 to 188.6 ms per microbatch (4.2%)**; mean loader
wait was below 0.13 ms in both cases. Parallel loading hid the additional CPU
sampling work in this probe. It repeatedly used six real locations with synthetic
observed histories and warm caches, without bank sampling, augmentation or
optimizer updates, so this is not a full training-throughput guarantee. See
`pipeline_baseline.json`, `pipeline_directions.json` and `benchmark_pipeline.py`
in the validation artifact directory. Training remains stopped.

Both direction stores were already uncompressed at levels3 and4. Metadata and64
raw chunks per channel/level were checked for memory-mapped access; no source
Zarr rewrite was necessary. Compressed stores remain supported, but are slower.

## Spatial identity memory v3

The separate [spatial memory architecture](SPATIAL_MEMORY.md) adds direct
memory-conditioned route selection and pre-write identity gating. Existing v2
models remain loadable and trainable; the new launcher opts into v3 explicitly.
