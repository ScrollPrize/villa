# Local Fiber3D metric embeddings

## S1a eight-GPU training

Use `fiber_trace_3d/configs/train_s1a_nml_embedding_16x8.json` for a fresh run.
It preserves the merged fast S1a baseline (8 NML files, one held-out JSON,
Paris4 input level 1, 128³ patches, GroupNorm, stride 4, BF16, 16 samples/GPU,
global batch 128). The only config changes are the run name, embedding settings,
and the matching local Paris4 Lasagna normals manifest for train/test.
The normals manifest selects level 4; coordinate mapping uses its base shape
and scale independently of the level-1 input.

LR warms to 0.01 over 100 updates. This was the prior short-run recommendation,
not a retuned optimum for embedding loss. Do not resume a legacy non-embedding
checkpoint. GPU memory and throughput with embeddings remain unmeasured.

Production training must use the cooperative manager. Do not launch with
`torchrun`: the CPU supervisor starts and reconfigures GPU workers itself.

No manager configuration is necessary. The command below connects to an existing
manager or starts one automatically, and derives its CPU/GPU/VRAM request from
the available inventory, GPU visibility and measured calibration. The manager
outlives the initiating task; simultaneous starters reuse one service.

Run the [manual memory calibration](cooperative_training.md#memory-probe--run-only-after-gpu-permission)
once for this embedding config, omitting `--output`. Its report is saved under
`/ephemeral/hendrik/fiber3d_train2/runs/.memory-calibrations/<config-fingerprint>.json`
and loaded automatically. No copying JSON into the config is required. Missing
or stale calibration produces an error before a service or training is started;
old non-embedding measurements cannot be reused.

Then run from any directory:

```bash
export PYTHONPATH=/home/hendrik/villa/gpu_manager/src:/home/hendrik/villa/vesuvius/src:/home/hendrik/villa/vesuvius:/home/hendrik/villa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

/home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.train \
  /home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_embedding_16x8.json
```

The manager chooses physical GPU UUIDs and measured microbatch sizes; global
batch remains 128 through accumulation. Do not force a static eight-rank launch.
The default socket is `/run/villa-gpu-manager/manager.sock` if its location is
writable/searchable or a listener is reachable, otherwise
`~/.local/run/villa-gpu-manager/manager.sock`;
`--manager-socket` is only needed to override it. Only explicit `--unmanaged`
may bypass the manager for testing/debug; it is not a production command.
With the PYTHONPATH above, query the existing manager without installing anything:

```bash
/home/hendrik/.venv_las/bin/python -m villa_gpu_manager.cli status
```

Output: `/ephemeral/hendrik/fiber3d_train2/runs/s1a_nml_embedding_128_sd1_16x8_<timestamp>`.
The explicit PYTHONPATH selects this embedding checkout instead of another
editable-install worktree. The config contains machine-local data paths.

Add this to an existing **unconditioned** Fiber3D training config:

```json
"embedding": {
  "enabled": true,
  "channels": 16,
  "pairs_at_128": 65536,
  "weight": 1.0
}
```

Each train/test dataset must also have `lasagna_manifest_path` pointing to its
matching normals manifest (grad_mag/nx/ny, with original base shape). Existing
dataset transforms and selected input scale continue to apply. Missing or invalid
normals are errors, not spherical/uniform fallbacks. Relative paths resolve against
the training config. No normals download convention or cache format was added:
the existing Lasagna channel loader opens and caches these channels.

Use the managed launch above with the modified config. It was **not** run as part
of CPU validation. Cooperative training requires a fresh matching memory
calibration: the extra output channels, pair graph and two extra FP32 gradient
banks change its memory footprint. Existing checkpoints without an embedding head
cannot be strictly resumed into the new head; ordinary legacy configs still work.
The two added banks alone require `8 * trainable_parameter_count` bytes. An
eligibility slot needs one byte per output voxel (32 KiB for a 128 input patch
with stride 4). Pair/autograd storage is additional. Bounded training performs
two additional component VJPs; its GPU training cost must be measured rather than
assumed to match the old three-component backward.

## Objective and geometry

- 32,768 positive and 32,768 negative pairs at input width 128. Total pair count
  grows linearly with maximum input dimension, not output voxel count.
- Positive endpoints are sampled independently by arc length along the patch's
  annotated trace; presence does not affect them. Sampling is with replacement.
- Negative endpoints are output-grid voxels sampled proportional to **detached**
  presence after exclusion. For multiple branches, use maximum branch presence.
  Probability 1 has ten times the sampling weight of 0.1; zero is never chosen.
- Map output coordinates through the existing inverse augmentation into base
  coordinates; find the nearest continuous GT segment point and sample its
  Lasagna surface normal. Exclude the inclusive ellipsoid with radii **20 base
  voxels normal / 80 base voxels tangential**. Annotation transforms, nonzero
  source scale, native output stride and nonlinear augmentation are included.
- The added loss is `weight * (mean(1-cos_positive) + mean(relu(cos_negative)))/2`.
  Means use separate actual pair counts across the effective batch. No negative
  support contributes zero negative loss; the positive coefficient stays 1/2.
  No nonzero GT segment is an error. Pairs never cross samples.

The eligibility grid is built by CPU loader workers in blocks, then travels in
bounded shared slots and the existing pinned/CUDA preparation path. A conservative
midpoint spatial index accelerates exact nearest-segment projection; unresolved
cases use the bounded scan, not an approximate nearest-vertex assignment. Pair
sampling, embedding lookup, FP32 cosine arithmetic and gradients run on the model
device. U-Net AMP policy is unchanged. The loader's 128-bit stream seed is
reduced modulo `2**63 - 1` at the PyTorch generator boundary, matching the
existing augmentation convention. Stream-seeded sampling is stable under
microbatch/rank repartition for identical predictions; floating-point changes to
presence can still change stochastic choices near a sampling boundary.

Training/test output includes `loss_embedding` and negative pair support.
TensorBoard includes `train/embedding`, `train/embedding_positive`,
`train/embedding_negative`, `train/embedding_negative_pairs` and their `test/`
counterparts. The component scalars already include the overall embedding weight.

## Extracting embeddings

```python
result = model(volume, return_embeddings=True)
prediction = result["prediction"]  # existing sigmoid 7*K channels
embeddings = result["embeddings"]  # signed, B,E,D,H,W
embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
```

Default `model(volume)` and ordinary volume inference still return the established
direction/presence predictions. This change does not add embedding Zarr export or
Atlas publication. Conditioned-decoder embeddings are explicitly rejected.

## Validation (CPU only)

From the Villa root, using the existing venv:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_fiber_embeddings.py
```

The suite includes real local Zarr/manifest loading (input level 1, normals level
2), annotation translation plus actual rotation/warp, geometry boundary and exact
projection checks, detached sampling ratios, zero support, shared-slot transport,
checkpoint reconstruction, ordinary-versus-bounded gradients, and a three-process
Gloo reduction with a zero-work rank. It requires local sockets, but no GPUs.

The small synthetic benchmark uses one CPU thread, 128 input width, native
quarter-resolution output, 16 channels, 65,536 pairs, one pair-loss warmup and
three measured repeats. The midpoint-index change reduced exclusion median from
0.363 s to 0.161 s (min/median/max 0.161/0.161/0.162 s); pair forward+backward
was 0.123/0.128/0.131 s. These are CPU wall seconds **per patch on one CPU
thread**, not GPU times or aggregate worker throughput. Constant normals exclude
disk I/O in that timing. Existing release NumPy/SciPy/PyTorch wheels were used.

GPU overhead, real-dataset end-to-end throughput and embedding quality have not
been measured. Do not infer a negligible training cost from the CPU tests.
The vectorized training ellipse is a separate NumPy implementation of the native
replay formula; shared normal interpolation is reused directly. Analytic/parity
geometry tests cover the formula, but no new native binding was built.
