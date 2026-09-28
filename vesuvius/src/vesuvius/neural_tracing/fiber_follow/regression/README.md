# Crop-only axial fiber follower

This model follows a marked fiber through a single heading-aligned CT crop. The
current architecture is `axial_fiber_v3`; previous direct/identity architectures
and their checkpoint loaders have been removed.

## Architecture

The input is CT plus presence from the existing `s1_ds2` volume. The fine crop is
120 × 101 × 101 samples, with its head at depth index 48 and spacing 0.5 trace-grid
voxels (one finest-level CT sample). Its physical extents are ±25 trace voxels
transversely and −24 through +35.5 along the forward axis.

A residual 3D CNN extracts 32-channel full-resolution features. A stride-two
stage produces 64 channels, followed by 128 channels and another residual CNN
block before compression. A learned 4 × 1 × 1 convolution, stride four along
forward, produces 128-channel tokens with effective spacing 8 × 2 × 2 CT samples.
The token grid is 15 × 51 × 51: 39,015 tokens. Compression centres are at input
coordinates `(8*z + 3, 2*y, 2*x)`. Spatial sampling and dense reconstruction use
this exact lattice, including its depth offset.

Four residual axial blocks apply full bidirectional attention along X, Y, and
forward/backward, followed by one expansion-four MLP and a depthwise 3D convolution
with pointwise channel mixing. Four heads are used. Physical positions and
rasterized observed history/seed cues condition the tokens. Ground-truth fiber
membership never enters the model inputs.

A lightweight decoder upsamples contextual features and combines them with the
full-resolution stem skip. One shared 32-channel dense feature map supplies localization,
correction, confidence, and the contrastive projection. Four path-decoder layers
predict the existing 16 future points at one trace-voxel spacing. Two bounded
corrections refine the curve. The same prefix confidence classifier scores
predictions and explicitly supplied candidate curves. The default model has
4,464,453 parameters.

The confidence head receives explicit cosine comparisons between each candidate
point and two separate references: the visible observed seed, and pooled visible
recent history. Each has point similarity, prefix mean, prefix minimum and valid
coverage. No annotation membership is used for these inputs. Missing references
have zero evidence and zero coverage; contaminated history cannot average away
the separate seed comparison. Predicted paths and supplied candidates use this
same head. The eight added inputs start with zero weights and learn through
prefix/candidate BCE, with gradients also reaching the shared embedding projection.

There is no coarse image branch, separate appearance encoder, remote CT patch
reader, or persistent embedding cache. History and seed features are sampled only
inside the current crop. References outside it have their position, orientation,
age and features masked out before they can influence the network. Seed metadata
is retained in rollout/replay records so that visibility can be determined later;
this does not expose out-of-crop seed information to the model.

## Training and supervision

The full existing annotation set, holdout split and live neighbor bank are used.
The default launcher uses `output/neighbor_samples_r0_32_l80_160_v2`; its miner can
continue publishing shards independently. All contrastive positive and negative
queries must lie inside the main crop, with the same support and presence checks.
No remote positive or negative patches are encoded.

Losses combine direct curve regression, prefix confidence, matched candidate BCE,
and auxiliary InfoNCE computed from the shared dense features. Observed history
references on the labeled fiber are pooled for InfoNCE; a visible original seed
can supply the reference when history is insufficient. Membership annotations
select supervised terms only and are never supplied as input cues.

Matched cases use identical current CT/history and different **visible** observed
seeds. Their short 4/8/12 trace-voxel tails leave both references inside the crop.
Old departure states with no visible original-fiber reference have identity and
confidence supervision masked, rather than receiving contradictory labels for
indistinguishable inputs. Geometry is already masked for confirmed departures.
The `identity_observable_fraction` metric records this coverage.

This model learns local visual continuity. It cannot verify original-seed identity
once every distinguishing observation has left its crop. Full tracing evaluation
still measures departures from the original annotation, including this case.

## Launch and logs

```bash
bash scripts/launch_axial.sh

tail -f output/logs/axial_crop_822_run2.log
```

The launcher starts a fresh 100,000-update experiment: effective batch 16,
microbatch 4, 12 loader workers, BF16, torch.compile, no activation checkpointing, AdamW at 3e-4, 500 warmup
updates and EMA 0.999. It refuses to overwrite a run. Set `RUN_NAME` and `BANK_PATH`
to change destinations. Extra training options can be appended.

Checkpoints and structured logs are written under `output/axial_crop_822_run2`.
The current architecture can resume its own checkpoints with the same run options
and `--resume PATH`; there are no old-model migration paths. Frozen monitor seeds,
recovery probes, online replay collection and neighbor-aware evaluation remain in
use. Compare coverage, precision, departure count, wrong-fiber length and candidate
acceptance/rejection, in addition to losses and contrastive ranking.

## Validation

No dependency installation is required. This workspace already has a training
Python environment at `../../../../.venv/bin/python` and an existing test environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MPLCONFIGDIR=/tmp/axial-mpl PYTHONPATH=../../.. \
/home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest tests -q \
  -o cache_dir=/tmp/axial-pytest
```

Use the real-data preflight to check crop support, visible references, bank
negatives and forward/backward gradients at the production crop dimensions:

```bash
PYTHONPATH=../../.. ../../../../.venv/bin/python \
  -m vesuvius.neural_tracing.fiber_follow.regression.identity_preflight \
  --out output/axial_preflight --device cuda --forward --batches 4
```

This is a batch validation, not a reduced training experiment. The preflight saves
its first batch and a JSON report. `tests/test_axial.py` covers the physical token
lattice, bidirectional global context, within-patch forward order, reference
visibility, supervision masks, gradient flow and current-checkpoint round trips.

## Measured implementation cost

The initial and kernel-optimization measurements below describe the earlier
16-channel model. The current default's capacity comparison follows them.

On the RTX 5090, BF16, microbatch 4, torch.compile, four warmup iterations and
12 measured full forward/backward/AdamW iterations on seeded representative
tensors (no data I/O):

| Model | Mean update | Median | P95 | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| Previous encoders and identity heads | 65.6 ms | 65.4 ms | 68.2 ms | 4.07 GiB |
| Axial crop-only, no activation checkpointing | 200.8 ms | 201.6 ms | 204.1 ms | 9.72 GiB |

Before kernel optimizations, the measured GPU update cost was about 3.06x,
higher than the earlier MAC estimate.
A real-data profiler identifies masked path cross-attention backward and grouped
3D convolutions as substantial costs. Activation checkpointing reduces memory
but adds recomputation; it is available as an option and disabled in the launcher.
These timings do not include CT reads, workers, monitoring or online collection.

Reproducible benchmark scripts, JSON measurements, a real-data optimizer/checkpoint/
trace integration report, and the profiler table are saved under
`output/axial_implementation_validation/`. The benchmark uses the previous run's
saved source snapshot for its baseline; no old-model loader remains in production.

### Kernel optimization measurements

The two largest avoidable costs have been reduced without changing crop size,
parameters, precision or training objectives. Depthwise convolutions now use
contiguous inputs **and weights** locally; ordinary convolutions retain their
channels-last layout. A singleton weight dimension made an input-only conversion
ineffective. Masked path-decoder cross-attention prefers cuDNN, retaining other
PyTorch backends for unsupported devices/dtypes. Axial attention is unchanged.

On the same RTX 5090/PyTorch 2.12.1, a saved real CT batch of four states, BF16,
torch.compile, complete forward/loss/backward/gradient clipping/AdamW, five warmup
iterations and 12 measured iterations, excluding data I/O:

| Axial implementation | Mean | Median | P95 |
|---|---:|---:|---:|
| Before kernel changes | 206.4 ms | 206.7 ms | 211.6 ms |
| Depthwise layout only | 174.0 ms | 174.1 ms | 177.5 ms |
| Masked attention only | 152.2 ms | 152.4 ms | 154.5 ms |
| Both | 119.3 ms | 119.1 ms | 121.5 ms |

This is 42.2% less GPU step time, or 1.73x throughput for this workload. It is not
an end-to-end data-loading or tracing throughput claim. Different BF16 reductions
are not bitwise identical: on the initial comparison batch the combined change
altered total loss by 0.000162, confidence probabilities by at most 0.000305 and
predicted coordinates by at most 0.0177 trace voxels. CPU double-precision and GPU
BF16 tests check outputs, input/parameter gradients and masked-reference invariance.

Reproduce the measurements from the fiber-follow directory with the GPU idle:

```bash
PYTHONPATH=../../.. ../../../../.venv/bin/python \
  output/axial_speed_validation/benchmark_full_step.py
```

The benchmark's fixed input, frozen baseline source, profiler tables, individual
kernel probes and numerical comparisons are recorded under
`output/axial_speed_validation/`. No extra packages or custom CUDA kernels are
needed. See [TRAINING_REVIEW.md](TRAINING_REVIEW.md) for the separate training
simplification review; those proposed training changes have not been applied.

### Current default: wider CNN with residual processing before compression

The default is now 32 → 64 → 128 CNN channels with a residual block on the
stride-two 128-channel grid immediately before learned forward compression.
The axial width, depth, crop dimensions and token spacing remain as above.

Measured with the optimized kernels on the same RTX 5090, BF16, torch.compile,
real CT microbatch of four, five warmups and twelve full measured optimizer steps:

| Configuration | Parameters | Mean | Median | P95 | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|
| Earlier 16-channel model | 2.93M | 116.5 ms | 117.3 ms | 118.1 ms | 9.72 GiB |
| Current default | 4.46M | 180.1 ms | 181.4 ms | 186.0 ms | 14.48 GiB |

The increase is 54.6% in GPU step time and 4.75 GiB in allocated memory for this
workload; data I/O, EMA and monitoring are excluded. Every measured step had a
finite loss and gradient norm. This establishes compute cost, not model quality.
The fixed input, frozen model source, script and measurements are under
`output/axial_capacity_validation/`. Reproduce with:

```bash
PYTHONPATH=../../.. ../../../../.venv/bin/python \
  output/axial_capacity_validation/benchmark_stem.py
```
