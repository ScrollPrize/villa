# Eight-GPU S1a NML retraining

Current comparison (2026-09-10): LR **0.001**, label **s1a_repro1_lr0p1x**,
resuming the same `s1a_repro1_20260909_193241/snapshots/current.pt`
20,000-update checkpoint as the extended LR 0.01 continuation. The limit
remains 2,000,000 total updates. This supersedes the earlier current-config
notes below. The resume loader restores optimizer state, then applies the
configured LR; the already completed warmup is not restarted.

Latest continuation: the run `s1a_repro1_20260909_193241` reached its configured
20,000-update limit (2,560,000 samples). The config now allows 2,000,000 total
updates (256,000,000 samples at global batch 128), a 100× increase. Resume with
`--resume /ephemeral/hendrik/fiber3d_train2/runs/s1a_repro1_20260909_193241/snapshots/current.pt`.
LR remains 0.01. Earlier run-length/checkpoint references below describe the
initial reproduction stages.

Current config: restored peak LR **0.01** and run
label **s1a_repro1**, with the same 100-update warmup. The original
reproduction settings and LR-tuning recommendation described below used 0.01.
The launch command is unchanged; new outputs are under
`/ephemeral/hendrik/fiber3d_train2/runs/s1a_repro1_<timestamp>`.

To continue the first run, append
`--resume /ephemeral/hendrik/fiber3d_train2/runs/s1a_repro1_20260909_160150/snapshots/current.pt`.
This checkpoint is update 11,900 (1,523,200 samples, inferred from its saved
global batch 128). It restores optimizer/model state and continues to update
20,000 in a new timestamped output directory. The original run's event files
remain in update units; the continuation uses the new sample axis.

Config: `fiber_trace_3d/configs/train_s1a_nml_8gpu_fast.json`, using the measured shared sample pipeline, async CUDA preparation, and full CUDA value augmentation. Compare with existing historical runs; no new standard-loader run is configured.

The config uses the exact datasets, affine transform, augmentation settings, held-out data and seed from `training_baseline_s1a_20260729.json`: 8 S1a NML files and 1 held-out JSON, Paris4 level 1, 128³ input, GroupNorm, output stride 4, BF16, eager execution, 16 samples/GPU, global batch 128. Run length is 20,000 optimizer updates (2,560,000 samples). LR warms linearly to 0.01 over 100 updates then stays constant. Clipping is disabled. This is the recorded best short fixed-LR verification, not the LR-range minimum (0.08125). Tuning predates global loss normalization, so the recommendation needs this longer verification.

The fast topology uses 128 total pipeline workers, 87 load-affinity workers, 2048 shared slots (256/rank), initial in-flight 128 (16/rank), and CUDA prefetch depth 2. CPU/CUDA augmentation is tolerance-equivalent, not bitwise identical.

## Launch

Production runs must use the cooperative manager, not a fixed-rank launcher.
The baseline JSON needs a `training.cooperative` resource request and matching
measured calibration before launch; see [managed setup](cooperative_training.md#managed-launch).
For the embedding variant use [the S1a embedding command](local_fiber_embeddings.md#s1a-eight-gpu-training).

```bash
export PYTHONPATH=/home/hendrik/villa/gpu_manager/src:/home/hendrik/villa/vesuvius/src:/home/hendrik/villa/vesuvius:/home/hendrik/villa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

/home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.train \
  /home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_8gpu_fast.json
```

Timestamped outputs are under `/ephemeral/hendrik/fiber3d_train2/runs/s1a_repro1_<timestamp>` for this first reproduction test. Evaluation, scalar logging and checkpoint saving remain every 100 steps; visualization every 500; retained snapshots every 10,000. Resume by appending `--resume /absolute/path/to/checkpoint`.

## Interpretation

New TensorBoard scalar/image steps are cumulative global training samples.
Console metrics show `samples=12800 update=100` after 100 updates at this
global batch of 128. Evaluation and visualization use the same sample axis;
evaluation samples do not count toward it. Warmup, stopping, interval settings,
checkpoint filenames and internal profiling/diagnostic steps still use updates.
Checkpoints save cumulative samples, preserving the axis when resuming with a
different batch size. A legacy checkpoint without a saved sample count or
global batch requires `training.resume_samples_seen` in the config. Existing
TensorBoard event files retain their old update axis: convert those separately
when plotting against new sample-axis runs; they do not become comparable
automatically just by changing the new logger.

This run uses the current trainer, including corrected global loss normalization. The July historical run used BatchNorm, output stride 1 and batch 8/GPU (global64), versus GroupNorm/stride4/global128 here. Thus historical loss is useful context, not a pure speed comparison; coarse output targets also differ. At 20,000 steps this run sees twice as many samples as the historical global64 setup. Compare held-out loss against both elapsed wall time and samples consumed, not only update number. The unchanged reference artifact records historical best held-out loss 1.017569 at update 17,700.

## Validation

The JSON config parses through `load_config`, instantiates a 28,523,527-parameter model, resolves all source globs, and has valid eight-rank pipeline partitions. Native `vc.volume.generate_fiber3d_coords` and `fiber3d_value_transform` import successfully. Config loading was also checked from `/ephemeral/hendrik/fiber3d_train2`. Eight relevant existing tests pass (run this test command from the repository root):

```bash
PYTHONPATH=/home/hendrik/villa2/vesuvius/src:/home/hendrik/villa2/vesuvius:/home/hendrik/villa2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'linear_warmup or training_cuda_value_augmentation_validates_suffix or config_defaults_volume_cache or config_preserves_volume or distributed_config or mixed_precision_config_parsing_cpu'
```

Validation did not launch eight-GPU training, fetch volume chunks, or measure new throughput.
