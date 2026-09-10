# Local Fiber3D embedding task log

- 2026-09-10: confirmed 65,536 pairs, 16 channels, equal positive/negative means,
  and tracer exclusion radii 20/80 in original-volume voxels.
- Discovery: native replay owns the ellipsoid; ordinary and bounded trainers share
  LossStatistics but bounded accumulation currently hard-codes three components.
- Planning/discovery only so far; no GPU tests or dependency changes.
- Independent review requested; first findings incorporated: Euclidean nearest
  GT match before ellipse, inclusive boundary, explicit legacy prediction split,
  and independent embedding numerator/count reductions.
- Cross-language deviation: native replay has no Python binding; use a tested
  vectorized NumPy formula for training rather than adding a native build/install
  dependency. Existing normal sampling will be exposed and reused, not copied.
- Implemented separate model embedding outputs (legacy 7*K forward unchanged),
  stream-seeded arc-length positive pairs and detached-presence negative pairs,
  and five-component effective-batch statistics/gradients when enabled.
- Exclusion uses inverse loader augmentation in base coordinates, exact nearest
  GT segment projection and shared Lasagna normal decoding. Required invalid
  normals raise (explicit difference from replay's spherical invalid fallback).
- Independent implementation review prompted a real-file test: input level 1,
  normals level 2, annotation transform and actual nonlinear/rotated loader map.
  Also fixed empty-test scalar keys and state-only checkpoint branch inference.
- Shared eligibility now has its own bounded bank within existing shared slots;
  queue metadata contains no eligibility tensor. Covered pin/transfer/stream
  ownership, batch split, ordinary pool and cooperative broker construction.
- CPU regression results: embedding suite 24 passed (6.83s), including 3-process
  Gloo with zero-work rank; selected legacy Fiber3D tests 34 passed/1 skipped
  (3.08s); gradient accumulation 12 passed/12 CUDA skips (5.53s); cooperative
  integration 6 passed/1 CUDA skip (13.20s); existing vectorized-vs-scalar normal
  sampler 1 passed (1.70s). No GPUs used, no dependencies installed.
- Exact commands (from repo root unless noted):
  `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_fiber_embeddings.py`
  Same prefix + `vesuvius/tests/neural_tracing/test_gradient_accumulation.py`.
  Same prefix + `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'sample_pipeline or cuda_batch or shared_tensor or loss or model or checkpoint or native_coarse or materialize_targets or lasagna_normal_sampler' --maxfail=2`.
  Same prefix + `vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py -k vectorized_lasagna_normals_match_scalar_path -o faulthandler_timeout=30`.
  From `vesuvius/`: `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/home/hendrik/villa/gpu_manager/src /home/hendrik/.venv_las/bin/python -m pytest -q tests/neural_tracing/test_cooperative_integration.py`.
- Sandbox Gloo socket failure resolved by approved outside-sandbox rerun. An
  overly broad 2D normal-test selection was interrupted; a focused normal test
  also stalled in Zarr fixture creation inside sandbox (before changed code),
  then passed in 1.70s outside. Updated the old fake-reduction test for four added
  statistics fields. Fixed the new real-file test's helper import before it passed.
- One-CPU-thread synthetic benchmark: 128 input/stride4, 16 channels, 65,536 pairs,
  3 measured repeats after pair warmup. Exact indexed nearest-segment broad phase
  reduced exclusion median 0.363s -> 0.161s; final min/median/max
  0.1608/0.1611/0.1618s. Pair loss+backward 0.1227/0.1283/0.1306s. Release installed
  NumPy/SciPy/PyTorch wheels, constant normals, no disk I/O in timing.
- Limitations: no real-dataset GPU throughput/learning-quality run, no native
  parity executable built, no embedding Zarr export/Atlas publication, and no
  conditioned decoder support. Those last two are outside this unconditioned
  training feature and fail explicitly where applicable. Model API exposes
  embeddings. Cooperative use requires recalibrating the changed memory footprint.
