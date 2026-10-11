# Fused winding-pair evaluation (2026-10-11)

Implemented opt-in `pair_backend: "compiled"` in `fiber_uv_layout.py` using
`torch.compile(pair_objective, fullgraph=True, dynamic=False)`. This compiles the
existing shared objective and backward; there is no second implementation of the
loss, silent fallback, hypothesis pruning, mixed precision, or freezing.
The current 4096-constraint minibatches and per-minibatch optimizer updates stay
unchanged. The compiler fuses elementwise work and reductions, avoiding many
intermediate arrays; it does not eliminate every intermediate or emit one single
kernel. Eager remains the default and portable reference.

## Measurements

Exclusive H100 80GB, PyTorch 2.11.0+cu128 / Triton 3.6.0. Real Paris4 CCW tensor
cache `data/uv_anchor_v1/ccw_tensors.npz`, 151 hypotheses, FP32. No native build:
Inductor/Triton JIT. Three warmups then 20 CUDA-event forward/backward measurements.

| Workload | Eager mean / p50 / p95 ms | Fused mean / p50 / p95 ms | Speedup |
|---|---:|---:|---:|
| Reference initialization, B4096 | 41.115 / 41.117 / 41.130 | 17.280 / 17.280 / 17.295 | 2.379× |
| Optimized state, B4096 | 41.128 / 41.127 / 41.141 | 17.239 / 17.238 / 17.248 | 2.386× |
| Uneven batch, B4095 | 41.128 / 41.129 / 41.138 | 18.098 / 18.094 / 18.123 | 2.273× |
| Production minibatch update | 53.345 / 53.354 / 53.391 | 29.258 / 25.580 / 33.864 | 1.823× |

Production measurement uses 25 updates, excluding the first five from timing
statistics. Includes coordinate construction, regularizers, backward and sparse
Adam/width Adam, but excludes initial width-boundary calculation and audits.
It is not a full-run wall-clock speedup. Different 4095/4096 partitions explain
two compilation specializations; the precise cause of alternating update timing
has not been isolated. No full ten-pass fused solve was run.

Pair-evaluation incremental peak allocation: 7,097,962,496 → 1,883,997,184 bytes
(6.61 → 1.75 GiB). Profiler CUDA events per call: 239 → about 12 (includes memory
events, not a pure kernel count). Initial fused forward/backward took 8.851s;
the new tail shape took 4.948s. Production first two updates took 5.300/2.647s.
These compile costs are excluded from steady-state measurements.

## Correctness and limitations

All six input gradients and loss passed comparisons on fresh, optimized and tail
batches. Gradient relative L2 errors were about 1e-7–1e-6. An additional test
covered zero/sign boundaries, half turns, lookup-bin edges, lookup extrapolation,
and the vertical clamp. Existing focused regression suite: 32 tests passed.

Fusion preserves the formula and FP32 dtype, **not bitwise rounding or identical
optimizer trajectories**. After 25 matched updates:

| Metric | Eager | Fused |
|---|---:|---:|
| Mean online objective | 54.5097885 | 54.5098114 |
| Sign violations | 18.7617540% | 18.7624369% |
| Reference winding L1, offset corrected | 0.0218821 | 0.0218899 |
| Within two reference windings | 99.9992156% | 99.9992156% |

Four of 489,663 pieces changed MAP choice. Maximum shared phase difference was
0.3184 turns; maximum selected-bucket U offset difference was 0.1871 turns.
These are locally material despite small aggregate differences. Maximum width
difference was 1.2266 base voxels (width range roughly 2313–82427 voxels).
Thus this backend stays opt-in: these short tests do not establish equivalent
long-run solutions. Independent review agreed with this restriction.

Deviation: optional sign-charge and expected-cost configurations were not GPU
fusion-tested; current production likelihood configuration was. Those optional
paths retain shared code and eager tests, but compiled use is not validated here.
No installation changes. CUDA measurements do not establish macOS/ARM performance.

Historical short-run JSON metadata says `event_samples=0` and uses the full-pass
loss label: these runs stopped before completing a pass. They actually performed
25 minibatch updates. Reporting has since been fixed to count partial-pass events
and label the short-run objective correctly; original output files are preserved.

## Reproduction / use

Use the already-approved runner (each configuration has its own output directory;
benchmark output is protected against accidental overwrite):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_pair_fusion_benchmark.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_pair_eager_updates.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_pair_compiled_updates.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_pair_fusion_boundaries.json
```

Outputs are under `data/uv_pair_{fusion_benchmark,eager_updates,compiled_updates,fusion_boundaries}_v1`.
For a normal run add `"pair_backend": "compiled"` to the solver configuration;
do not include `benchmark_updates` or `benchmark_pair_fusion`. Keep the existing
batch size, full-pass traversal and minibatch update mode. Compilation is lazy at
first use and explicit compilation errors are not hidden.

Tests:

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_pair_backend.py vesuvius/tests/test_fiber_uv_layout.py vesuvius/tests/test_fiber_uv_sign_charge.py vesuvius/tests/test_fiber_uv_full_pass.py vesuvius/tests/test_fiber_uv_global_accumulation.py
```
