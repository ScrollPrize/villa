# Local Development Notes For This Checkout

These notes are for coding agents and local runs in this checkout:

`/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3`

They document machine-specific workflow details that are not general project
specs.

## Python/VC3D Import Layout

Normal training/debug commands import the editable installed `vc` package from:

`/home/hendrik/.local/lib/python3.14/site-packages/vc`

Even if `PYTHONPATH` contains:

`$SRC/volume-cartographer/build/python-bindings/python`

the editable scikit-build import hook can still route `import vc.volume` to the
installed editable package. Therefore, after changing VC3D Python bindings,
update the installed editable package, not only the CMake build tree.

## Updating The Installed VC Package

From repo root:

```bash
python -m pip install -e volume-cartographer --no-deps --break-system-packages
```

Use `--no-deps`. Without it, pip may upgrade `numcodecs` to a version
incompatible with the local `zarr 2.18.7` install.

Known compatible local versions after the update:

```text
zarr==2.18.7
numcodecs==0.15.1
volume-cartographer==3.0.3
```

Do not run the fiber-trace 2D runner with `PYTHONNOUSERSITE=1` in this
checkout. On this machine that bypasses the working user-site package set and
can select a zarr/numcodecs combination where `zarr.storage.Store` is missing.
Use the normal `PYTHONPATH=...` runner commands below instead.

Verify the active import:

```bash
python -c "import vc.volume, zarr, numcodecs; print(vc.volume.__file__); print(zarr.__version__); print(numcodecs.__version__); help(vc.volume.Volume.sample_coords)"
```

For the current blocking coordinate sampler, the help output must include:

```text
sample_coords(..., tile_size: int = 32, blocking: bool = True) -> tuple
```

## Optional Build-Tree Rebuild

For checking the C++ binding build before reinstalling:

```bash
cmake --build volume-cartographer/build/python-bindings --target vc_volume -j 8
```

This updates:

`volume-cartographer/build/python-bindings/python/vc/volume.cpython-314-x86_64-linux-gnu.so`

However, normal Python may still import the editable installed module from
`~/.local`. Use the editable reinstall above before relying on normal runner
commands.

## Fiber Trace 2D Runner Command

Use this command to inspect the augment contact sheet for the current example
config:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.runner $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --export-dir ./ --sample-index 1 --augment-vis
```

The runner prints startup timings, per-augmentation timings with total and
average-per-patch summaries, and volume sampler stats. For the blocking sampler
fix, the first `unaugmented` patch should visibly match `noise_min` and
`blur_min`.

## Fiber Trace 2D Training Command

Use this command to run the V0 strip-direction trainer with the current example
config:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json
```

The trainer writes TensorBoard events and snapshots under the configured
`training.run_path`. Do not add `PYTHONNOUSERSITE=1` to this command in this
checkout.

## Agent Benchmark Command Reuse

When comparing training performance variants as an agent, avoid changing the
shell command shape repeatedly. Use one temporary config path and overwrite that
JSON between runs, so the same approved command can be reused:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train /tmp/fiber_trace_p_d2_w1.json --benchmark --benchmark-batches 30
```

For example, write the next variant into the same temp file:

```bash
jq '.training.tensorboard_enabled=false | .training.pipeline_depth=4 | .training.pipeline_workers=2' $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json > /tmp/fiber_trace_p_d2_w1.json
```

Then rerun the exact same benchmark command above. Do not introduce a new temp
config filename or extra CLI flags unless the user explicitly approves it.

For the S1A 3D fiber loader benchmark, reuse this exact command:

```bash
PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all.json --benchmark --load-only --benchmark-batches 10
```

This is the approved load-only comparison command for the current 3D loader
work. Do not change the config path, flags, or PYTHONPATH shape unless the user
explicitly asks for a different measurement.

Latest 3D DataLoader process-worker validation with that exact command and the
checked-in S1A NML config:

- `training.loader_workers=8`, `training.loader_prefetch_factor=2`,
  `training.loader_worker_device=cpu`, multiprocessing context `forkserver`.
- Previous single-batch VC3D coordinate-sampler baseline after warmup:
  median `1359.06 ms`, post-warmup mean `1359.79 ms`.
- 2026-07-15 forkserver run: all-batch mean/median/min/max
  `1989.75 / 169.02 / 124.12 / 17115.35 ms`; post-first-batch
  mean/median/min/max `309.13 / 163.80 / 124.12 / 1131.87 ms`.
- The first row includes DataLoader worker startup plus worker-local loader
  construction. `load_ms` includes CPU batch wait and main-process batch
  transfer; `to_device_ms` reports the transfer portion.
- A follow-up run with benchmark `cpu_ms/cpu_x` columns showed post-startup
  weighted `cpu_x=2.44`, median row `cpu_x=3.02`, and mean row `cpu_x=3.02`;
  this is not full CPU utilization on a 32-logical-CPU machine.
- A diagnostic rerun with worker-stage timings showed the blockage explicitly:
  the first eight returned rows each carried a worker-local
  `FiberTrace3DLoader` construction cost of roughly `4.6-5.1 s`, because every
  DataLoader worker parses/opens the full dataset on its first item. After
  worker construction, rows 9-10 showed worker patch builds of roughly
  `2.7-2.8 s` wall and `3.38 s` CPU. The dominant worker stages were target
  generation (`0.96-0.99 s`), VC3D volume sampling (`0.63-0.65 s`), value
  augmentation (`0.34-0.38 s`), geometry-map creation (`0.28-0.38 s`), and
  valid-mask generation (`0.17-0.19 s`). Main-process transfer to CUDA was
  still visible at about `100-110 ms` on the steady rows.
- The worker-overlap diagnostic showed `avg_active=6.94`, `max_active=8`, and
  `worker_cpu_x=5.11` for ten produced items, so DataLoader workers are
  overlapping, but only up to the configured eight workers and with about five
  effective CPU cores over the measured window. The target-generation
  sub-breakdown showed that steady-state `target_ms` is dominated by dense
  direction target encoding/tensor materialization: rows 9-10 had
  `target_ms=899.52/932.30 ms`, `raster_ms=19.18/19.57 ms`, and
  `encode_ms=852.53/887.88 ms`.

## Fiber Trace 2D Training Prefetch Commands

Prefetch the base-volume chunks needed for the first 10 configured training
steps:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --prefetch --prefetch-steps 10
```

Prefetch every configured training control point once in deterministic
pseudo-random order, plus every configured held-out `test_datasets` control
point once when present. The explicit `--prefetch-steps 0` overrides
`training.max_steps`:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --prefetch --prefetch-steps 0
```

Training prefetch exits before model/TensorBoard/snapshot setup. It still uses
the same deterministic control-point sample mode and the same CP-local
source-strip path as training. For each CP and strip-z offset it covers the
configured augmentation envelope through the sampler-level VC3D blocking
coordinate/cache path, so it is intentionally independent of one particular
random augmentation draw. Lasagna manifest channels are not prefetched.

Set `training.max_sample_index` to a positive exclusive sample-index count to
run many training steps over a bounded deterministic prefix. The default `0`
means no limit. Prefetch progress prints `idx=<exclusive-index>` for the
largest contiguous deterministic shuffled training-stream prefix whose required
chunks are already in the cache or represented by missing markers. This is not
the original flat fiber/CP id; it counts through the same seeded random order
that training uses. After stopping a long prefetch, use that `idx` value as
`training.max_sample_index` to train only on the cache-complete random-prefix
stream.

The bounded prefix controls which CP/data samples are loaded. Training
augmentations are still keyed by the unbounded deterministic training stream
index, so repeated passes over a bounded prefix keep producing fresh
deterministic geometric and value/image augmentation draws.

Use `prefetch_sampler_workers` to limit CPU-side dependency/source generation
without reducing download concurrency. `prefetch_workers` controls chunk
download workers; `prefetch_sampler_workers` controls the producer threads that
build CP-local source strips and collect chunk dependencies. Prefetch
temporarily forces PyTorch CPU intra-op threads to `1` and restores the previous
value afterward, so each producer does not fan out over the full machine.

The old `strip_coord_cache_dir` CP-local dense source-coordinate cache has been
removed. The loader now builds a compact in-RAM fiber-line geometry store at
startup. The populated geometry is shared by cloned/threaded loaders inside the
process. This is separate from the remote Zarr chunk cache under
`volume_cache_dir`, which remains the only persistent cache used for volume
chunks.

Startup compact-geometry construction uses `loader_workers` for record-level
parallelism. Use `loader_workers: 0` to resolve to all logical CPU cores on the
machine, `loader_workers: 1` for the serial/debug path, and values above one to
build records concurrently while storing the final compact geometry in original
record order.

The current startup-geometry benchmark command is:

```bash
PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --benchmark --load-only --profile
```

Baseline before CP-window filtering/vectorized normals/parallel startup:
startup compact geometry build was `8m53s` for `464` records,
`14773/184` valid/skipped CPs, and `63.5 MiB`.

After CP-window filtering, batched Lasagna normal sampling, and
`loader_workers` startup parallelism on the same command/config: startup was
`2m12s` for `464` records, `14773/184` valid/skipped CPs, and `40.0 MiB`;
load-only/profile throughput was `119.96 patches/s` over `12800` patches.

## Focused Test Command

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

These tests use local/fake arrays and should not require network access.
