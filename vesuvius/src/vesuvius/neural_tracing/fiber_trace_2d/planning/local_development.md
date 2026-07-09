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

The runner prints startup timings, per-augmentation timings, volume sampler
stats, and raw image stats. For the blocking sampler fix, the first
`unaugmented` row must have nonzero valid pixels and should match `noise_min`
and `blur_min`.

## Fiber Trace 2D Training Command

Use this command to run the V0 strip-direction trainer with the current example
config:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json
```

The trainer writes TensorBoard events and snapshots under the configured
`training.run_path`. Do not add `PYTHONNOUSERSITE=1` to this command in this
checkout.

## Fiber Trace 2D Training Prefetch Commands

Prefetch the base-volume chunks needed for the first 10 configured training
steps:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --prefetch --prefetch-steps 10
```

Prefetch all configured training steps:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --prefetch --prefetch-steps 0
```

Training prefetch exits before model/TensorBoard/snapshot setup. It still uses
the same deterministic control-point sample sequence and final augmented
base-volume coordinates as training. Lasagna manifest channels are not
prefetched.

## Focused Test Command

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

These tests use local/fake arrays and should not require network access.
