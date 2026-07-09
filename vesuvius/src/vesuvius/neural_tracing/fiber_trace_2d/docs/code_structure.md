# 2D Fiber Trace Initial Loader Code Structure

- `fiber_json.py` re-exports the existing VC3D fiber JSON parser so the 2D loader keeps the same JSON semantics as the current fiber-trace code.
- `strip_geometry.py` ports the VC3D/Lasagna `LineViewBuilder` side-strip frame construction and builds explicit side-strip coordinate grids. The dense per-pixel strip interpolation also has a torch-vectorized path that preserves the same frame semantics.
- `augmentation.py` implements deterministic strip augmentations using torch tensor operations for coordinate mapping, line-coordinate mapping, and post-Zarr image/value changes.
- `loader.py` parses Vesuvius-style JSON configs, opens base-volume Zarr arrays through the existing Vesuvius cache-aware opener, samples explicit coordinates, computes prefetch chunk requests, and reuses CP-local source strip geometry across augment-visualization cells.
- `runner.py` provides a small command-line loader/prefetch tester and augment contact-sheet exporter.
- `configs/loader_example.json` shows the expected Vesuvius-style JSON shape.

Run a loader smoke test with:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0
```

Run prefetch for the addressed samples with:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0 --prefetch --prefetch-samples 8
```

Export an augmentation contact sheet with:

```bash
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_2d.runner config.json --sample-index 0 --augment-vis --export-dir /tmp/fiber_trace_2d_aug
```
