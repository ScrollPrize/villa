# Task Log

## Vectorized Strip And Line Coordinate Generation

- Added a torch-backed dense side-strip grid builder in `strip_geometry.py`. It keeps the existing Python/Numpy VC3D/Lasagna frame transport and roll smoothing, then vectorizes the per-pixel arc-coordinate, cubic Hermite, and normal interpolation work.
- Added CP-local source geometry reuse for augment visualization. The runner now builds descriptor/window/Lasagna normals/source strip coordinates once for the selected sample and reuses that source grid for all contact-sheet cells.
- Replaced affine line-coordinate generation with a direct vectorized inverse mapping. Smooth-offset cases use a chunked tensor nearest search over the geometric source-coordinate grid instead of a Python loop over every line sample.
- Kept VC3D blocking volume sampling and coordinate-space geometric augmentation unchanged.
- Added tests comparing the torch strip grid against the existing NumPy path and checking vectorized shifted line coordinates.

Validation:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: `26 passed in 1.96s`.
- Augment-vis command:
  - `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.runner /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --export-dir /tmp/fiber_trace_aug_debug_blocking --sample-index 1 --augment-vis`
  - Result: command completed and wrote `augment_contact_sheet.jpg` plus `augment_summary.txt`.
  - Current timing sample: first `unaugmented` row total `2486.4 ms`, dominated by `volume_sample=2204.9 ms`; one-time `strip_coords=103.0 ms`; affine repeated cells typically `3.6..4.8 ms`; smooth/combined line-coordinate rows mostly `1.8..2.4 ms` after the vectorized fallback.

Environment note:

- The same command with `PYTHONNOUSERSITE=1` failed before reaching the loader because that environment resolved a zarr install where `zarr.storage.Store` is missing. The normal local command above matches the currently working checkout environment.
