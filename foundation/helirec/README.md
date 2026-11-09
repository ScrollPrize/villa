# helirec

Standalone reconstruction of **extended-field-of-view helical tomography** (ESRF BM18 "HA + TA" scan pairs), from
the raw Bliss projection folders to xy slices and volumes. The core is C++17; every compute stage has a CPU
(OpenMP / FFTW) and a CUDA (cuFFT) kernel with identical semantics, exposed to Python through nanobind.

helirec reimplements the Night Rail BM18 + nabu-helical + nabuxx workflow. Two modes (`--set mode=...`, Python
`params={"mode": ...}`):

* `corrected` (default): output in attenuation (cm⁻¹); every measured line counted once in the half-acquisition
  rebalance; ramp padding reflected about the true rotation axis; mending over the merged columns both scans cover;
  a unit-DC Paganin kernel independent of the chunk height. No projection filters by default.
* `legacy_reference`: the reference workflow's conventions (half-attenuation units, its rebalance endpoints,
  integer padding reflection, mending bounds, chunk-dependent Paganin) with every parameter pinned (GHBP, Paganin
  1000 + unsharp 1.2/4, …); overriding one is an error, and only the measured calibration is allowed.

Outputs record the mode, its conventions and every resolved parameter (`manifest.json`, OME-Zarr attributes).

## Slices and volumes (`helirec-slices`)

```bash
helirec-slices PROJ -n 8 --tag unfiltered                                    # 8 slices, default mode
helirec-slices PROJ -n 8 --set mode=legacy_reference --tag reference         # the reference workflow
helirec-slices PROJ --fractions 0.4 0.5 --set "filters=paganin(delta_beta=500)" --tag db500
helirec-slices PROJ -n 4 --set backprojector=fbp --device cpu --tag exact_fbp
```

Range mode reconstructs contiguous slices in chunks (`--chunk`, default 32) and writes an OME-Zarr (NGFF 0.4,
Zarr v2) by default:

```bash
helirec-slices PROJ --z 900 1700 --roi X0 X1 Y0 Y1 --uint8 --levels 4 --tag block       # float32 + uint8 OME-Zarr
helirec-slices PROJ --heights H0 H1 --format tiff --uint8 --tag block_tif                 # TIFF stacks
```

* `--z` is the volume slice index (z = view height − first doable height, z = 0 at the top); `--heights` uses
  view heights; both are [start, end). `--roi X0 X1 Y0 Y1` crops the output: every method reconstructs whole
  slices, so a cropped slice is identical to the same pixels of an uncropped run. The OME-Zarr transforms place the
  crop in the full volume.
* float32 is always written (raw chunks). `--uint8` adds an 8-bit copy with `--u8_window` (default [−0.06, 0.29]
  cm⁻¹; [−0.03, 0.145] in `legacy_reference`), 0 outside the reconstruction disk, compressed losslessly with
  vc-delta3d (`{"id": "vc-delta3d", "quant": 1}`, preset `fast`; `--u8_delta3d_preset delta` writes D3D1 frames
  for older readers). Chunks default to 128³ (keep multiples of 32). All-fill chunks are not written.
* In `legacy_reference`, chunked slices inherit the reference Paganin kernel's chunk-height dependence (a small
  uniform offset compared with single-slice runs); the default mode does not.

Python:

```python
from helirec.recon import HelicalRecon
r = HelicalRecon("/path/to/projections", cache_dir="cache", device="cuda")
img = r.reconstruct_fractions([0.5])[0]                  # N x N float32
r.set_params(filters="paganin(delta_beta=500)")           # rebuilds the reconstructor, keeps the calibration
dbg = r.reconstruct_heights(h, h + 1, debug=True)         # + sinogram, weights, balanced weights, filtered rows
```

## Surface composites (`helirec-composite`)

Renders layer composites (max and mean) along a tifxyz surface, plus `--xy_slices N` slices over its z range,
from a reconstructed OME-Zarr or straight from the projections:

```bash
helirec-composite VOLUME.zarr SURFACE_TIFXYZ --layers 15 --start -4 --direction auto --xy_slices 8 --tag s
helirec-composite --projections PROJ --cache_dir CAL SURFACE_TIFXYZ --z Z0 Z1 --layers 15 --start -4 \
    [--set "filters=paganin(delta_beta=1000)" --sweep paganin.delta_beta=250,500,1000,2000] --tag s
```

Surface geometry and sampling come from vc_reader (bicubic Hermite patches, normal dP/du × dP/dv, trilinear
layers; extra `composite`, with vc_reader from `villa/vc_reader`). Layer k sits at direction × (start + k·step) voxels along the normal; `--direction
auto` steps toward the denser side. Output: `max.tif`, `mean.tif` (float32), uint8 TIFF/PNG copies, `mask.png`,
`composite.json`, `layers/`, `xy_slices/`; `--surface_zarr` also writes the layer stack as a uint8 OME-Zarr.

From the projections, `helirec.band.BandComposite` skips slices and volumes: it runs the exact backprojector only
at the slice pixels the composite's sample points need, samples them on the GPU and returns max/mean. Raw data
streams through pinned memory with read-ahead, and a filter sweep shares everything before the filters:

```python
bc = BandComposite(r, pts, valid)                       # pts (L, H, W, 3) XYZ voxels, valid (H, W)
bc.run(chunk=64); mx, mean = bc.composites()
bc = BandComposite(r, pts, valid, n_sets=4)
bc.run(chunk=40, sweep=[{"filters": f"paganin(delta_beta={v})"} for v in (250, 500, 1000, 2000)])
mx, mean = bc.composites(set=2)
```

## Pipeline

| # | stage | module (C++ `include/helirec/…`) | backends |
|---|---|---|---|
| 0 | Bliss master parsing: entries, angles, z, machine current, frame→file map | `python/helirec/bliss.py` | Python |
| 1 | raw row-window reads (`pread`, one HDF5 chunk per frame) | `raw_reader.hpp` | CPU (threads) |
| 2 | calibration: darks/flats, weight map + apodisation, rotation axis, HA→TA vector | `calib.hpp`, `python/helirec/calibration.py` | CPU |
| 3 | helical geometry: span info, doable span, per-projection shifts and grid bins | `helical.hpp` | CPU |
| 4 | gridding: flat-field, weights, vertical/angular/horizontal splat (gather form) | `accumulate.hpp` | CPU, CUDA |
| 5 | HA/TA luminosity equalisation, divide, mend | `accumulate.hpp` | CPU, CUDA |
| 6 | projection filter chain (default: none) | `filters.hpp`, `fourier.hpp` | CPU, CUDA |
| 7 | −log | `sino.hpp` | CPU, CUDA |
| 8 | half-acquisition rebalance (opposite-ray weights) | `sino.hpp` | CPU |
| 9 | Ram-Lak ramp filter with opposite-ray padding, × balanced weights | `sino.hpp`, `fourier.hpp` | CPU, CUDA |
| 10 | reconstruction: GHBP, exact FBP, SIRT, MLEM or gridrec | `ghbp.hpp`, `kernels.hpp`, `tomo.hpp` | CPU, CUDA |

`SliceReconstructor` (`pipeline.hpp`) chains stages 3–10 for a chunk of view heights. Every stage is also callable
on its own from Python (`helirec._core`, re-exported in `helirec`) on NumPy arrays with `device="cpu" | "cuda"`.

### Reconstruction methods (`--set backprojector=SPEC`)

* `ghbp`: the reference hierarchical backprojector (default).
* `fbp(interp=linear|cubic|lanczos3|bspline3|bspline5, oversample=1..8)`: exact pixel-driven backprojector;
  `oversample` FFT-upsamples the filtered rows before interpolating (default bspline3, 2×).
* Whole-slice methods (`tomo.hpp`; CPU `src/cpu/tomo.cpp`, CUDA `src/cuda/tomo.cu`). SIRT, MLEM and gridrec began
  as ports of tomopy's libtomo implementations (https://github.com/tomography/tomopy):
  * `sirt(num_iter, relax, norm=rowsum|len2, projector=sf|chord, init=constant|gridrec, init_value)`
  * `mlem(num_iter, projector, init, init_value)`
  * `gridrec(filter_name, filter_par=a:b)`, filters `none`, `shepp`, `cosine`, `hann`, `hamming`, `ramlak`,
    `parzen`, `butterworth`.
  * SIRT divides each ray's residual by its row sum (`norm=rowsum`, stable for relax < 2 with either projector);
    `len2` (|a_i|²) is unstable with the `sf` footprint. MLEM ignores rays with b ≤ 0 or Ax ≤ 0.
  * Projectors: `sf` (default), the separable footprint (Long, Fessler & Balter 2010), integrates the chord
    trapezoid over the detector cell; `chord` takes it at the cell centre. Both are exact transpose pairs.
  * SIRT and MLEM use the −log rows over 360° and the rays that have data; gridrec reconstructs the
    balance-weighted rows.
* `python(fn=…)`: a function registered in `helirec.backprojectors`, for experiments.

### Other parameters

* **Projection filters**: a chain configured as data, `filters` (list of dicts), default `[]`. The reference chain
  is `filters.REFERENCE_CHAIN` (Paganin δ/β 1000 + unsharp σ 1.2, coefficient 4). Built-in types: `paganin`,
  `unsharp`, `gaussian`, `median`, `python`; register more in C++ (`register_filter`, `filters.hpp`) or Python
  (`helirec.filters.register`). Each filter runs at `stage=intensity` (before −log, default) or `stage=log` and
  declares the margin rows it needs. Consecutive linear shift-invariant filters of one stage fuse into one FFT
  filter. Command line: `--set "filters=paganin(delta_beta=2000)+median(radius=1,stage=log)"`, or override one
  parameter of the current chain with `--set paganin.delta_beta=500`. Sweeps take the same keys
  (`--sweep paganin.delta_beta=250,500`, or `filters=CHAIN1;CHAIN2`); a sweep of one intensity-stage FFT step shares
  the forward FFT.
* **Gridding**: `luminosity_repartition`, `mend_threshold`, `n_grid`, `angular_tolerance_steps`, `block_angles`
  (memory/speed only).
* **Geometry**: `--cor` / `--vector` override the calibration for one run (never cached; not allowed in
  `legacy_reference`).
* **Calibration cache** (`--cache_dir`): darks/flats reduction and the measured calibration, keyed by a
  fingerprint of the acquisition and the calibration version; written atomically.
* **New kernels**: add a function with the same signature in `src/cpu` and `src/cuda`; shared per-element
  arithmetic lives in `src/common/*_core.hpp` so the backends cannot drift apart.

## Accuracy

* `legacy_reference` follows the reference operation by operation, including its float32/float64 choices. GHBP
  uses explicit FMAs in nvcc's contraction order and CPU code is built with `-ffp-contract=off`, so CPU and CUDA
  GHBP are bit-identical and match nabuxx to 3e-8.
* The rotation-axis merit keeps the reference's filter roles in both modes, so the calibration matches it.

## Method comparison (`evaluation/compare_methods.py`)

Reconstructs the same slices with each method of a JSON config (default `evaluation/default_methods.json`: FBP,
SIRT, MLEM and gridrec on unfiltered data) and writes `slices/<method>/z<Z>.tif|png`, per-slice side-by-side
sheets, a contact sheet and `manifest.json` (time, mean, std, fine-scale std). Methods in a slice share one window.
Options: `--projections`, `--cache_dir`, `-n`, `--z Z0 Z1`, `--roi X0 X1 Y0 Y1` (`-1`: whole slices),
`--methods NAME ...`, `--overwrite`; existing slices with matching provenance are reused.

## Tests

`pytest` runs the unit tests: CPU vs CUDA vs NumPy references for every kernel, projector adjointness, the
whole-slice methods, and regression checks of the corrected conventions (`tests/test_regressions.py`).

## Build

Requirements: CMake ≥ 3.24, a C++17 compiler, FFTW3 (float + double), HDF5 (C), Python ≥ 3.10, and optionally
the CUDA toolkit. CUDA code is compiled for the build machine's GPU (`native`); override with
`-DCMAKE_CUDA_ARCHITECTURES=...`. `HELIREC_CUDA=AUTO` (default) builds CUDA only if `nvcc` is found, so put it on
`PATH` (or pass `-C cmake.define.HELIREC_CUDA=ON`); check with `helirec.devices()`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy nanobind scikit-build-core h5py tifffile pytest
uv pip install --python .venv/bin/python --no-build-isolation -e .
.venv/bin/python -m pytest -q
```

## Limitations

These raise explicit errors:
* exactly one companion scan, on the left of the main scan, with the rotation axis in the main scan's right half;
* both scans moving in the same z direction;
* no per-projection lateral-correction drift;
* parameter sweeps (`reconstruct_band(..., sweep=...)`) run on CUDA only;
* xz slices are not implemented.

The calibration kernels are CPU-only (they are cheap).
