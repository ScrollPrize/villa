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
helirec-slices PROJ --fractions 0.4 0.5 --set "filters=paganin(delta_beta=500)+unsharp(sigma=1.2,coeff=4)" --tag db500
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
r.set_params(filters="paganin(delta_beta=500)+unsharp(sigma=1.2,coeff=4)")           # rebuilds the reconstructor, keeps the calibration
dbg = r.reconstruct_heights(h, h + 1, debug=True)         # + sinogram, weights, balanced weights, filtered rows
```

## Surface composites (`helirec-composite`)

Renders layer composites (max and mean) along a tifxyz surface, plus `--xy_slices N` slices over its z range,
from a reconstructed OME-Zarr or straight from the projections:

```bash
helirec-composite VOLUME.zarr SURFACE_TIFXYZ --layers 15 --start -4 --direction auto --xy_slices 8 --tag s
helirec-composite --projections PROJ --cache_dir CAL SURFACE_TIFXYZ --z Z0 Z1 --layers 15 --start -4 \
    [--set "filters=paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)" --sweep paganin.delta_beta=250,500,1000,2000] --tag s
```

Surface geometry and sampling come from vc_reader (bicubic Hermite patches, normal dP/du × dP/dv, trilinear
layers; vc_reader is a required dependency from `villa/vc_reader`). Layer k sits at direction × (start + k·step) voxels along the normal; `--direction
auto` steps toward the denser side. Output: `max.tif`, `mean.tif` (float32), uint8 TIFF/PNG copies, `mask.png`,
`composite.json`, `layers/`, `xy_slices/`; `--surface_zarr` also writes the layer stack as a uint8 OME-Zarr.

From the projections, `helirec.band.BandComposite` skips slices and volumes: it runs the exact backprojector only
at the slice pixels the composite's sample points need, samples them on the GPU and returns max/mean. Raw data
streams through pinned memory with read-ahead, and a filter sweep shares everything before the filters:

```python
bc = BandComposite(r, pts, valid)                       # pts (L, H, W, 3) XYZ voxels, valid (H, W)
bc.run(chunk=64); mx, mean = bc.composites()
bc = BandComposite(r, pts, valid, n_sets=4)
bc.run(chunk=40, sweep=[{"filters": f"paganin(delta_beta={v})+unsharp(sigma=1.2,coeff=4)"} for v in (250, 500, 1000, 2000)])
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
  * `sirt(num_iter, relax, norm=rowsum, weighting=mask|confidence, projector=sf|chord, init=constant|gridrec, init_value)`
  * `mlem(num_iter, projector, init, init_value)`
  * `gridrec(filter_name, filter_par=a:b)`, filters `none`, `shepp`, `cosine`, `hann`, `hamming`, `ramlak`,
    `parzen`, `butterworth`.
  * SIRT divides each ray's residual by its row sum (`norm=rowsum`, finite `0 < relax < 2`);
    `len2` is rejected because it lacks this stability guarantee. `weighting=mask` (default) selects valid rays;
    `confidence` uses nonnegative measured weights in both the residual and column normalization. Accumulated
    support is an exposure/interpolation proxy, not a calibrated inverse noise variance.
    MLEM ignores rays with b ≤ 0 or Ax ≤ 0.
  * Projectors: `sf` (default), the separable footprint (Long, Fessler & Balter 2010), integrates the chord
    trapezoid over the detector cell; `chord` takes it at the cell centre. Both are exact transpose pairs.
  * SIRT and MLEM grid only measured detector pixels: no edge padding or invalid-pixel values enter their sums.
    `mend_threshold` removes weak support relative to each row's mean positive weight. Filter context can fill
    unsupported values, but their weights stay zero. Missing angles remain excluded. `init=gridrec` fills and
    balances a separate copy; gridrec as a standalone method still uses balance-weighted rows.
  * Without intensity filters or two-scan luminosity correction, iterative methods take −log before gridding.
    Paganin and luminosity correction retain intensity-domain processing, then −log. These are **rebinned 2D
    models**, not an exact raw-measurement forward model: fractional z/angle/x interpolation and phase retrieval
    remain approximations. The paired projectors are adjoints of the rebinned slice model. Iterative luminosity
    correction reads a full vertical context window so its central values do not depend on slice chunk height.
* `python(fn=…)`: a function registered in `helirec.backprojectors`, for experiments.

### Other parameters

* **Projection filters**: a chain configured as data, `filters` (list of dicts), default `[]`. The reference chain
  is `filters.REFERENCE_CHAIN` (Paganin δ/β 1000 + unsharp σ 1.2, coefficient 4), also used by the standard
  Paganin baseline recipes. Explicit chains are literal: omit `unsharp` for Paganin alone, or set
  `unsharp.coeff=0` to disable sharpening. Built-in types: `paganin`,
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
SIRT, MLEM and gridrec on unfiltered data) and writes float32 `slices/<method>/z<Z>.tif` and `manifest.json`
(time, mean, std, fine-scale std). PNG slices and contact sheets are opt-in with `--previews`, also for
`helirec-slices`. Preview windows are shared in attenuation units; legacy TIFFs retain native reference units.
Options: `--projections`, `--cache_dir`, `-n`, `--z Z0 Z1`, `--roi X0 X1 Y0 Y1` (`-1`: whole slices),
`--methods NAME ...`, `--overwrite`; existing slices with matching provenance are reused.

`evaluation/343p_baselines.json` compares FBP and SIRT (20 iterations from gridrec), each with no projection
filters and with Paganin δ/β 1000 + unsharp σ 1.2, coefficient 4, over the same five full slices of the local PHerc0343P projections.
It uses the corrected mode and CUDA backend, requiring a CUDA-enabled build and host.
The calibration cache lives under the output directory. Run from `foundation/helirec`:

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python -u evaluation/compare_methods.py evaluation/343p_baselines.json \
    --out /mnt/raid_nvme/343p_recon_work/output/baselines
```

`evaluation/343p_corrected_baselines.json` uses `_v2` folders for unfiltered methods and new
`_paganin_unsharp_v3` folders for the sharpened Paganin methods, preserving the earlier unsharpened slices. It adds
`legacy_reference` (pinned Night Rail/nabuxx GHBP, Paganin 1000 + unsharp 1.2/4, half-attenuation units).
Legacy methods ignore shared corrected-mode overrides and enforce their pinned settings. To retain the earlier
metadata as well as images, use a new `--run-name`. To run only the sharpened Paganin methods:

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python -u evaluation/compare_methods.py evaluation/343p_corrected_baselines.json \
    --out /mnt/raid_nvme/343p_recon_work/output/baselines --run-name paganin_unsharp_v3 \
    --methods fbp_paganin_unsharp_v3 sirt_paganin_unsharp_v3
```

Measured-support and interpolation regressions are in `tests/test_sirt_pipeline.py`: explicit-matrix SIRT
updates, excluded-ray invariance, a fractional helical analytic phantom, chunk consistency and CPU/CUDA parity.
`tests/test_evaluation.py` checks pinned legacy settings and TIFF-only defaults.

To rebuild the local package with CUDA required (adjust the compiler path for your host):

```bash
AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 uv pip install --python .venv/bin/python --no-deps --reinstall -e . \
    -C cmake.define.HELIREC_CUDA=ON -C cmake.define.CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

Run GPU commands on a host/session with NVIDIA device access. Use `--overwrite` to recompute every slice after
switching devices; `run_cuda.log` can be captured with `2>&1 | tee /mnt/raid_nvme/343p_recon_work/output/baselines/run_cuda.log`.

## Tests

`pytest` runs the unit tests: CPU vs CUDA vs NumPy references for every kernel, projector adjointness, the
whole-slice methods, and regression checks of the corrected conventions (`tests/test_regressions.py`).

## Build

Requirements: CMake ≥ 3.24, a C++17 compiler, FFTW3 (float + double), HDF5 (C), Python ≥ 3.10, and optionally
the CUDA toolkit. CUDA code is compiled for the build machine's GPU (`native`); override with
`-DCMAKE_CUDA_ARCHITECTURES=...`. `HELIREC_CUDA=AUTO` (default) builds CUDA only if `nvcc` is found, so put it on
`PATH` (or pass `-C cmake.define.HELIREC_CUDA=ON`); check with `helirec.devices()`.

Run from `foundation/helirec` in a Villa checkout containing `vc_reader` at `../../vc_reader`.
The local dependency source is configured in `pyproject.toml`; `uv sync` includes composite support.
To also install the test dependencies:

```bash
uv sync --python 3.12 --extra test
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
