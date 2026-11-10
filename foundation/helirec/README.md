# helirec

Standalone reconstruction of **extended-field-of-view helical tomography** (ESRF BM18 "HA + TA" scan pairs), from
the raw Bliss projection folders to xy slices and volumes. The core is C++17; every compute stage has a CPU
(OpenMP / FFTW) and a CUDA (cuFFT) kernel with identical semantics, exposed to Python through nanobind.

helirec reimplements the Night Rail BM18 + nabu-helical + nabuxx workflow. Two modes (`--set mode=...`, Python
`params={"mode": ...}`):

* `corrected` (default): output in attenuation (cm⁻¹); every measured line counted once in the half-acquisition
  rebalance; ramp padding reflected about the true rotation axis; mending over the merged columns both scans cover;
  a unit-DC Paganin kernel and a two-scan luminosity window independent of the chunk height; a scan at a
  fractional column offset shifted exactly (`x_interp=sinc`), a slice's fractional detector row interpolated with
  a Lanczos-3 kernel (`z_interp`); each projection gridded whole into one angular bin and backprojected at its measured angle
  (`angle_interp=nearest`; GHBP keeps the two-bin splat); the exposure arc of a continuous scan deconvolved
  (`exposure_arc=auto`); the helical pitch measured from the frames (`z_translation_scale=auto`); the companion
  scan's frames placed as they register on the main scan's, row offset and roll per projection
  (`companion_registration=auto`); recordings of a ray combined with band-dependent weights where the detector's
  sharpness depends on the place (`detector_smear=auto`). No projection filters by default.
* `legacy_reference`: the reference workflow's conventions (half-attenuation units, its rebalance endpoints,
  integer padding reflection, mending bounds, chunk-dependent Paganin and luminosity window, two-column splat of
  fractional column offsets, linear rows, two-bin angular splat, no exposure-arc correction) with every parameter pinned (GHBP, Paganin 1000 + unsharp 1.2/4, …); overriding one
  is an error, and only the measured calibration is allowed.

Outputs record the mode, its conventions and every resolved parameter (`manifest.json`, OME-Zarr attributes).

## Slices and volumes (`helirec-slices`)

```bash
helirec-slices PROJ -n 8 --tag unfiltered                                    # 8 slices, default mode
helirec-slices PROJ -n 8 --set mode=legacy_reference --tag reference         # the reference workflow
helirec-slices PROJ --fractions 0.4 0.5 --set "filters=paganin(delta_beta=500)+unsharp(sigma=1.2,coeff=4)" --tag db500
helirec-slices PROJ -n 4 --set backprojector=fbp --device cpu --tag exact_fbp
helirec-slices PROJ -n 8 --set backprojector=fbp --set "filters=mtf_paganin(delta_beta=1000,sigma=1.06)" \
    --set sld=fraction=0.6,length=400 --tag mtf        # fused phase retrieval + detector deblur, calibrated glare
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
* Each chunk's raw rows are read while the previous chunk computes, reusing the rows the two share.
* float32 is written by default (raw chunks). `--uint8` adds an 8-bit copy and `--uint8_only` writes that copy
  alone (no float32 output), with `--u8_window` (default [−0.06, 0.29]
  cm⁻¹; [−0.03, 0.145] in `legacy_reference`), 0 outside the reconstruction disk, compressed losslessly with
  vc-delta3d (`{"id": "vc-delta3d", "quant": 1}`, preset `fast`; `--u8_delta3d_preset delta` writes D3D1 frames
  for older readers). Chunks default to 128³ (keep multiples of 32). All-fill chunks are not written.
* The uint8 slices are converted on the device (`to_uint8`, `SliceReconstructor.reconstruct(uint8=...)`): a
  `--uint8_only` run downloads one byte per voxel and keeps no float32 volume on the host.
* `--reference_volume ZARR` reuses the mask of a masked uint8 volume on the same grid (e.g. the published volume of
  the scan): the uint8 output is 0 wherever the reference is 0; the float32 output is not masked. The reference's
  level 0 must have the reconstruction's slices (y, x); its number of slices may differ by up to 2 %, as it does
  when the reference was made with another helical pitch calibration (`z_translation_scale`): its slices are then
  spread evenly over the reconstruction's, which places the mask to within a few slices (PHerc0343P, published
  volume of 5398 slices on the 5411 of the measured pitch: within 0.1 slice mid-volume, about 2 at the two ends).
  `--reference_scale LEVEL` takes the mask from a coarser
  pyramid level instead (each of its voxels decides the block of output voxels it covers), which is a far smaller
  read: an 8x level is 1/512 of the data. The reference is read in C++ with helirec's own Zarr reader
  (`helirec::ReferenceMask` on `zarr.hpp`: a Zarr v2 array or OME-Zarr pyramid at a local path, or at an `http(s)://`
  / `s3://` location (a public bucket, no credentials) with `--reference_cache DIR` as its disk cache: each chunk is
  kept under `DIR/helirec_zarr/` once read). A pyramid level that declares no translation is taken as block-reduced.
  Under a mask the exact backprojector computes only its 32 x 32 pixel tiles that hold a kept pixel.
* The volume is written in C++ (`helirec::OmeZarrWriter`): the reconstructor hands it each slice as the block it
  computed, so no whole slice is built on the host and only the chunks that hold data are buffered, encoded and stored.
* Only the pixels that are needed are backprojected, and their values are those of a whole-slice run (the exact
  backprojector computes each pixel on its own): `--bbox X0 X1 Y0 Y1` reconstructs that box of each slice and leaves
  the rest of the output 0 (the output keeps its shape; `--roi` crops it), and a `--uint8_only` run with a
  `--reference_volume` computes only the bounds of what the mask keeps in each chunk; chunks it empties are skipped.
* Progress is a bar per multi-part step (calibration reductions and searches, slices, pyramid levels) with the step
  in hand after the rate; when the output is not a terminal it is one line every 10 s (`helirec.ProgressBar`, the
  C++ `helirec::ProgressBar`).
* In `legacy_reference`, chunked slices inherit the reference Paganin kernel's chunk-height dependence (a small
  uniform offset compared with single-slice runs) and the luminosity window clipped to the chunk; the default mode
  has neither (it reads the window's 12 context rows either side of a chunk).

Python:

```python
from helirec.recon import HelicalRecon
r = HelicalRecon("/path/to/projections", cache_dir="cache", device="cuda")
img = r.reconstruct_fractions([0.5])[0]                  # N x N float32
r.set_params(filters="paganin(delta_beta=500)+unsharp(sigma=1.2,coeff=4)")           # rebuilds the reconstructor, keeps the calibration
dbg = r.reconstruct_heights(h, h + 1, debug=True)         # + sinogram, weights, balanced weights, filtered rows
vol = r.reconstruct_heights(h, h + 32, next=(h + 32, h + 64))   # reads the next chunk's raw rows meanwhile (CUDA)
```

Page-locked host buffers (the chunk's raw windows, several GB) are kept when a reconstructor is replaced, e.g. by
`set_params`, so the next one does not lock them again (~0.2 s/GB); `helirec.release_pinned_memory()` frees them.

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
streams through pinned memory with read-ahead (`--no_prefetch` turns the read-ahead off: it holds a second chunk's
raw windows, tens of GB where a z range overlaps a stationary turn and the filters need many context rows), and a
filter sweep shares everything before the filters:

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
| 4 | gridding: flat-field, weights, vertical / horizontal kernels, angular bins (gather form) | `accumulate.hpp` | CPU, CUDA |
| 5 | HA/TA luminosity equalisation, divide, mend | `accumulate.hpp` | CPU, CUDA |
| 6 | projection filter chain (default: none) | `filters.hpp`, `fourier.hpp` | CPU, CUDA |
| 7 | −log | `sino.hpp` | CPU, CUDA |
| 8 | half-acquisition rebalance (opposite-ray weights); exposure-arc deconvolution along the angle | `sino.hpp` | CPU, CUDA |
| 9 | Ram-Lak ramp filter with opposite-ray padding, × balanced weights | `sino.hpp`, `fourier.hpp` | CPU, CUDA |
| 10 | reconstruction: GHBP, exact FBP, SIRT, MLEM or gridrec | `ghbp.hpp`, `kernels.hpp`, `tomo.hpp` | CPU, CUDA |

`SliceReconstructor` (`pipeline.hpp`) chains stages 3–10 for a chunk of view heights. Every stage is also callable
on its own from Python (`helirec._core`, re-exported in `helirec`) on NumPy arrays with `device="cpu" | "cuda"`.

### Reconstruction methods (`--set backprojector=SPEC`)

* `ghbp`: the reference hierarchical backprojector (default).
* `fbp(interp=linear|cubic|lanczos3|bspline3|bspline5, oversample=1..8)`: exact pixel-driven backprojector;
  `oversample` FFT-upsamples the filtered rows before interpolating (default bspline3, 2×).
* `ghbp` and `fbp` take the ramp filter as `filter_name` (default `ramlak`) and `filter_par=p0:p1` (defaults 0.5, 8),
  e.g. `fbp(filter_name=hann,filter_par=1)`: the windows of gridrec (`sino.hpp` `ramp_windowed`) on the band-limited
  Ram-Lak ramp. `p0` is the cutoff in cycles per pixel (default 0.5, Nyquist) and `p1` the Butterworth order;
  `hann`, `hamming` and `parzen` are zero beyond `p0`. These differ from tomopy ≥ 1.1 for `hann` and `hamming`
  (there the window's minimum is at `p0 / 2`, half Nyquist by default, and it returns to 1 at `p0`) and for cutoffs
  below Nyquist (tomopy's windows are not cut beyond `p0`); gridrec uses the same windows.
* Whole-slice methods (`tomo.hpp`; CPU `src/cpu/tomo.cpp`, CUDA `src/cuda/tomo.cu`). SIRT, MLEM and gridrec began
  as ports of tomopy's libtomo implementations (https://github.com/tomopy/tomopy):
  * `sirt(num_iter, relax, norm=rowsum, weighting=mask|confidence, projector=sf|chord, init=constant|gridrec, init_value)`
  * `mlem(num_iter, projector, init, init_value)`
  * `gridrec(filter_name, filter_par=a:b)`, filters `none`, `shepp`, `cosine`, `hann`, `hamming`, `ramlak`,
    `parzen`, `butterworth`.
  * SIRT divides each ray's residual by its row sum (`norm=rowsum`, finite `0 < relax < 2`);
    `len2` is rejected because it lacks this stability guarantee. `weighting=mask` (default) selects valid rays;
    `confidence` uses nonnegative measured weights in both the residual and column normalization. Accumulated
    support is an exposure/interpolation proxy, not a calibrated inverse noise variance.
    MLEM takes rays with b ≤ 0 as zero measurements (they stay in the normalisation A^T 1, so negative noise is
    clipped rather than excluded); rays with Ax ≤ 0 give no update.
  * gridrec is quantitative: its de-apodisation is the transform of the gridding kernel actually used, and its
    scale is that of the exact FBP (peak values of small features within 0.1 % at any radius,
    `tests/test_tomo.py`). The closed form inherited from tomopy overestimated them by 12 % at the centre, rising
    to 44 % at 0.59 of the padded half width, and `init=gridrec` carried that into SIRT / MLEM.
  * Projectors: `sf` (default), the separable footprint (Long, Fessler & Balter 2010), integrates the chord
    trapezoid over the detector cell; `chord` takes it at the cell centre. Both are exact transpose pairs.
    `sf` evaluates the footprint's cumulative integral at the cell boundaries, which neighbouring cells share: the
    forward projector walks one boundary per thread (two pixels per step, positions in 2⁻³² pixel fixed point) and a
    cell is the difference of its two boundaries; the backprojector reaches a pixel's four cells through three
    boundaries and skips pixels that project off the detector (`src/common/tomo_core.hpp`).
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
  `epr`, `mtf_paganin`, `unsharp`, `gaussian`, `median`, `python`; register more in C++ (`register_filter`, `filters.hpp`) or Python
  (`helirec.filters.register`). Each filter runs at `stage=intensity` (before −log, default) or `stage=log` and
  declares the margin rows it needs. Consecutive linear shift-invariant filters of one stage fuse into one FFT
  filter. Command line: `--set "filters=paganin(delta_beta=2000)+median(radius=1,stage=log)"`, or override one
  parameter of the current chain with `--set paganin.delta_beta=500`. Sweeps take the same keys
  (`--sweep paganin.delta_beta=250,500`, or `filters=CHAIN1;CHAIN2`); a sweep of one intensity-stage FFT step shares
  the forward FFT.
* **Fused MTF / Paganin filter** (`mtf_paganin(delta_beta=…, sigma=… | mtf=…, target=0.5, gain=12 | nsr=…)`, in
  place of `paganin + unsharp`): phase retrieval and detector deblurring as one filter. The detector records the
  propagated intensity blurred by its point spread, `Î_det = (M / P) Î` (`P`: the Paganin response, `M`: the
  detector MTF, a Gaussian of `sigma` pixels or a measured radial table `mtf=ν:M/ν:M/…`, ν in cycles per pixel). The
  filter is `P · D`, with one of two deblur gains `D`:
  * **Gaussian target** (default): `D = T · √(1 + ε²) / √(M² + ε²)`, `T` the MTF of a Gaussian of `target` pixels,
    `ε = 1 / gain`. Where `M ≫ ε` the detector blur is replaced by the target blur; the gain never exceeds about
    `gain`. The system response `M · D` falls monotonically, so edges do not ring and the noise spectrum has no
    hump. `target` is the sharpness / noise trade (smaller: sharper, noisier; `0`: the gain cap alone), `gain` limits
    the noise where the detector passes almost nothing.
  * **Wiener** (`nsr=…`, excludes `target` / `gain`): `D = (1 + nsr) M / (M² + nsr)`, the minimum-error estimate for
    white detector noise and an object spectrum falling as `P²`: the exact inverse where `M² ≫ nsr`, at most
    `(1 + nsr) / (2 √nsr)` (5 for 0.01, 9 for 0.003) at `M = √nsr`, falling with `M` beyond. That peak is a band-pass
    boost (0.36 cycles per pixel for σ 1.06, nsr 0.003): a slab rings by ±7 % at its edges even with the exact MTF,
    and the noise is band-limited (a grain of about 3 px period, 1.6 × the low-frequency noise amplitude on 343P).
  Like `paganin` the kernel is fixed and has unit DC, so the result is independent of the chunk height; its margin
  is Paganin's plus the deblur kernel's, truncated where it falls below 10⁻⁴ of its peak, at most 64 rows (σ 1.06:
  44 rows for the default target form, 29 for nsr 0.01, 40 for nsr 0.003; the response is within about 3 % of the
  formula along the rows near Nyquist). It is one Fourier step, so sweeps share the forward FFT
  (`--sweep mtf_paganin.target=0.4,0.5,0.6`).
  On the PHerc0343P pair (slice z 2698, calibrated SLD, v5 gridding) the Teflon wrap's edge has, as overshoot / dip
  beside the edge at a radial and a tangential site, and noise in the outer air:

  | chain | radial | tangential | noise (cm⁻¹) |
  |---|---|---|---|
  | `paganin(1000)` | 1 % / 4 % | 2 % / 3 % | 0.0042 |
  | `paganin(1000) + unsharp(1.2, 4)` | 26 % / 45 % | 24 % / 31 % | 0.0160 |
  | `mtf_paganin(1000, sigma=1.06, nsr=0.003)` | 10 % / 24 % | 8 % / 9 % | 0.0173 |
  | `mtf_paganin(1000, sigma=1.06)` (target 0.5, gain 12) | 4 % / 15 % | 4 % / 4 % | 0.0129 |
  | `mtf_paganin(1250, sigma=1.06)` | 2 % / 9 % | 4 % / 2 % | 0.0104 |

  `delta_beta` matters more once the blur is removed: a blur alone cannot undershoot, yet Paganin alone leaves a
  4 % dip outside the wrap at δ/β 1000, 1 % at 1250 and none at 1500 (the residual phase fringe, which the
  deblurring amplifies). Recipes: `evaluation/343p_v5_resolution.json`.
* **Eikonal phase retrieval** (`epr(delta_beta=…, iters=3, relax=0.75, max_shift=12.5)`, in place of `paganin`):
  the local second-order closure of Mirone et al. (*J. Synchrotron Rad.* 32, 2025; arXiv:2601.22793, eq. 1) for a
  homogeneous object. Paganin inverts the first-order (TIE) model `I_det = I − A ∇²I`, with
  `A = δ/β · λL / (4π px²)` (px²; the ray shift in pixels is `A ∇ln I`); `epr` starts from that result and iterates
  `I += relax · Paganin[I_det − F(I)]`, where `F` adds the terms of second order in the ray shift (and the small
  diffraction term) of `src/common/epr_core.hpp`. `iters=0` is exactly `paganin`; `max_shift` limits the modelled
  ray shift (pixels). It needs 2 × Paganin's margin + 4 rows and runs at the intensity stage on CPU and CUDA,
  deterministically. It matters where the estimated ray shift approaches a pixel (`A` is 13.6 px² for the 8.64 µm
  PHerc0343P pair at δ/β 1000, and ~290 px² for a 1.1 µm, 0.22 m, 59 keV scan); where the shift is a few hundredths
  of a pixel the result is Paganin's. Not implemented: the non-local (ray-mapping) solver for shifts well beyond a
  pixel, oversampling, and the polychromatic spectrum lines of the reference implementation.
  Example: `--set "filters=epr(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)"`.
* **Scintillator light diffusion** (`sld="fraction=0.1,length=100"`, default `""`: none; `bin=8`, `stride=10`):
  the correction of the same paper (section 3.6) for the detector's long-range glare. The recorded frame is the
  image convolved with `K = (δ + f·t) / (1 + f)`, `t(r) ∝ exp(−r/length) / r` (pixels); the glare `S ∗ M`,
  `Ŝ = f t̂ / (1 + f t̂)`, is computed on frames binned by `bin`, subtracted from `(raw − dark) / current` before the
  flat division, and the flats are corrected the same way (`include/helirec/sld.hpp`). Each chunk reads the whole
  frame of every `stride`-th projection for its glare map (kept across consecutive chunks) and interpolates in
  between. `fraction` and `length` are properties of the scintillator and optics and must be measured; 0.1 / 100
  are the Night Rail defaults, not a calibration of any particular scan. Not available in `legacy_reference`.
  Example: `--set sld=fraction=0.1,length=100`.
  For ESRF **BM18 calibration files**, use the new full-resolution profile:

  ```bash
  helirec-slices PROJ -n 4 --set "sld=json=/path/to/diffusion.json" --tag bm18_sld
  # Explicit model, with two diffusion tails (illustrative values, not a calibration):
  helirec-slices PROJ -n 4 --set "sld=profile=bm18,fraction=0.06:0.03,length=40:200" --tag bm18_sld
  ```

  Night Rail JSON fields `fraction` and `scale_l` may be scalars or equal-length lists; lengths are in original
  detector pixels. The BM18 profile defaults to **100-pixel illuminated borders** on both axes, `bin=1` and
  `stride=1`. Override borders with `border_x=…`, `border_y=…` (zero means dark outside that detector edge).
  It uses the unit-mass multi-tail PSF from Nabuxx `nr_26_10_27`, correcting both flats and projections before
  flat division. Both low and high spatial frequencies are retained by full-frame FFTs; it does not approximate
  the high-frequency part with a cropped kernel. Flats are clipped at zero; projection corrections stay linear.
  This is the reference-quality path, with substantially more I/O and FFT work than the older binned setting.
  Each full frame is processed separately and only the rows needed by the current chunk are retained/uploaded;
  consecutive chunks recompute those full-frame corrections. `stride=4` explicitly trades temporal accuracy for
  fewer FFTs, interpolating current-normalized correction maps. Spatial binning is rejected for this profile.
  Correction FFTs run on the CPU; the corrected data can feed either CPU or CUDA reconstruction.

  The resolved model, borders and sampling are saved as `provenance.resolved_sld`. Nonzero shifted replicas and
  JSONs fitted with `save_binning != 1` are rejected rather than silently misinterpreted. Use a full-resolution
  calibration for the detector/scintillator configuration in question; fitted legacy parameters below should be
  rechecked when changing the illuminated-border assumption. Existing scalar specs retain their old behavior,
  and SLD remains disabled unless requested. Python frame checks use `helirec.sld.correct(frames, spec)`.

  Validation: `python -m pytest -q tests/test_sld.py tests/test_sld_bm18.py tests/test_fourier.py tests/test_provenance.py`.
  The small fixture `tests/reference/sld_bm18.json` is generated from Nabuxx commit
  `515e0df11cf34f669ae1cf918f219a8454c90e1c`, with NumPy FFTs replacing PyFFTW, by
  `python tests/reference/generate_sld_bm18.py /path/to/pinned/diffusion_digest.py` (source hash checked).

  `evaluation/calibrate_sld.py` measures them from the scan itself: air must reconstruct to zero, and glare leaves
  it positive around and inside the object. It reconstructs slices over a grid of (fraction, length) and reports the
  pair that brings the air closest to zero, in bands of distance to the object, separately for the open air and the
  air the object encloses:

  ```bash
  .venv/bin/python evaluation/calibrate_sld.py --projections PROJ --cache_dir CAL --out OUT --z 2698 \
      --fractions 0.3 0.4 0.5 0.6 0.7 --lengths 200 400 800
  ```

  PHerc0343P (GAGG 100 µm, 116 keV): uncorrected, the air within 0.3 mm of the object reads 0.012–0.015 cm⁻¹ (9 % of
  the nylon case) and 0.010 cm⁻¹ inside the case; 0.1 / 100 removes about a tenth of that; **fraction 0.6, length
  400** brings every band, open and enclosed, within 0.001 cm⁻¹ on slices z 2698 and 3598 (0.5 and 0.7 are both
  worse). The corrected contrast is higher (nylon 0.143 → 0.172 cm⁻¹; noise relative to contrast unchanged), and
  within about 10 px of a strong edge the single-scale kernel over-corrects slightly (a dip of ~4 % of the Teflon
  wrap's height beside it).
* **Gridding**: `luminosity_repartition`, `mend_threshold`, `n_grid`, `angular_tolerance_steps`, `block_angles`
  (memory/speed only), `x_interp`.
  * `z_interp=lanczos3|cubic|linear` (default `lanczos3`): a slice lies at a fractional detector row of every
    projection. The reference interpolates the two neighbouring rows linearly; along a helix the fraction sweeps all
    values, which makes that a triangle kernel along z: sinc²(ν), 0.81 at half Nyquist and 0.41 at Nyquist. The
    kernels use 6 / 4 rows wherever all of them are on the detector with a positive weight (else the two rows); the
    accumulated weights stay the linear ones. Measured on PHerc0343P (64 slices of papyrus, Paganin): the z spectrum
    amplitude rises by 5 % at 0.125, 18 % at 0.22 and 27 % at 0.27 cycles/pixel over `linear`, as 1 / sinc² predicts.
  * `angle_interp=nearest|linear` (default `nearest`): the reference splits every projection between the two grid
    angles around its own (weights 1 − α, α), an average of neighbouring views wherever the recorded angles sit off
    the regular grid (PHerc0343P: α drifts through all values, the step being 0.0449985° rather than 0.045°).
    `nearest` puts each projection whole into one bin, consecutive projections in consecutive bins
    (`helical.hpp`, independent of the chunk), and backprojects every slice at the measured mean angle of each bin
    (`debug["angles"]`). A skipped projection leaves its bin empty (filled as any empty angle). It needs a
    backprojector that takes each slice's angles: FBP, the whole-slice methods and bands; GHBP, whose plan fixes its
    angles, keeps the splat (`info["angle_interp"]` records which is in use).
  * `exposure_arc=auto|F` (default `auto`): in a continuous scan the detector integrates while the sample turns, a
    box blur along the angle of F angular steps (tangential, growing with the radius: 0.7 px at r = 1000 px for
    PHerc0343P). `auto` takes F from the scan (the recorded rotation swept per exposure / per period, 0.91 here;
    0 for step scans); its inverse 1 / sinc(F f) is applied to every sinogram as a 9-tap kernel along the angle
    (`sino.hpp`), at most 1.43 at the angular Nyquist frequency. 0: off.
    On PHerc0343P `nearest` and the arc correction together sharpen the tangential edge of the wrap at r ≈ 1200 px by
    3 % (Paganin only) to 7 % (with the unsharp mask); as the removed blurs also averaged noise, the noise in the outer
    air rises by 16 % (Paganin) to 33 % (unsharp).
  * `z_translation_scale=auto|S` (default `auto`): detector rows the sample really moves per row of recorded z
    translation, the calibration of the helical pitch. A slice is seen at every row of the detector as the sample
    travels; if the recorded travel is off by a fraction e, what two rows see of it is placed e × (rows between
    them) apart along z, a blur along z that no filter undoes. `auto` uses what the calibration measures from the
    frames (`calibration.measure_pitch`): two projections half a turn apart are mirror images about the rotation
    axis, displaced along the rows by the travel in between, so the displacement of ~40 such pairs over the
    recorded one is the scale (1 when the difference is not significant or nothing can be measured: a scan that
    does not move, an axis off the detector). PHerc0343P: 1.00316 ± 0.00003, i.e. 1053.4 rows per turn where the
    metadata gives 1050.1; against the 2.215 µm volume of the same sample the correlation at 0.39 cycles per voxel
    rises from 0.56 to 0.71 (0.10 to 0.74 for frequencies along z). `1` uses the recorded translations as they
    are. The volume's slice indices (z = view height − first doable height) and the sample's place in them change
    with the scale: compare runs by view height (`--heights`). The rotation-axis and overlap searches still use
    the recorded translations.
  * `angle_subset=I/N` (default: every projection): subset I of N interleaved subsets of the projections, chosen
    by angle (the same angles in every scan) and reconstructed on a grid of 1 / N of the angles. `0/2` and `1/2`
    are two reconstructions of one scan with independent noise and everything else in common
    (`evaluation/shell_correlation.py --pair`).
  * `x_interp=sinc|lanczos3|cubic|linear` (default `sinc`): how a scan at a fractional column offset in the merged
    frame is resampled (the PHerc0343P main scan sits at 843.5). `sinc` delays every projection row by the offset
    exactly, as a phase ramp on its Fourier transform (`shift_rows`, `sino.hpp`): no part of the band is lost. Any
    kernel of a few columns loses the top of it, and at half a pixel all of it at Nyquist:

    | half a pixel, response at | 0.25 | 0.36 | 0.45 cycles/pixel |
    |---|---|---|---|
    | `sinc` | 1.00 | 1.00 | 1.00 |
    | `lanczos3` (6 columns) | 1.02 | 0.83 | 0.35 |
    | `cubic` (4 columns) | 0.88 | 0.60 | 0.23 |
    | `linear` (the reference's two-column splat, cos(πν)) | 0.71 | 0.43 | 0.16 |

    Measured on PHerc0343P (Teflon film, main scan alone): `sinc` equals an integer placement of the scan to 0.001;
    against the 2.215 µm volume the default reconstruction's in-plane correlation rises from 0.60 / 0.40 / 0.18 to
    0.67 / 0.54 / 0.41 at 0.42 / 0.45 / 0.48 cycles/voxel over `lanczos3`, unchanged below 0.33. Cost: 0.035 s per
    32-slice chunk on an RTX 5090 (gridding 0.22 → 0.26 s). The accumulated weights are the splat's in every case.
    Next to the detector edges and to columns without weight the splat is used, so unmeasured values never enter.
    Notes: a row's two ends leak into a shifted row as 1 / distance (0.3 % of the high-frequency amplitude at the
    end, 400 columns in); the gridding of measured pixels only (SIRT, MLEM) uses `lanczos3` under `sinc`.
  * `companion_registration=auto|off` (default `auto`): the calibration compares the companion's frames with the
    main scan's frames of the same angle where the two detector positions overlap, at 240 projections along the
    scan (`calibration.measure_registration`; tiles registered by `calib::tile_shifts` on the continuous
    cross-correlation), and fits per pair a row displacement at the rotation axis and its slope along the columns:
    a roll of the companion's frames about the axis. One calibrated vector is right only in the middle of the
    overlap when there is a roll. Smoothed along the scan, the two are applied per projection in the gridding
    (`ScanTrajectory::row_offset`, `row_slope`): every sample is read at its registered row, in the one vertical
    interpolation there already was. PHerc0343P: roll −0.72 mrad (−0.53 … −0.91 along the scan), 0.1–1.1 rows at
    the axis; the companion's content lay −0.79 + 0.50 r / 1000 voxels from the main scan's along z (r: px from
    the axis), after it +0.03 + 0.07 r / 1000; axial correlation with the 2.215 µm volume 0.774 → 0.850 at 0.36
    and 0.564 → 0.674 at 0.45 cycles/voxel (centre block). 17 s once per scan (cached); +0.06 s per 32-slice
    chunk. `off`: the vector alone. The lateral part of the registration is measured (`details.registration`) and
    not applied.
  * `detector_smear=auto|off|a11:a12:a21:a22:c1:c2` (default `auto`), `sharp_f` (0.35), `sharp_band` (`0.15:0.30`):
    band-dependent redundancy weights for a detector that is not equally sharp everywhere. The field is a uniform
    smear of A r + c pixels at r = (column, row) from the detector's centre in units of 1000 pixels; `auto` takes
    the one the calibration fits to the noise of the scan's flats (`calibration.measure_smear`: the noise that
    passed the optics is smeared like the signal; none if it reaches less than 0.5 px). Below `sharp_band` the
    recordings of a ray are combined with the usual weights, above it with the usual weights × the detector's
    squared response at `sharp_f` along the columns, with a raised-cosine crossover: both sets sum to one, so no
    band is biased, and a zero field gives the usual result bit for bit. PHerc0343P (2 px of smear per 1000
    columns from column 1194): in-plane correlation with the 2.215 µm volume 0.859 → 0.885 at 0.36 and
    0.546 → 0.626 at 0.45 cycles/voxel, nothing lost below. Cost on an RTX 5090, whole slices: 6.9 → 7.5 s per
    32-slice chunk and 7.5 GB more GPU memory (a second gridded stack); use a smaller `--chunk` or `off` on a
    smaller card. FBP and GHBP; the gridding of measured pixels only and the legacy reference keep the usual
    weights.
  * The two-scan luminosity equalisation estimates each row's HA / TA difference in a 25 × 25 window at the seam
    and applies it as an offset of the intensity. Twelve context rows are read either side of a chunk, and context
    rows (the filters' too) are gridded from every projection that sees them, not only from those of the chunk's
    own heights, so that the window, and therefore the result, does not depend on the chunking (about 0.2 s more
    per chunk on the PHerc0343P pair).
* **Geometry**: `--cor` / `--vector` override the calibration for one run (never cached; not allowed in
  `legacy_reference`).
* **Calibration cache** (`--cache_dir`): darks/flats reduction and the measured calibration (rotation axis,
  overlap vector, helical pitch), keyed by a fingerprint of the acquisition and the calibration version; written
  atomically. A cache written before the pitch measurement existed gets it added on its next use.
* **New kernels**: add a function with the same signature in `src/cpu` and `src/cuda`; shared per-element
  arithmetic lives in `src/common/*_core.hpp` so the backends cannot drift apart.

## Accuracy

* `legacy_reference` follows the reference operation by operation, including its float32/float64 choices. GHBP
  uses explicit FMAs in nvcc's contraction order and CPU code is built with `-ffp-contract=off`, so CPU and CUDA
  GHBP are bit-identical and match nabuxx to 3e-8.
* The rotation-axis merit keeps the reference's filter roles in both modes, so the calibration matches it.
* The sub-pixel HA → TA overlap search (the reference's: L1 difference against the bilinearly interpolated
  companion) favours half-pixel offsets on noisy stripes, because the interpolation averages the companion's noise.
  The calibration repeats it on smoothed stripes and logs a warning if the two disagree by more than 0.375 px
  (`details.overlap.smoothed_sub`); the reference estimate is kept either way.

## Method comparison (`evaluation/compare_methods.py`)

Reconstructs the same slices with each method of a JSON config (default `evaluation/default_methods.json`: FBP,
SIRT, MLEM and gridrec on unfiltered data) and writes float32 `slices/<method>/z<Z>.tif` and `manifest.json`
(time, mean, std, fine-scale std). PNG contact and crop sheets are opt-in with `--previews` (the slice folders hold
TIFFs only); `helirec-slices --previews` writes PNG slices. Preview windows are shared in attenuation units; legacy TIFFs retain native reference units.
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

`evaluation/343p_epr_sld.json` adds, next to those baselines and over the same five slices, eikonal phase retrieval
in place of Paganin and the scintillator diffusion correction (uncalibrated Night Rail defaults), each with the
reference unsharp mask:

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python -u evaluation/compare_methods.py evaluation/343p_epr_sld.json \
    --out /mnt/raid_nvme/343p_recon_work/output/baselines --run-name epr_sld_v1
```

`evaluation/343p_v4_baselines.json` repeats the sharpened Paganin baselines (FBP and SIRT, 20 iterations from
gridrec), with and without the scintillator diffusion correction, after the gridrec scale, column-offset kernel and
luminosity-window corrections (`_v4` folders):

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python -u evaluation/compare_methods.py evaluation/343p_v4_baselines.json \
    --out /mnt/raid_nvme/343p_recon_work/output/baselines --run-name v4
```

`evaluation/343p_v5_resolution.json` compares 16 FBP variants on two slices: the previous methods (no filters,
Paganin, + unsharp, + SLD, eikonal) with the previous gridding (`z_interp=linear`, `angle_interp=linear`,
`exposure_arc=0`), and the v5 gridding with the unsharp mask, the fused `mtf_paganin` filter (three `nsr`, and the Gaussian target at δ/β 1000 and 1250), the
calibrated SLD and eikonal phase retrieval. The earlier recipes pin the previous gridding, so they keep reproducing
their folders bit for bit.

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python -u evaluation/compare_methods.py evaluation/343p_v5_resolution.json \
    --out /mnt/raid_nvme/343p_recon_work/output/resolution_v5/compare --previews
```

With `--previews` every sheet tile carries a header with the method's name, description and display
window. All methods of a config share one window (`window`: here the standard −0.06 … 0.29 cm⁻¹), so a grey level
is the same attenuation in every image; methods are not rescaled individually (the SLD-corrected ones are brighter
because their attenuation is higher). `crops` writes full-resolution crop sheets `sheets/z<Z>_crop<K>.png`;
`sheet_columns` wraps the tiles.

`tests/test_resolution.py` checks the fused filter's response, the vertical kernels against a NumPy reference and a
row pattern, nearest-bin gridding and its angles on drifting angles with a missing projection, and the exposure-arc
kernel, each on CPU and CUDA and across chunkings.

`tests/test_epr.py` checks `epr` against exact Fresnel propagation of a homogeneous object (about half of Paganin's
error where ray shifts reach a fraction of a pixel, Paganin's result where they do not), CPU/CUDA parity and chunk
independence; `tests/test_sld.py` checks the glare model against the exact point spread and through the
reconstructor.

`tests/test_gridding.py` checks the column-offset kernels against their responses on both gridding paths, the
chunk independence of the luminosity window, and the calibration guards.
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

## Shell correlation (`evaluation/shell_correlation.py`)

How much of a reconstruction is signal, per spatial frequency and direction: the 3D Fourier shell correlation of
volumes against a registered reference volume of the same sample (a finer scan: `--ref` JSON with its location and
the affine from the volumes' grid, as the open-data catalogue's `volume_transforms`), and / or between two
reconstructions of one scan (`--pair`, e.g. `angle_subset=0/2` and `1/2`). Computed on cubic blocks with
`helirec.BlockCorrelator` (`block_spectra.hpp`, CPU and CUDA): the reference is resampled twice finer than the
volumes' grid and band-limited to it in the Fourier domain, registered per block by a sub-voxel shift, and the
cross-spectra are summed in bins of |k| and of the angle of k from the z axis, so that in-plane and axial
frequencies are reported apart. `matched` is the number to compare reconstructions with (no linear filter that is
constant over a bin changes it; the plain shell correlation rises under a blur along z). Sharpening filters leave
every curve where it was.

```bash
.venv/bin/python evaluation/shell_correlation.py --ref evaluation/343p_reference_2p215um.json \
    --volume base=RUN/base/volume.zarr --volume unsharp=RUN/unsharp/volume.zarr --baseline base --out OUT
```

Against a reference the implied signal-to-noise ratio is a lower bound (the reference has noise of its own); between
two halves of a scan it is an upper bound, and blind to anything all projections share (a geometry error leaves the
halves in agreement). `evaluation/detector_noise.py` measures the detector's noise spectrum, lag and flat-field
stability from a scan's flats and darks.

## Speed

Seconds per slice on the PHerc0343P pair (8000 × 3372 sinogram, 5057² slice), RTX 5090, raw data in the page cache
(`evaluation/343p_v4_baselines.json` methods; isolated slices as `compare_methods.py` reconstructs them, and
consecutive 8-slice chunks):

| method | isolated slice | in 8-slice chunks |
|---|---|---|
| FBP, Paganin + unsharp, `z_interp=linear angle_interp=linear exposure_arc=0` | 0.81 | 0.27 |
| FBP, SLD + Paganin + unsharp | 1.35 | 0.31 |
| SIRT 20 iterations from gridrec, Paganin + unsharp | 5.4 | 5.0 |
| SIRT, SLD + Paganin + unsharp | 5.9 | 5.0 |

The table predates the default Lanczos-3 rows, nearest-bin angles and arc correction: with them the FBP chunk time
rises by 11 % (0.29 → 0.32 s per slice measured back to back; the gridding reads six rows per sample instead of two,
the other two are free). `mtf_paganin` costs as `paganin + unsharp` per row, with 33 more context rows either side
of a chunk (default target form; 18 for `nsr=0.01`).

A SIRT iteration is 0.22 s (forward 0.09 s, backward 0.13 s); the exact backprojection of one slice 0.16 s. An
isolated slice uploads 8.7 GB of raw rows (0.4 s), and with SLD reads 12 GB of whole frames for its glare maps: from
disk rather than the page cache that read dominates. The first chunk of a process also page-locks its raw buffers
(1.4 s).

## Tests

`pytest` runs the unit tests: CPU vs CUDA vs NumPy references for every kernel, projector adjointness, the
whole-slice methods, and regression checks of the corrected conventions (`tests/test_regressions.py`).

## Build

Requirements: CMake ≥ 3.24, a C++20 compiler, FFTW3 (float + double), HDF5 (C), zlib, libcurl, Python ≥ 3.10, and
optionally the CUDA toolkit. The vc-delta3d codec is fetched at configure time (offline:
`-DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D=<src>`). CUDA code is compiled for the build machine's GPU (`native`); override with
`-DCMAKE_CUDA_ARCHITECTURES=...`. `HELIREC_CUDA=AUTO` (default) builds CUDA only if `nvcc` is found, so put it on
`PATH` (or pass `-C cmake.define.HELIREC_CUDA=ON`); check with `helirec.devices()`.

Run from `foundation/helirec` in a Villa checkout containing `vc_reader` at `../../vc_reader`: `helirec-composite`
uses the vc_reader Python package (surfaces), which `pyproject.toml` takes from there; `uv sync` includes it. The
native library itself does not use vc_reader.
To also install the test dependencies:

```bash
uv sync --python 3.12 --extra test
.venv/bin/python -m pytest -q
```

## Limitations

These raise explicit errors:
* exactly one companion scan, on the left of the main scan, with the rotation axis in the main scan's right half;
* both scans moving in the same z direction;
* no per-projection lateral-correction drift: a rotation axis measured more than 0.25 px apart at the two ends of
  the scan is an error (one axis is used for every slice; `--cor` reconstructs with a fixed one);
* parameter sweeps (`reconstruct_band(..., sweep=...)`) run on CUDA only;
* xz slices are not implemented.

The calibration kernels are CPU-only (they are cheap). The helical pitch is one scale for the whole scan: a
stage that comes to rest away from its recorded position is not corrected (PHerc0343P: the projection pairs that
reach into the final stationary turn sit about one row off the scale).
