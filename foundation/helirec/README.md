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
  a 16-row Lanczos kernel (`z_interp=lanczos8`); each projection gridded whole into a sinogram row of its own and backprojected at its measured angle
  (`angle_interp=views`; GHBP keeps the two-bin splat); every projection read where the geometry calibration
  puts it (`geometry`: the detector map, the frame's own row, axis column, roll and scales, and the angle of each
  of its rows; `geometry.json`, made on first use by the geometry calibration step and cached); recordings of a ray
  combined with band-dependent weights where the detector's sharpness depends on the place
  (`sharpness_weighting=auto`). No projection filters by default, and since 2026-11-11 the exposure arc of
  a continuous scan is not deconvolved (`exposure_arc=0`; `auto` turns it on): that is left for the intensity side of
  the calibration, after the geometry.
* `legacy_reference`: the reference workflow's conventions (half-attenuation units, its rebalance endpoints,
  integer padding reflection, mending bounds, chunk-dependent Paganin and luminosity window, two-column splat of
  fractional column offsets, linear rows, two-bin angular splat, no exposure-arc correction) with every parameter pinned (GHBP, Paganin 1000 + unsharp 1.2/4, …); overriding one
  is an error. Of the geometry calibration it takes the merged frame alone: one rotation axis and one vector
  between the scans (on a quarter-pixel grid, as the reference's searches gave them), with the recorded angles and
  heights.

Outputs record the mode, its conventions and every resolved parameter (`manifest.json`, OME-Zarr attributes).

## Slices and volumes (`helirec-reconstruct`)

```bash
helirec-reconstruct PROJ -n 8 --tag unfiltered                                    # 8 slices, default mode
helirec-reconstruct PROJ -n 8 --set mode=legacy_reference --tag reference         # the reference workflow
helirec-reconstruct PROJ --fractions 0.4 0.5 --set "filters=paganin(delta_beta=500)+unsharp(sigma=1.2,coeff=4)" --tag db500
helirec-reconstruct PROJ -n 4 --device cpu --tag cpu
helirec-reconstruct PROJ -n 8 --set "filters=mtf_paganin(delta_beta=1000,sigma=1.06)" \
    --set sld=fraction=0.6,length=400 --tag mtf        # fused phase retrieval + detector deblur, calibrated glare
```

Range mode reconstructs contiguous slices in chunks (`--chunk`, default 32) and writes an OME-Zarr (NGFF 0.4,
Zarr v2) by default:

```bash
helirec-reconstruct PROJ --z 900 1700 --roi X0 X1 Y0 Y1 --uint8 --levels 4 --tag block       # float32 + uint8 OME-Zarr
helirec-reconstruct PROJ --heights H0 H1 --format tiff --uint8 --tag block_tif                 # TIFF stacks
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
  when the reference was made with the recorded helical pitch and not the measured one: its slices are then
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
| 2 | intensity calibration: darks/flats, weight map + apodisation, the detector's blur field; the merged frame | `calibration/intensity.hpp`, `python/helirec/calibration.py` | CPU |
| 2b | geometry calibration: layout, rows' timing, detector map, poses and motion (`helirec-geometry`) | `calibration/`, `python/helirec/geometry.py` | CPU, CUDA |
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

* `fbp(interp=linear|cubic|lanczos3|bspline3|bspline5, oversample=1..8)` (default): exact pixel-driven
  backprojector; `oversample` FFT-upsamples the filtered rows before interpolating (default bspline3, 2×).
* `ghbp`: the reference hierarchical backprojector (pinned by `mode=legacy_reference`); keeps the two-bin angular splat.
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
  `epr`, `mtf_paganin`, `wiener`, `unsharp`, `gaussian`, `median`, `python`; register more in C++ (`register_filter`, `filters.hpp`) or Python
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
* **Wiener filter from measured tables** (`wiener(table=FILE, snr=snr, weight=1, target=0)`, in place of
  `paganin + unsharp` or `mtf_paganin`): the minimum-error linear estimate of the object when the system's transfer
  `T` (detector blur × phase contrast × what the gridding does) and the signal-to-noise power ratio `S` of the scan
  have been measured: `W = g / T`, `g = S / (S + weight)`. The output shows the object through `g`: flat where the
  data carry it, falling where they do not; no δ/β and no sharpening strength to choose. `T` and `S` are tables
  over the frequency magnitude ν (cycles per pixel) and, optionally, its angle θ from the row axis of a frame (the
  scan's z), in a JSON file `{"nu", ["theta",] "transfer", "snr"}` (several `snr` tables may be kept; `snr=NAME`
  picks one). A filter of the frames that depends on (ν, θ) only is a filter of the volume symmetric about z at the
  same frequencies, so the tables can be measured in the volume. `weight` scales the noise (1: Wiener), `target`
  multiplies `g` by the MTF of a Gaussian of that many pixels. Fixed unit-DC kernel as `paganin`'s; the margin is
  where the kernel's rows beyond it hold less than 2 % of it (18 rows for the table below, against 31 for
  `paganin + unsharp`): no more cost. Give smooth, finely sampled tables (a kink is a long kernel).
  `evaluation/wiener_table.py` builds the file from an in-plane transfer measured on a thin film of the sample
  (optionally also where the detector is sharpest, which separates a blur along the columns) and the two half
  reconstructions' shell correlation. By default no direction gets more signal-to-noise ratio than the in-plane
  one, and the loss of the interpolation between detector rows is not undone (`--snr_direction measured`,
  `--row_response restore` do otherwise: on PHerc0343P that left a ridge of power at the slices' Nyquist
  frequency, alternating slices). Inputs: the film's transfer and the halves' shell correlation (`angle_subset=0/2`, `1/2`;
  `evaluation/shell_correlation.py --pair`):
  ```bash
  .venv/bin/python evaluation/wiener_table.py wiener.json --transfer film_transfer.json --sharp ta_nophase 1250 1400 \
      --curves RUN/fsc_halves --pair half0 half1 [--filtered base]
  helirec-reconstruct PROJ --set backprojector=fbp --set "filters=wiener(table=wiener.json)" ...
  ```
  On PHerc0343P (8.64 µm, glare correction 0.6 / 400) the Teflon wrap is reproduced through 1.00 ± 0.03 up to
  0.28 cycles per pixel and through the designed `g` beyond (0.84 at 0.36, 0.56 at 0.45), where
  `paganin(1000) + unsharp(1.2, 4)` passes 1.5–1.9 at 0.06–0.25 and 0.39 at 0.45, and `paganin(1000)` alone 0.33 at
  0.25. Solid nylon reads 0.1730 cm⁻¹ (Paganin: 0.1731; with the unsharp mask 0.1789); noise in air 0.0128 cm⁻¹
  (Paganin + unsharp: 0.0129; with `snr=snr_mean, target=0.3`: 0.0086). The correlation with a finer scan of the
  sample is unchanged in every band and direction, as for any linear filter. The result is quantitative and has
  no rims around sheets; it is flatter than Paganin + unsharp, whose emphasis of 4–16 px structure it does not
  have. Check a new table on a vertical reslice and on the power spectrum along z before using it.
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
* **Scintillator light diffusion** (`sld="fraction=0.1,length=100"`, default `""`: none; `bin=8`, `stride=10`,
  `outside=dark|lit`):
  the correction of the same paper (section 3.6) for the detector's long-range glare. The recorded frame is the
  image convolved with `K = (δ + f·t) / (1 + f)`, `t(r) ∝ exp(−r/length) / r` (pixels); the glare `S ∗ M`,
  `Ŝ = f t̂ / (1 + f t̂)`, is computed on frames binned by `bin`, subtracted from `(raw − dark) / current` before the
  flat division, and the flats are corrected the same way (`include/helirec/sld.hpp`). Each chunk reads the whole
  frame of every `stride`-th projection for its glare map (kept across consecutive chunks) and interpolates in
  between. `fraction` and `length` are properties of the scintillator and optics and must be measured; 0.1 / 100
  are the Night Rail defaults, not a calibration of any particular scan. Not available in `legacy_reference`.
  `outside` says what lies beyond the detector's field: `dark` (default: nothing) or `lit` (the frame continued by
  its edge values over the padding, for a beam wider than the field; there `dark` takes the light that comes in
  from beyond the edge for the image's own and can do worse near the edges than no correction).
  The numbers can be read off the projections where a sharp edge has open beam beside it: the air's deficit
  against the distance from the edge is the kernel's one-sided integral. On PHerc0343P (8.64 µm) the wall of the
  sample's case gives fraction 0.43, length 170 px (0.48 / 150 to 0.37 / 200 fit alike), with `outside=lit`
  agreeing better than `dark` between the two scans; the 0.6 / 400 that `evaluation/calibrate_sld.py` found by
  bringing the reconstructed air to zero spreads 37.5 % of the light where the edge shows 30 %, and reverses the
  air's dependence on the angle. `--set sld=fraction=0.43,length=170,outside=lit`.
  Example: `--set sld=fraction=0.1,length=100`.
  For ESRF **BM18 calibration files**, use the new full-resolution profile:

  ```bash
  helirec-reconstruct PROJ -n 4 --set "sld=json=/path/to/diffusion.json" --tag bm18_sld
  # Explicit model, with two diffusion tails (illustrative values, not a calibration):
  helirec-reconstruct PROJ -n 4 --set "sld=profile=bm18,fraction=0.06:0.03,length=40:200" --tag bm18_sld
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
  * `z_interp=lanczos8|lanczos3|cubic|linear` (default `lanczos8`): a slice lies at a fractional detector row of every
    projection. The reference interpolates the two neighbouring rows linearly; along a helix the fraction sweeps all
    values, which makes that a triangle kernel along z: sinc²(ν), 0.81 at half Nyquist and 0.41 at Nyquist. The
    kernels use 16 / 6 / 4 rows wherever all of them are on the detector with a positive weight (else the two rows); the
    accumulated weights stay the linear ones. Measured on PHerc0343P (64 slices of papyrus, Paganin): the z spectrum
    amplitude rises by 5 % at 0.125, 18 % at 0.22 and 27 % at 0.27 cycles/pixel over `linear`, as 1 / sinc² predicts.
    `lanczos8` (the default since 2026-11-12; `lanczos3` before) uses 16 rows: lanczos3 still loses the top of the
    band along z (over the fractions of a helix it keeps 0.82 of the amplitude at 0.40 cycles/pixel and 0.68 at
    0.45; lanczos8 1.00 and 0.90). Measured on PHerc0343P
    (FBP, ramp only): the signal's power along z rises by 39 % at 0.35–0.45 and 29 % at 0.45–0.5 cycles/pixel over
    `lanczos3` and is unchanged below 0.3, for 2.4 % more time per chunk and 10 more rows read of every projection.
    A value within 7 rows of an unmeasured row or of the detector's ends takes the two rows, as with every kernel.
  * `angle_interp=views|nearest|linear` (default `views`): the reference splits every projection between the two grid
    angles around its own (weights 1 − α, α), an average of neighbouring views wherever the recorded angles sit off
    the regular grid (PHerc0343P: α drifts through all values, the step being 0.0449985° rather than 0.045°).
    `nearest` puts each projection whole into one bin, consecutive projections in consecutive bins
    (`helical.hpp`, independent of the chunk), and backprojects every slice at the measured mean angle of each bin
    (`debug["angles"]`). A skipped projection leaves its bin empty (filled as any empty angle). It needs a
    backprojector that takes each slice's angles: FBP, the whole-slice methods and bands; GHBP, whose plan fixes its
    angles, keeps the splat (`info["angle_interp"]` records which is in use).

    `views` is `nearest` without the wrap to one turn. Where the detector is taller than the travel per turn a slice
    is seen twice at a grid angle, a turn apart (and again by a turn at rest), and those views were not recorded
    at one angle: a turn is not a whole number of steps, and with a rolling shutter the rows they are read at were
    exposed at different times. PHerc0343P: the two views of a bin are 8000 frames apart and differ by 0.90 frames
    by the record and the row timing (0.27 from the turn of 8000.27 frames, 0.63 from 1053 rows of row timing, the
    same way) and by 0.93–0.94 with the helix's measured excess: 0.7 px at 1000 px from the axis, in 13.8 % of the
    angles in the helix; a rest turn against the helix 0.27 to 0.90 frames. `nearest` averages them into one row at
    their mean angle. `views` keeps a sinogram row per bin of every turn (`debug["sino"]` has `n_rows >= n_grid`
    rows, row r being bin `debug["first_bin"] + r`): each view is ramp-filtered and backprojected at its own angle,
    with its share of the weight, the balanced weights now a partition of unity over every recording of a line
    (the rows a half turn's multiple away; `rebalance_turns`, `sino.hpp`). Values and weights are those `nearest`
    grids; with the turns at one angle the slices are the same to 1e-7. On an analytic phantom with views 0.4 to
    0.8 of a step apart the error of `nearest` is 2.6 times that of the aligned case and that of `views` is the
    aligned case's (`tests/test_views.py`). What is not exact in it: the weights of two views of a bin are shared
    bin by bin, not at equal angle, which shows only where a weight jumps (a missing projection, a weight map
    without a taper at its top and bottom).

    PHerc0343P, geometry calibration, ramp filter only, the signal shared by reconstructions from two interleaved
    halves of the angles (cross-power in blocks of 512 px, 16 slices), `views` over `nearest`: in the middle of the
    helix nothing changes outside one sector of each slice (which turns with the height); in it, 570–1080 px from
    the axis, the power rises by 4–8 % at 0.15–0.25 and 7–12 % at 0.25–0.35 cycles/px in the 45° of directions
    the doubled views measure (1.7 % over all directions), and by a third at 1000–1500 px (outside the sample
    there). Where the last rest turn and the helix both record: 5–6 % over all directions at 0.25–0.35 cycles/px
    and 10–25 % at 0.35–0.5 at 520–1080 px, the split-half correlation 0.691 → 0.718 and 0.880 → 0.889 at
    0.25–0.35 (`development_notes/calibration/05_experiments/exp34_views.py`). Cost: the sinogram has 9523 rows
    for 8000 (helix) and about 2.2 times as many with a rest turn; a chunk of 32 whole slices without projection
    filters takes 8.7 s for 7.6 s (the backprojection's rows) and the device holds 15.3 GB for 13.4 GB, with
    `sharpness_weighting=auto` 9.2 s for 8.0 s and 22.7 GB for 19.7 GB; the slab of 256 slices with Paganin and
    unsharp, masked uint8, 46–47 s for 44 s, with the weighting 54 s for 50–51 s (18.4 and 27.3 GB at most on the
    device). `nearest` itself is unchanged: the slab is the same bytes as before. It needs the exact backprojector
    (FBP, bands; at most 16 turns of rows); the whole-slice methods use `nearest`. The exposure arc is not
    deconvolved with it (below).
  * `exposure_arc=auto|F` (default `0`: off): in a continuous scan the detector integrates while the sample turns, a
    box blur along the angle of F angular steps (tangential, growing with the radius: 0.7 px at r = 1000 px for
    PHerc0343P). `auto` takes F from the scan (the recorded rotation swept per exposure / per period, 0.91 here;
    0 for step scans); its inverse 1 / sinc(F f) is applied to every sinogram as a 9-tap kernel along the angle
    (`sino.hpp`), at most 1.43 at the angular Nyquist frequency. 0: off. **Disabled for now with
    `angle_interp=views`** (the default): the kernel runs along one turn of grid angles and has not been carried
    over to a row per view; `info["exposure_arc_bins"]` is then 0 and `helirec-reconstruct` says so. `angle_interp=nearest`
    applies it.
    On PHerc0343P `nearest` and the arc correction together sharpen the tangential edge of the wrap at r ≈ 1200 px by
    3 % (Paganin only) to 7 % (with the unsharp mask); as the removed blurs also averaged noise, the noise in the outer
    air rises by 16 % (Paganin) to 33 % (unsharp).
  * `geometry=PATH` (default: `geometry.json` in the calibration cache, made there on first use), `row_time_us`:
    the geometry calibration (`helirec-geometry`, below), the only one. Every projection is read on the grid of its
    detector map, at the row, axis column, roll and scales it gives that frame and at the angle of each of its
    rows (the rows' timing), in a merged frame laid out from the same poses. It holds what the earlier calibration's
    options stood for, which are gone and refused by name: the helical pitch (`z_translation_scale`), the turns at
    rest against the helix (`stationary_registration`), the axis' refinement (`axis_refinement`), the second
    scan's frames on the first's (`companion_registration`) and the rolling shutter (`rolling_shutter`).
    `row_time_us`: the time of a detector row within the frame in microseconds per stored row (signed), for making
    the calibration when no stretch of the scans holds a refill of the ring to measure it from.
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
    | `lanczos3` (6 columns) | 1.02 | 0.82 | 0.35 |
    | `cubic` (4 columns) | 0.88 | 0.60 | 0.23 |
    | `linear` (the reference's two-column splat, cos(πν)) | 0.71 | 0.43 | 0.16 |

    Measured on PHerc0343P (Teflon film, main scan alone): `sinc` equals an integer placement of the scan to 0.001;
    against the 2.215 µm volume the default reconstruction's in-plane correlation rises from 0.60 / 0.40 / 0.18 to
    0.67 / 0.54 / 0.41 at 0.42 / 0.45 / 0.48 cycles/voxel over `lanczos3`, unchanged below 0.33. Cost: 0.035 s per
    32-slice chunk on an RTX 5090 (gridding 0.22 → 0.26 s). The accumulated weights are the splat's in every case.
    Next to the detector edges and to columns without weight the splat is used, so unmeasured values never enter.
    Notes: a row's two ends leak into a shifted row as 1 / distance (0.3 % of the high-frequency amplitude at the
    end, 400 columns in); the gridding of measured pixels only (SIRT, MLEM) uses `lanczos3` under `sinc`.
  * `sharpness_weighting=auto|off|a11:a12:a21:a22:c1:c2` (default `auto`), `sharp_f` (0.35), `sharp_band` (`0.15:0.30`):
    (named `detector_smear` until 2026-11-11; that key is still accepted, and calibration caches written with the
    old names still load.)
    band-dependent redundancy weights for a detector that is not equally sharp everywhere. The field is a uniform
    blur of A r + c pixels at r = (column, row) from the detector's centre in units of 1000 pixels; `auto` takes
    the one the calibration fits to the noise of the scan's flats (`calibration.measure_detector_blur`: the noise that
    passed the optics is blurred like the signal; none if it reaches less than 0.5 px). Below `sharp_band` the
    recordings of a ray are combined with the usual weights, above it with the usual weights × the detector's
    squared response at `sharp_f` along the columns, with a raised-cosine crossover: both sets sum to one, so no
    band is biased, and a zero field gives the usual result bit for bit. PHerc0343P (2 px of blur per 1000
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
* **Calibration cache** (`--cache_dir`): the darks / flats reduction and the detector's blur field, keyed by a
  fingerprint of the acquisition and the calibration version, and `geometry.json`, the geometry calibration of the
  two scans, made there by the geometry calibration step when it is missing or is other scans' (about 13 minutes
  on PHerc0343P; `--set geometry=PATH` uses one made before and makes nothing). Written atomically. A
  `calibration.json` of the earlier calibration is read for its blur field and otherwise left alone.
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
TIFFs only); `helirec-reconstruct --previews` writes PNG slices. Preview windows are shared in attenuation units; legacy TIFFs retain native reference units.
Options: `--projections`, `--cache_dir`, `-n`, `--z Z0 Z1`, `--roi X0 X1 Y0 Y1` (`-1`: whole slices),
`--methods NAME ...`, `--overwrite`; existing slices with matching provenance are reused.

`evaluation/343p_baselines.json` compares FBP and SIRT (20 iterations from gridrec), each with no projection
filters and with Paganin δ/β 1000 + unsharp σ 1.2, coefficient 4, over the same five full slices of the local PHerc0343P projections.
It uses the corrected mode and CUDA backend, requiring a CUDA-enabled build and host.
The calibration cache lives under the output directory. Ceres must already be installed; configuration uses `find_package(Ceres 2.2 REQUIRED COMPONENTS EigenSparse)`
and links `Ceres::ceres`. There is no bundled download or fallback solver. On Ubuntu use the distribution's
`libceres-dev` package if it provides version 2.2 or newer (`sudo apt install libceres-dev`); on macOS use
`brew install ceres-solver`. For a separate installation, pass `-DCeres_DIR=/prefix/lib/cmake/Ceres`
(or the directory containing `CeresConfig.cmake`). A Ceres source build must enable `EIGENSPARSE`.

For an already configured build, use `cmake --build build/<configuration> -j 4` and
`ctest --test-dir build/<configuration> --output-on-failure`. The native fitting test checks the complete
blur-model Jacobian against numerical derivatives. Python fitting coverage is in `tests/test_solve.py`,
`tests/test_refill.py`, `tests/test_geometry.py`, and `tests/test_regressions.py`; registration is covered by
`tests/test_calibration.py` and `tests/test_block_spectra.py` on each available backend.

Run from `foundation/helirec`:

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
kernel, each on CPU and CUDA and across chunkings. `tests/test_views.py` checks the row per view: the balanced
weights, hole filling and ramp padding of a sinogram of more than one turn against NumPy, that it gives the slices
of `nearest` when the turns are at one angle (also with missing projections and two scans) and the aligned case's
error when they are not (a turn that is not a whole number of steps, a rolling shutter), chunk independence, bands
and sweeps.

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

## Checking an acquisition record (`helirec-calibration`)

`helirec::calibration` (`include/helirec/calibration/`, `src/calibration/`) checks what the acquisition record of
a scan states, first against itself and then against the raw frames. It shares nothing with the calibration the
reconstruction uses (`helirec/calibration/intensity.hpp`, `helirec.calibration`): it reads the BLISS master file and the frame
files directly, and the only processing is a dark subtraction and a division by the mean flat of the scan.

```bash
cmake --build build/<tree> --target helirec-calibration
helirec-calibration selftest [--device cpu|cuda]          # the registration estimator on synthetic data
helirec-calibration record <scan.h5> [<more scans.h5> ...]   # the record against itself; no frame is read
helirec-calibration verify <scan.h5> [<more scans.h5> ...] --out DIR [--checks record,inventory,series,beam,views]
```

The scans are one scan; two scans of one acquisition that differ in the lateral stage of the rotation axis, the
one with the axis inside the detector's field first; or the tiles of a mosaic (below), the reference tile first.
`verify` prints one finding per check (what the record states, what was found, and `verified`, `fails`, `unverified` or `note`) and
writes `report.txt`, `report.json` and `views.tsv` (every registered block) to `DIR`.

The geometry target is **0.01 pixel** (`--position-target PX`, Python `position_target=...`, C++
`VerifyOptions::view.eps`); angle, axis and pitch targets follow from their geometric lever arms. Every accepted
block is checked: any residual above target fails, even if the mean is small or its uncertainty overlaps zero.
Reports retain min/max, RMS, maximum absolute residual, counts above target, and the descriptive block SEM.
Smaller discrepancies remain visible and require explanation; sampled agreement alone is `unverified`.
`verified` is restricted to the specific record or identity check performed. Other findings carry
`requires_follow_up: true`, including measured facts without a validated model and rejected/unsampled views.
These verdicts assess the evidence and model; they do not identify the physical cause of a discrepancy.
The report schema replaces the former `holds`/`open` labels; clients should use the new labels and follow-up flag.
The CLI returns zero when it successfully writes a report, not when the acquisition meets every requirement.

`helirec::calibration::assess_residuals` and Python `calibration_assess_residuals` expose the same per-sample
assessment for other checks. They operate on small measurement summaries; the image registration and its
CPU/CUDA arithmetic are unchanged by the reporting policy.

The checks:

* `record`: frame counts, the map from frame to file, the trigger timer, clocks, the derived angles and heights
  against the raw encoder counts, commanded against recorded travel, the scans against each other;
* `inventory`: every frame on disk is read once and its sum, minimum and maximum are compared with the statistics
  the detector software recorded for it (`--inventory DIR` reuses the pass of an earlier `--out DIR`).
  `--inventory-step N` reads one frame in N; by default every frame is read, except in a stretch that is stored
  compressed and has no recorded statistics, where nothing can be compared with a frame and decoding is what
  takes the time: there one frame in 16 is read and every chunk of every frame is located in its file. The
  finding says what was read;
* `series`: the dark and flat frames; `beam`: the level of the air beside the sample against the ring current;
* `views`: blocks of frames that recorded the same lines are registered to a fraction of a pixel, row and frame
  (`helirec/calibration/registration.hpp`): opposite views of a turn at rest (axis column, angle over half a
  turn), a turn at rest against the helix and the helix a turn apart (travel per turn in rows, offsets between the
  stretches), and the second scan against the first (lateral move, height, angle).

The registration exists for the CPU (FFTW) and for CUDA (cuFFT, `src/cuda/registration.cu`), one estimator on
two implementations of its array work; `--device cpu|cuda|auto` chooses (`auto`, the default, takes the GPU when
there is one). On the 428 blocks of the 343P pair (32 frames × up to 256 rows × up to 2500 columns) a registration
takes 0.151 s on the CPU (8 FFT threads) and 0.016 s on an RTX 5090, uploads included; the two agree to 1e-5 of a
sample on every block.

The same is available from Python as `helirec._core.calibration_verify(...)` (the report as JSON),
`calibration_register_blocks` and `calibration_selftest`, each with a `device` argument.

For measurements the fixed checks do not make, `helirec._core.CalibrationSession(scans)` keeps the scans of an
acquisition open: `stretches`, `series` (the record's per-frame angles, heights and times), `read_rows` (raw
frames as stored), `line_integrals` (the scan's own dark and mean flat, nothing else), and
`measure(requests, row_sign, device)`, which registers pairs of blocks placed by the caller exactly as the view
checks cut and register theirs (C++: `helirec::calibration::Session`, `ViewRequest`, `ViewMeasurement`).
`CalibrationRegistrar(shape, margin)` keeps one registration geometry for many pairs of blocks. The experiments
under `development_notes/calibration/04_experiments/` are written on these; none of them uses the
reconstruction's calibration.

**Patches.** `measure_patches(requests, row_sign, patch_rows, patch_cols, device)` cuts the block of each request
into patches and measures every patch as `measure` measures a request of that size at that place: the same cut,
the same registration, the same re-cuts by whole samples, and the same numbers (equal on CUDA, to 2e-7 px on the
CPU, whose registrars here run on one thread each so that the result does not depend on the number of threads).
What differs is the work: the frames of a block are read and converted once for all its patches, a patch that
the registration moves by whole frames, rows or columns is cut again from that reading (B is read once more only
where patches drift past a margin of 1 frame and 4 rows and columns), and the patches are registered side by
side, one per thread on the CPU and four at a time on a CUDA device (C++: `Session::measure_patches`,
`PatchMeasurement`; `src/calibration/view_patches.cpp`). Each patch also returns `curvature`, the second
derivatives of the correlation at its maximum over frames, rows and columns: how sharply each shift is determined
and how the three are tied. `calibration_measure_timing(reset)` says where the time of `measure` and
`measure_patches` went. On PHerc0343P the 1320 blocks of the geometry step's plan as 20 256 patches of
32 frames × 128 × 128 take 91 s on an RTX 5090 (22 s reading, 9 s converting, 58 s registering) and 218 s on 24
cores (183 s registering); half of those patches as requests of their own through `measure` took 448 s on the
CPU. `tests/test_calibration.py::test_measure_patches_equals_measure` holds the two routes together. With
`from_block=True` each block is registered whole first, from the same readings, and its patches start where it
ends instead of at the plan: for a plan that is further off than a patch can find by itself (the test's patches
five columns off: three of four found without it, all four with it). Where the plan is within a patch's reach
it costs more than it saves (Paris4, 128 opposite-view blocks across tiles: 137 ms a block with it, 102 ms
without, 76 ms for the blocks alone), so it is off unless asked for.

**The solver.** Ceres is the sole parameter-fitting and continuous-optimization engine. The C++
API (`helirec/calibration/solve.hpp`) and `helirec.solve.Problem` expose equations over named unknowns:
`unknowns("axis", n)`, `e = equations(measured, fit=...)`, `e.add("axis", tile, coefficient)`.
Duplicate coefficients add; held-out equations are predicted without affecting the fit, rank or cost.
Residuals are always measured minus predicted. `p.fix("axis", index, value)` chooses an explicit gauge.
Unused columns are frozen at zero (or their fixed values); dependent active columns cause an unavailable
fit, rather than an arbitrary choice of parameters. Python calibration steps reject unavailable fits.

Least-squares fits use Ceres Levenberg–Marquardt, dense QR up to 64 active parameters, and EigenSparse
normal Cholesky above that. Scalar parameter dependencies remain sparse. Linear designs are scaled to
unit column norm. Equations with the same dependencies share assembly; each scalar residual receives
its own Huber loss in residual units. The batched residual is `sign(r) sqrt(rho(r²))`, differentiated
including the loss; the returned residuals remain the original `y - prediction`. A Ceres least-squares
initialization precedes robust fitting. Eigen QR is used only to diagnose rank, with orthogonal row
compression preserving the design's singular values.

`solve(huber=0, max_iterations=100, function_tolerance=1e-7, gradient_tolerance=1e-12,
parameter_tolerance=1e-12, rcond=1e-12)` controls convergence. Fits run in double precision on one thread;
independent measurements may run concurrently. Obsolete IRLS `iterations` and `relative` controls have
been removed. Solutions include parameters, residuals, final square-root robust weights, robust spread
(1.4826 MAD, diagnostic only), cost (half the robust sum of squares), iteration/evaluation counts,
termination, a diagnostic message and Ceres version. `Solution.diagnostics` carries the metadata.
The low-level C++/Python results expose `usable`; `NO_CONVERGENCE` records an iteration limit even when
the current finite solution is usable. Geometry pass records retain these diagnostics.

Refill timing (`calibration/refill.hpp`, `_core.calibration_fit_refill_shots`) selects shots, initializes
both row-time signs, fits the exposure-integrated dip model with positive decay and automatic derivatives,
and uses Ceres covariance for row-time uncertainty. It requires 12 shots, at least five 20-row bands and
positive finite quiet-frame noise. Unidentifiable uncertainty is explicitly unavailable. Detector-blur
fitting also uses automatic derivatives, including spectral normalization and the eliminated nuisance
spectrum. Its binding returns `(field, rms_none, rms_fit, solver)`.
The bounded film-transfer and smear fits used by `evaluation/wiener_table.py` also live in C++
(`calibration/transfer.hpp`), with automatic derivatives and the same solver diagnostics.

Registration and block-spectrum alignment use Ceres BFGS with a Wolfe line search on negative normalized
correlation. Integer initialization, registration window recentering, shift signs and whole-pixel recut
requests retain their measurement conventions. A fixed curvature-based coordinate scaling conditions
BFGS; exact final curvature and stationarity validate the peak independently of solver termination.
Image buffers and correlation evaluations remain on the selected CPU/CUDA backend. CUDA returns only ten
scalars per correlation evaluation, using a fixed double-precision reduction tree; CPU/CUDA comparisons
retain their existing floating-point accuracy tolerances.

**Partial copies.** Frame files that are not on disk are reported (`map from frame to file`, with the frames
that are there) and their frames are never read: the master file's virtual dataset would return zeros for them.
Reading such a frame through `CalibrationSession` or `Frames` is an error; `stretches(scan)[i]["on_disk"]` lists
the ranges that can be read.

**Frames stored as JPEG 2000.** Projections written as one JPEG 2000 (HTJ2K) codestream per chunk of rows through
the blosc2 HDF5 filter (id 32026), as ESRF BM18 writes them, are read by `RawReader` when helirec is built with
OpenJPEG (see Build); only the chunks a request touches are decoded, in parallel (about 140 whole frames of
2368 × 4416 per second on 24 cores; a block of 56 frames × 300 rows in 0.1 s). A chunk is always decoded whole:
OpenJPEG 2.5.4 returns only the level shift for a decode area of an HTJ2K codestream. The container is parsed by
helirec itself, so neither the blosc2 library nor the HDF5 filter plugin is needed. Where each chunk lies in a
file is listed in one pass over the dataset's index (HDF5 1.14 and later; asked chunk by chunk before, which
was 2 to 3 s of a reconstruction's first slab over a hundred files). `RawReader::set_cache_bytes`
keeps decoded chunks between reads (off by default; the calibration's `Frames` keep 3 GB per scan,
`CalibrationSession.set_cache(gigabytes)`, `cache_counts()`): comparisons placed at the same frames, measured
in the order of their frames, then decode each chunk once (164 → 60 ms per comparison on Paris4, same results).
The session also decodes the frames of the comparisons that follow the one in hand ahead, on workers of the
reader's own (`RawReader::set_prefetch_threads`, `prefetch_rows`; `CalibrationSession.set_read_ahead(threads)`,
default a third of the machine's threads, 0 for none): 60 → 56 ms, since decoding already fills the cores. Such frames are usually compressed
irreversibly: the inventory reports the codec, the transform and the bits per value of each stretch
(`how the frames are stored`), the decoded values are OpenJPEG's (another decoder may differ by a count), and
their noise is not the detector's.

**At a half, a quarter, an eighth.** A JPEG 2000 codestream holds its image at lower resolutions too, and
`RawReader(files, dataset, rows, cols, reduce=1 | 2 | 3)` reads the frames there: (rows, cols) / 2^reduce values,
every row index of the reader in reduced rows, decoded 3.4, 12 and 44 times faster than the whole (a Paris4
chunk: 8.5, 2.5, 0.7, 0.2 ms) with nothing written to disk. A reduced value is not the mean of 2^reduce pixels:
value (j, k) is the wavelet's low band at the stored pixel (j, k) 2^reduce, each chunk of rows an image of its
own, so a geometry in stored pixels holds for the reduced frame with every position divided by 2^reduce and
nothing added. Darks, flats and weights are taken to the same grid by `helirec::calibration::at_resolution`
(`helirec.calibration.at_resolution(image, reduce, chunk_rows)` in Python) (the same low-pass; it reproduces a Paris4 frame read at reduce 1, 2, 3 to 0.3 counts rms,
the rounding), so that a flat divides the fixed pattern out as at full resolution. Plain (uncompressed) frames
have no such resolutions and are refused; rows, columns and rows per chunk must be multiples of 2^reduce.
`helirec.mosaic.tile_reconstructor(..., reduce=)` reconstructs a tile on it (below); the calibration session and
the pair pipeline read at full resolution only.

**Reduced reads of reconstructed float32 volumes.** `RawReader(..., dtype="float32", reduce=0 | 1 | 2 | 3)`
reads ESRF's B2JPF32 HTJ2K chunks, including the signed 32-bit codestreams in Paris4. This uses
[OpenJPH](https://github.com/aous72/OpenJPH), fetched and compiled automatically by CMake; neither Kakadu nor
an HDF5 filter plugin is needed. The C++ API has the same constructor options and `float*` overloads for
`read_rows` and `read_frames`. Decoding and I/O run on the CPU, in parallel over chunks.

```python
import numpy as np
from helirec import RawReader

reader = RawReader(
    ["tile.hdf5"], "entry0000/reconstruction/results/data",
    rows=8223, cols=8223, dtype="float32", reduce=3,
)
slices = reader.read_frames(np.array([0], np.int32), np.array([0], np.int32))
# slices.shape == (1, 1028, 1028); slices.dtype == np.float32
print(reader.bytes_read)
```

Odd frame sizes round up. Reduction acts on X/Y within each slice; choose frame indices separately for Z.
With `selective=True` (default), a seekable stream skips unused resolution packets instead of loading the
whole compressed chunk. On a Paris4 8223 × 8223 slice, full/half/quarter/eighth reads requested
18.03/8.52/3.47/1.57 MB. Savings depend on packet order and chunk layout. `bytes_read` counts payload bytes
requested by the reader, excluding HDF5 metadata; a mounted filesystem such as rclone may fetch more through
read-ahead. Row windows still decode the touched chunks at the requested resolution. Internal chunk boundaries
must be multiples of 2^reduce; whole-frame chunks can have odd dimensions.

`selective=False` preloads the full codestream for the same OpenJPH decode, useful for comparison or storage
that favors sequential I/O. The selective and preloaded results are identical. Irreversible JPEG 2000 results
can differ slightly between decoder implementations. The default uint16 projection path uses OpenJPEG and
loads each touched compressed chunk in full. Set `decoder="openjph"` to enable selective HTJ2K reads of uint16
projections too; this chooses OpenJPH's decoded values explicitly.

**Download the Paris4 projection slab at 1/8 resolution.**
`python -m helirec.download_projections` (or `helirec-download-projections` after installation) exports all
19 tiles to `/mnt/raid_nvme/paris4_mosaic_recon/projections/scale_0.125`. The defaults select a 256-original-slice
slab (0.289024 mm, 32 slices on an isotropic 1/8 grid), centred at motor Z = −256.0385 mm, matching the existing
local ROI. Each tile keeps whole source files covering the slab plus 1.5 helix turns of context on either side
(at least one detector height), and every angle in that interval. Frames become 296 × 552; angles are not subsampled.

```bash
python -m helirec.download_projections --plan-only  # inspect selection and output size
python -m helirec.download_projections              # download; repeat this command to resume
tail -f /mnt/raid_nvme/paris4_mosaic_recon/projections/scale_0.125/download.log
```

The source is the mounted `keep_it_for_replaying_20260311/RAW_DATA` directory; files already in the local
`paris4_mosaic_recon/RAW_DATA` are preferred. `--source-root`, `--local-root`, `--output`, `--tiles`, `--slices`,
and `--center-mm` can override these defaults. `--workers` controls concurrent files (default 4), `--threads`
the decoder threads per file (default 2). `--max-files 1` runs one pending file as a smoke test. No SDK or HDF5
codec plugin is required; build helirec with `HELIREC_OPENJPH=ON` (the default).

Projection stacks are uncompressed uint16 HDF5. Darks and flats are read at full resolution, reduced with
`calibration.at_resolution` on the same chunk boundaries, and saved uncompressed as float32 to preserve the
fractional low-pass values. Their sources have no stored pyramid, so their full pixels must be read.
The output payload is approximately 209 GB; the plan reports the exact size before starting.

`download.log` records starts, completions, retries, and progress every five seconds: file counts, bytes written,
output rate, elapsed time, ETA, and active-file frame counts. `progress.json` holds the latest completed totals.
Completed files have metadata and a SHA-256 of their pixels; atomic publication prevents partial files from
being mistaken for completed ones. Resume validates job identity, shape, dtype, and absence of compression,
without rereading all pixel checksums. Incompatible existing files or plans are refused. Four workers keep
memory bounded by their batches (`--batch-frames`, default 8; calibration reads one full frame at a time).

`plan.json` and each output file preserve original frame indices, angles, heights, and timestamps. Each completed
tile receives `index.h5`, a relative-path virtual dataset per acquisition entry (`5.1/data` for projections,
`2.1/data`, `3.1/data`, `7.1/data` for calibration), with matching metadata arrays. Original acquisition masters
are copied into `metadata/` for reference; their original virtual datasets are not reduced-data entrypoints.
The selection planner and uncompressed writer are also public C++ APIs in `helirec/projection_subset.hpp`,
exposed through the Python bindings. Projection I/O skips unused HTJ2K packets; filesystem read-ahead can
still transfer additional bytes from the mount.

**Mosaics.** Scans that differ in where the sample sits on the rotation stage (the tomo configuration's
`sample_u`, `sample_v`) and not in the lateral stage of the axis are checked as tiles, any number of them:
which motors differ, the place of each tile and its agreement with the laboratory pseudo-motors, which lines
each pair records at equal angle and in opposite views and how often the lines of the merged acquisition are
recorded, and the angles, heights and times of the tiles at equal projection index. The lever arm of the angle
targets is then the farthest a tile's rays pass from its own axis inside the mosaic (`--radius PX` overrides
it), not the detector's width. The `views` checks do not cover a mosaic; its comparisons are placed through the
session. A sample displaced by `d` on the rotation stage is seen at angle θ and column `c` along the sample's
line `s = c − axis − n(θ)·d`, so two tiles record the same lines at equal angle on columns `n(θ)·(d_j − d_k)`
apart: a sinusoid once per turn. A request to `measure` states it as `sway_px`, `sway_phase_deg` (tile B's
frames are read `sway_px cos(angle − sway_phase_deg)` columns further on, in B's own columns, resampled per
frame by cubic convolution) and the registration returns what is left of it. With `mirror` and
`angle_offset_deg=180` the same request compares one tile with the opposite views of another; `sway_row_deg`
takes each row of B at its own angle (rows exposed one after another: between two tiles that is a shear of the
tiles' distance times the angle per row).

**A mosaic's geometry is not calibrated at present.** The geometry step that was in `helirec.mosaic` (its fits
and its check of two tiles' overlap in the volume) was removed in November 2026 and is to be done anew. What is
left there reconstructs one tile alone from a description of the tiles' geometry (the module's docstring gives
its form; nothing in helirec makes one): `frame_pose(cal, tile, angle, frame)` gives a frame's angle, column and
height, and `tile_reconstructor(master, cal, tile)` a `SliceReconstructor` of that tile; with `cache_dir=` the
tile's reduced darks and flats are kept there (`calibration.reduced_darks_flats`: 8 s of a tile's start
otherwise, 0.5 s read back), and with `reduce=1 | 2 | 3` it reconstructs at a half, a quarter, an eighth (the
frames read at the codestream's lower resolution, the geometry divided by 2^reduce; a reduced reconstruction is
the full one's low band, pixel for pixel from the axis, to under 0.01 of a reduced pixel in the slice). The
reconstruction of a mosaic as one volume does not exist (see Limitations).

For measurement-support diagnostics, `--columns N --column-fraction F` selects a narrower
window within each overlap (`N=0` keeps the full overlap; `F=0`, `0.5`, `1` selects left,
centre, right in the first image). Python uses `columns` and `column_fraction`; C++ uses
the corresponding `ViewOptions` fields. Mirrored comparisons crop corresponding rays,
and direction probes retain the full overlap. `views.tsv` includes the actual `columns`
count, which can be rounded down to a suitable FFT size. These options change only the
measurement support; they apply no geometric correction. Narrow windows may lack sample
texture, so compare accepted matches and retain rejected attempts.

## Geometry calibration step (`helirec-geometry`)

Measures the geometry of a two-scan helical acquisition from its own frames, with nothing typed in: the detector
map and the poses that a reconstruction needs before it can assign a raw value to the line it measured. It is a
step that writes what it measured (JSON), and it is the reconstruction's only geometry calibration:
`helirec-reconstruct` makes it on first use and keeps it in its calibration cache, or takes the file it is given
(`--set geometry=...`, below). The darks, flats, weights and the detector's blur field are
`python/helirec/calibration.py`'s.

```bash
helirec-geometry <main.h5> <companion.h5> --out geometry.json            # main: the scan with the axis inside the field
helirec-geometry <main.h5> <companion.h5> --out geometry.json --row-time-us -19.87   # if no stretch holds a refill
helirec-geometry <main.h5> <companion.h5> --out geometry.json --patch 128 --passes 1  # the map from patches, measured once
```

```python
from helirec import geometry
cal = geometry.calibrate_geometry([main_h5, companion_h5])
geometry.save(cal, "geometry.json")
```

What it needs of the acquisition: a helix whose travel per turn is less than the detector's height, a turn at rest
before and after it, the axis inside the field in the first scan, a second scan moved sideways, and, for the rows'
timing, a refill of the ring during one of the stretches (otherwise `--row-time-us`, which belongs to the camera's
configuration and not to the scan).

What it does (`python/helirec/geometry.py`):

1. **Layout**, from the record and single frames: frames per turn, the stretches, the sense of the rows against
   the axis stage, a coarse axis column and lateral move (profiles correlated), and the rows that hold something
   across the whole width (the mount), which are kept out of every comparison.
2. **Rows' timing**: the time of a detector row within the frame, from the beam dips of a refill.
3. **Map**: pixel (c, v) looks along the ray at (c + Eu, v + Ev), polynomials of order 3. Views that hold the same
   lines on different parts of the detector (opposite views, a rest turn against the helix, the helix a turn
   apart, one scan against the other) are registered on the current map's grid, what they still disagree by is
   fitted as a correction, and that is repeated until nothing is left (0.02 px).
4. **Poses**, from the same fit: the axis column per rest turn, the second scan's place against the first per
   stretch with its own line in the column, the heights of the rest turns against the helix, the axial scale.
5. **Motion within a turn**, on the finished map's grid (about 1400 full-width registrations): the rotating body's
   place against the rotation axis as a function of time (a point every half turn in the first scan, every turn
   in the second), the sample's height once per turn and its drift at rest, and how far the helix is further on
   in angle than the rows' timing gives.

`geometry.json` holds the layout, the row time, the map (`i`, `j`, `eu`, `ev` with its normalisation), the poses
and what every pass left per kind of comparison. On PHerc0343P (two scans of 49,202 frames, RTX 5090) it takes
10 minutes: 20 s for the layout and the rows' timing, five passes for the map (44 s for the first, on the frames
as stored, 72 s each for the others) and 4 minutes for the motion; `development_notes/calibration/05_experiments.md`, experiment 31, sets its result
beside the one found by hand and tests it on slices reconstructed twice from disjoint frames.

**Patches (`--patch 128`, `calibrate_geometry(..., patch=128)`).** The passes exist because a block of 1024
columns is registered as one rigid displacement while the map stretches it by up to a pixel: the blocks have to
be cut on a grid that already holds the map. A patch of 128 x 128 is stretched by a tenth of that, so the plan's
blocks cut into patches (`CalibrationSession.measure_patches`: the frames of a block read once) give the map
from the frames as stored. On PHerc0343P the first pass (20 256 patches, 15 008 accepted) takes 99 s with its
fit, and the step with `--passes 1` 6 minutes in all, the motion as before. Slices reconstructed twice from
disjoint frames agree as well with that calibration as with the passes' (within a scan 0.073, 0.056, 0.065 px
rms at three heights against 0.072, 0.058, 0.070; between the scans 0.099, 0.145, 0.102, 0.117 against 0.105,
0.143, 0.102, 0.119; the test's floor is 0.013 to 0.020). Left to run to the tolerance, the patches measured
again on the map's grid correct it by up to 0.066 px and then 0.007 (three passes), which that test does not
see. The default is the blocks (`--patch 0`), whose result is unchanged. The step uses
`helirec::calibration`'s session (raw frames, the scan's own dark and mean flat, the registration on CPU or CUDA)
and none of the pipeline's calibration.

**In the reconstruction:** `helirec-reconstruct ...` with nothing said (the cache's `geometry.json`, made if it is not
there), or `--set geometry=OUT/geometry.json` (`params["geometry"]`). Every
projection is read on the grid of the calibration's detector map (the rows at the mapped row by the row
kernel, the columns resampled exactly: `resample_rows`, a non-uniform transform of the interpolant the exact shift
uses, within 1e-6 of the direct sum), at the pose `geometry.frame_poses` gives its frame, and at the angle of its
rows. The frame the two scans are merged in is laid out from the same poses (`geometry.merged_layout`: the medians
over the helix of the axis' column and of the second scan's place, on a quarter-pixel grid; PHerc0343P: axis
1684.75 in the first scan's columns, the second scan 843.5 columns and 1.0 rows from it, the axis at 2528.25 of
3372 merged columns, as the earlier searches had it but for the rows, which they put at 0.5): a nominal frame,
every projection's own axis and row being in its terms. It needs `x_interp=sinc`, and `fbp` with
`angle_interp=views` or `nearest` for the row timing. `provenance()["geometry"]` records the file, the merged
frame and what the calibration measured that the reconstruction cannot express. `mode=legacy_reference` takes the
merged frame alone. On PHerc0343P, without projection filters it costs
no time (7.5 s per 32 slices against 7.3–8.4 s); with Paganin and unsharp a slab of 256 slices takes 53 s against
32–33 s. What the reconstruction still does not do exactly (the beam's divergence, the six-row kernel along the
rows; views a turn apart are backprojected each at its own angle since `angle_interp=views`, section 34) is listed
in `development_notes/calibration/05_experiments.md`, section 33.

## Tests

`pytest` runs the unit tests: CPU vs CUDA vs NumPy references for every kernel, projector adjointness, the
whole-slice methods, and regression checks of the corrected conventions (`tests/test_regressions.py`).
`tests/test_calibration.py` covers `helirec::calibration` on synthetic blocks and a synthetic acquisition record.
`tests/test_geometry.py` covers the geometry step's fit (a known map, roll, rotation and lines recovered from
comparisons made up from them) and its plan of comparisons; the step itself needs an acquisition (experiment 31).
`tests/test_detector_map.py` covers a scan's detector map in the gridding (`ScanTrajectory::map_*`, `col_scale`,
`row_scale`; `detector_map.hpp`): `resample_rows` (`sino.hpp`), which reads a row at the detector column of every
ideal column without loss, against the direct sum of the row's trigonometric interpolant (within 1e-6 of the row's
amplitude, content at Nyquist included; CPU and CUDA), the inversion of the map and its sign conventions, and scans
gridded through a map or a scale against the same scene on the ideal grid. `_core.time_rows(what, n_rows, n,
repeats, device, ...)` times `shift_rows` (`what="shift"`) and `resample_rows` (`"positions"`, or `"map"` with the
map's arguments) on rows of noise without the transfers, and returns the seconds of each call (the first includes
the transform plans).

## Build

Requirements: CMake ≥ 3.24, a C++20 compiler, **Ceres ≥ 2.2 with EigenSparse**, FFTW3 (float + double), HDF5 (C), zlib, libcurl, Python ≥ 3.10, and
optionally the CUDA toolkit and OpenJPEG ≥ 2.5 (`apt install libopenjp2-7-dev`, `brew install openjpeg`; found
through pkg-config, `-DHELIREC_OPENJPEG=OFF` to build without). OpenJPEG is needed only for frames stored as
uint16 JPEG 2000 (see `helirec-calibration`); without it such scans are refused with a message that names the package.
For float32 HTJ2K, CMake FetchContent downloads OpenJPH 0.32.0 with a SHA-256 check and links it statically.
Its BSD-2-Clause license is included with installed artifacts. Use `-DHELIREC_OPENJPH=OFF` to omit it, or
`-DFETCHCONTENT_SOURCE_DIR_OPENJPH=<src>` for an offline build with existing sources. The vc-delta3d codec is fetched at configure time (offline:
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
* an acquisition the geometry calibration step can read: a helix whose travel per turn is less than the
  detector's height, a turn at rest before and after it, two scans of equally long stretches;
* a mosaic (tiles that differ in the sample's place on the rotation stage) is checked by `helirec-calibration`
  and nothing calibrates its geometry; `helirec-geometry` and `helirec-reconstruct` do not take one, and a tile can
  only be reconstructed alone, from a geometry given to it (`helirec.mosaic.tile_reconstructor`);
* parameter sweeps (`reconstruct_band(..., sweep=...)`) run on CUDA only;
* xz slices are not implemented.

The intensity calibration's kernels are CPU-only (they are cheap); the geometry calibration step registers on
the CPU or on CUDA. The reconstruction is parallel-beam and slice by slice: what depends on the depth along the
ray (the beam's divergence, the axis' tilt) is measured by the geometry calibration and not applied
(`provenance()["geometry"]["measured_not_applied"]`).
