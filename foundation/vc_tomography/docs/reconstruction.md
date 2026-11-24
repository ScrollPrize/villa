# Reconstructing a scan

A reconstruction needs three things:

1. the scans' raw frames and acquisition records;
2. a geometry calibration (`geometry.json`): where every frame of every scan sees the sample;
3. an intensity calibration: darks, flats and weight maps.

`vc_tomography_reconstruct` makes both calibrations on first use and keeps them in its cache directory, so for a
standard acquisition one command is enough. [Calibration and alignment](calibration.md) describes how to make and
check them yourself.

## Input layout

`vc_tomography_reconstruct` takes one folder, `PROJ`, that holds all scans of one acquisition.

**ESRF BM18 (BLISS).** `PROJ` holds one folder per scan. Each scan folder holds the dataset folder
`<name>_0001/` with the master file `<name>_0001.h5` and its frame files. The scans of one acquisition are found
by name: `<base>_HA`, `<base>_TA`, `<base>_TA2`, and so on. The `HA` scan, which has the rotation axis inside its
field of view, comes first. A column recorded under another sample name (for example `..._C_TA3` beside
`..._B_HA`, `..._B_TA`, `..._B_TA2`) joins the family it fits.

**Diamond I12 (NeXus NXtomo).** `PROJ` holds the `<n>.nxs` masters and their frame files, plus a file
`vc_tomography_acquisition.json` that supplies what the masters do not record:

```json
{"pixel_size_um": 7.91, "energy_kev": 53, "propagation_distance_m": null,
 "source_sample_distance_m": null, "fan_sense": -1, "height_stage": "ss1_y1"}
```

Only `pixel_size_um` is required. `source_sample_distance_m` lets the calibration measure the beam's divergence.
`height_stage` names the motor that sets the sample's height (default `<stage>_y1`), or a list of stacked motors
whose positions add up to it (`["ss1_y1", "ss1_y3"]` for PHerc0051.04.08, where each moved between two of the scans).
`propagation_distance_m` is needed for Paganin and the other phase filters.
`fan_sense` is the sign of the divergence relative to the recorded angle (`-1` at I12, `1` at BM18; default `1`).

## Quick start

```bash
# five single slices spread over the scan's height
vc_tomography_reconstruct PROJ --tag first_look

# slices 900 to 1700 as an OME-Zarr volume, with an 8-bit copy and a 4-level pyramid
vc_tomography_reconstruct PROJ --z 900 1700 --uint8 --levels 4 --tag block

# the same with Paganin phase retrieval and an unsharp mask
vc_tomography_reconstruct PROJ --z 900 1700 --uint8 --levels 4 --tag block_paganin \
    --set "filters=paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)"
```

The first run on an acquisition also runs both calibrations. On PHerc0343P (two scans of 48 000 frames each,
RTX 5090) the geometry calibration takes about 2 minutes 20 seconds with the frames in the page cache and 5 minutes
without.

## Outputs

Every run writes into `OUT_ROOT/TAG/`. `--out_root` defaults to `./vc_tomography_slices`, `--tag` to the current
time. An existing folder is an error unless `--overwrite` is given.

- `manifest.json`: every resolved parameter, the calibration files used, and per-slice timing.
- `config.json`: the run's options. `vc_tomography_reconstruct --config config.json` repeats the run; options on
  the command line override the file.

The calibration cache defaults to `OUT_ROOT/_calibration`. Pass the same `--cache_dir` to every run on one
acquisition so the calibrations are made once.

Values are linear attenuation in cm⁻¹.

## Choosing slices

There are two modes.

**Single slices** (written as TIFF files, one per slice). Positions are fractions of the reconstructable height,
from 0 (top) to 1 (bottom).

| Option | Meaning |
|---|---|
| `--fractions F ...` | the slices at these fractions |
| `-n N` | `N` slices evenly spaced over `--range` (default 5) |
| `--range R0 R1` | the part of the height `-n` covers (default `1/6 5/6`) |
| `--previews` | also write PNG previews and a contact sheet |

**A range of slices** (written in chunks; OME-Zarr by default).

| Option | Meaning |
|---|---|
| `--z Z0 Z1` | slices `Z0` to `Z1 − 1`, numbered from 0 at the top of the reconstructable height |
| `--heights H0 H1` | the same range given as detector view heights |
| `--chunk N` | slices reconstructed at once (default 32). A larger chunk is faster per slice, because every chunk also grids the context rows its filters need above and below it (43 each with Paganin, unsharp and two scans), and takes more GPU memory; lower it if GPU memory runs out. |
| `--format zarr\|tiff` | OME-Zarr (default) or one TIFF stack per output |
| `--levels N` | pyramid levels in the OME-Zarr (default 1) |
| `--zarr_chunks CZ CY CX` | OME-Zarr chunk shape (default 128³) |

Both modes take:

| Option | Meaning |
|---|---|
| `--roi X0 X1 Y0 Y1` | crop each slice. Pixels are identical to the same pixels of an uncropped run. |
| `--bbox X0 X1 Y0 Y1` | compute only this box; the rest of the slice is written as 0 at full size |
| `--uint8` | also write an 8-bit copy, mapped from the window `--u8_window LO HI` (default `-0.06 0.29` cm⁻¹) |
| `--uint8_only` | write only the 8-bit copy |
| `--device cuda\|cpu` | the compute device (default `cuda`) |

8-bit volumes are compressed losslessly with the vc-delta3d codec (`--u8_compression`, `--u8_delta3d_preset`).
Chunks that hold only fill values are not written.

### Masking with an existing volume

`--reference_volume ZARR` takes the mask of an existing masked uint8 volume of the same scan (a local path,
`http(s)://` or a public `s3://` location). The uint8 output is 0 wherever the reference is 0, and only the
pixels the mask keeps are computed, which makes a masked run much faster. `--reference_scale L` reads the mask
from pyramid level `L` instead of level 0 (an 8× level is 1/512 of the data). `--reference_cache DIR` keeps the
chunks read from a remote volume. The reference may have up to 2 % more or fewer slices than the
reconstruction (as when it was made with the recorded rather than the measured helical pitch); its slices are
then spread evenly over the reconstruction's.

## Reconstruction parameters

Parameters are set with `--set KEY=VALUE` (repeatable) on the command line, `params={...}` in Python, or the
`set` list of a config file. The defaults are the recommended settings; most runs change only `filters`.

### Method

`--set backprojector=SPEC`:

| Spec | Method |
|---|---|
| `fbp(interp=bspline3, oversample=2)` | filtered backprojection with an exact, pixel-driven backprojector (default). `interp`: `linear`, `cubic`, `lanczos3`, `bspline3`, `bspline5`. `oversample`: 1 to 8. |
| `sirt(num_iter=…, init=fbp\|gridrec\|constant, …)` | SIRT |
| `mlem(num_iter=…, init=…, …)` | MLEM |
| `gridrec(filter_name=…, filter_par=a:b)` | gridrec (Fourier gridding) |
| `ghbp` | the hierarchical backprojector of the ESRF reference workflow |

`fbp` and `ghbp` take a ramp window as `filter_name` (`ramlak` by default, `hann`, `hamming`, `parzen`,
`butterworth`, ...) with `filter_par=cutoff:order`. SIRT and MLEM work on a rebinned 2D model of each slice;
gridrec is scaled to match FBP. The CUDA gridrec is not bit-reproducible between runs; the CPU gridrec is.

### Projection filters

`--set filters=CHAIN` sets the filters applied to each projection before reconstruction, joined with `+`. The
default is no filter.

| Filter | Purpose |
|---|---|
| `paganin(delta_beta=…)` | Paganin phase retrieval |
| `unsharp(sigma=…, coeff=…)` | unsharp mask |
| `mtf_paganin(delta_beta=…, sigma=…)` | phase retrieval and detector deblurring in one filter (in place of `paganin + unsharp`); `target` and `gain`, or `nsr`, set the trade between sharpness and noise |
| `wiener(table=FILE)` | Wiener filter from a measured transfer and signal-to-noise table (`evaluation/wiener_table.py` makes the table) |
| `epr(delta_beta=…)` | eikonal phase retrieval: Paganin plus second-order terms, for strong phase effects |
| `butterworth(cutoff=…, order=…)` | low-pass in both detector directions |
| `gaussian`, `median` | smoothing, median filter |

Each filter runs before the logarithm (`stage=intensity`, default) or after it (`stage=log`). The reference
workflow's chain is `paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)`. A single filter parameter can be
changed with `--set paganin.delta_beta=500`.

### Other parameters

| Key | Default | Meaning |
|---|---|---|
| `geometry` | `geometry.json` in the cache | the geometry calibration file to use |
| `row_time_us` | measured | the camera's row time, used only when the geometry calibration is made |
| `sld` | none | correction for the scintillator's long-range glare, e.g. `fraction=0.43,length=170,outside=lit`, or `json=FILE` for a BM18 calibration file. The values must be measured for the detector (`evaluation/calibrate_sld.py`). |
| `beam_divergence` | `true` | FBP follows the beam's divergence within the slice, when the calibration knows the source distance |
| `sharpness_weighting` | `auto` | weight the recordings of a ray by the detector's local sharpness at high frequencies. Uses about 5.6 GB more GPU memory per 32-slice chunk; `off` disables it. |
| `angle_interp` | `views` | `views`: every projection backprojected at its own measured angle. `nearest`: projections binned to one turn. |
| `z_interp` | `lanczos8` | interpolation between detector rows (`lanczos8`, `lanczos3`, `cubic`, `linear`) |
| `x_interp` | `sinc` | resampling of a scan at a fractional column offset (`sinc` is exact) |
| `exposure_arc` | `0` | `auto` deconvolves the blur from rotation during each exposure. Requires `angle_interp=nearest`. |
| `luminosity_repartition` | `100` | match the brightness of neighbouring scans at their seam; `0` disables it |
| `only_scan` | all scans | reconstruct one scan alone (`main`, `companion`, `companion2`, or `HA`, `TA`, `TA2`, ...) |
| `angle_subset` | all | `I/N`: the `I`-th of `N` interleaved subsets of the angles. `0/2` and `1/2` give two reconstructions with independent noise, for resolution measurements. |
| `noise_split` | none | add noise to the raw frames to make denoiser training pairs (below) |
| `mode` | `corrected` | `legacy_reference` reproduces the ESRF Night Rail / nabuxx workflow exactly, with all parameters fixed and values in its units (half of cm⁻¹) |

### Noise splitting

`noise_split=scale=S,seed=N,gain=G,floor=F` adds `S` times each raw sample's own noise to it. The noise's
standard deviation is `sqrt(G × (counts above the dark) + F)`, and the random numbers are fixed by the seed, the
file, the projection and the pixel, so they are the same in every run, chunk and device. Two runs with the same
seed and the scales `a` and `-1/a` have uncorrelated noise, and together use every projection (Pang et al. 2021).
The pairs train the denoiser in `networks/denoising/`.

- `G` and `F` belong to the detector and must be measured from a flat's variance against its level. Wrong
  values leave the two runs' noise partly correlated.
- `smooth=SHARE:SPACING[+SHARE:SPACING]` adds noise correlated over `SPACING` pixels, for a detector whose noise is
  not white. The total variance stays the one `G` and `F` give.
- `flats=N` also adds independent noise to each flat, which is the mean of `N` frames. Without it the two runs
  share the flats' noise, which shows as rings.

Not available for reduced frames or in `legacy_reference` mode.

## Notes by scan type

**BM18 helical pair or multi-column scan.** No extra options are needed. The brightness of each scan after the
first is matched to its neighbour at the seam. The calibration needs the camera's row time; it is measured from
a refill of the storage ring if one happened during a scan. If not, give it with
`--set row_time_us=US` (see [Calibration and alignment](calibration.md#row-time)).

**Diamond I12 grid of circular scans.** Make the geometry calibration first with the options in
[Calibration and alignment](calibration.md#diamond-i12-grid-of-circular-scans) and pass it with `--set geometry=FILE`. Use
`--set luminosity_repartition=0` when dense material (such as a case's base plate) crosses the seams. Where two
rows of scans cover a height, both contribute by their weights.

```bash
vc_tomography_reconstruct PROJ --heights 1200 1210 --format tiff --set geometry=geometry.json \
    --set luminosity_repartition=0 --set "filters=paganin(delta_beta=175)+unsharp(sigma=1.2,coeff=4)"
```

**BM18 mosaic.** A mosaic is reconstructed in one of two ways.

*Each tile alone.* `vc_tomography.mosaic.tile_reconstructor(master, cal, name)` reconstructs one tile from Python,
using a geometry calibration made for the tiles (see [Calibration and alignment](calibration.md#bm18-mosaic-tiles)).
A tile gives a disc around its own rotation axis. Its rows end inside the sample, so its slice carries a slow
shading, and neighbouring tiles differ in level where they overlap.

*All tiles as one slice (whole-canvas FBP).* `vc_tomography_reconstruct` takes a mosaic when the geometry
calibration given with `--set geometry=FILE` is of a mosaic's tiles. It reads every tile's place and height
against the first tile (`parameters.scans[k].d`, `.height`), which the geometry calibration measures:

```bash
vc_tomography_reconstruct RAW_DATA --set geometry=geometry.json --fractions 0.5 --tag canvas
vc_tomography_reconstruct RAW_DATA --set geometry=geometry.json --z 400 432 --chunk 4 --roi 3000 5000 3000 5000 \
    --projection_store projections.zarr --reduce 2 --keep_tiles --tag canvas_block
```

Every tile contributes to full-width projections in the same calibrated coordinates. By default, the canvas
uses ordinary projection weights for coarse structure and gives more weight to views with shorter exposure
travel for fine detail. The blend happens in the ramp-filtered detector rows before backprojection.

- Slices are numbered from 0 (`--fractions`, `-n`, `--z` and `--heights` all take these numbers) and are written
  as TIFF, `slice_<number>.tiff`, with `manifest.json`. `--roi` crops; `--uint8` adds an 8-bit copy.
- The canvas is the slice of the whole mosaic, with the first tile's rotation axis at its middle pixel. Pixels
  outside every tile's disc are 0.
- `--keep_tiles` keeps every tile's reconstructor between chunks. It avoids setting each tile up again and needs
  device memory for all of them at once: use it for reduced frames.
- `--set tile_taper=N` sets the stored columns at either end of a tile's rows over which its weight falls to
  nothing (default 200). `--set tile_levels=false` turns off the level solved per tile and slice from the rays two
  tiles share.
- `filters`, `z_interp`, `x_interp` and `backprojector=fbp(...)` are taken as for any scan. The filters act on
  each tile's frames, before the rows are joined. Only `fbp` is available.
- Whole-canvas FBP sets each tile's `angle_interp` to `nearest`, including when the input requests `views`.
  Repeated turns contribute to one row per angular bin. The `tile_rebin` setting controls how the canvas
  combines those rows across tiles. Standalone FBP with `angle_interp=views` keeps the turns separate.
- The default is `--set tile_rebin=exposure` (Python `rebin_angles="exposure"`). Each region is centred
  on a tile axis and uses projections from all tiles on the measured-angle grid. For a feature at the
  centre of that region, let `L` be the predicted detector travel in pixels during one recorded exposure.
  Multiply each view's support by `0.01 + 0.99 / (1 + (L / 2)^4)` for the fine band. The transition from
  ordinary to travel-weighted projections runs from 0.04 to 0.10 cycles per reconstruction pixel.
  Regions crossfade over the inner 600 stored pixels at the edge of each coverage disc, capped at its radius.
  The ordinary FBP supplies the balance wherever the tapers sum to less than one.
- Exposure weighting requires a continuous-rotation gate for every tile and at least two tiles. Otherwise
  the default uses ordinary measured-angle FBP. The manifest records the resolved `exposure_weighted` flag
  and each tile's `exposure_gate_rad`. Sinogram debug arrays describe the ordinary full-field pass.
  The exposure mode caches aligned tile rows for reuse and limits the number of heights processed together
  on large canvases. A cropped reconstruction evaluates only the regions intersecting the crop.
- `--set tile_rebin=false` blends the tiles directly, using the weighted mean angle of each folded bin. Use `--set tile_rebin=true` to
  interpolate onto a common parallel-ray angle grid. The first tile sets the grid phase for each slice.
  With `beam_divergence` enabled and a calibrated fan, the canvas first interpolates each detector
  column to its parallel-ray direction using the single-tile FBP's small-fan model. It then places each measured row at its own angle and interpolates
  the weighted line integrals and weights at **fixed canvas column** before folding. Linear angular
  interpolation can attenuate fine detail when adjacent views are far apart. In Python pass
  `rebin_angles=True` to `mosaic.canvas_reconstruction`.
- Use `--set tile_rebin=lanczos3` for a single six-tap angular interpolation, or
  `--set tile_rebin=linear` for a single linear interpolation. Both keep the native Fourier
  detector shift. At a fixed canvas column they account for the fan-dependent detector offset
  and interpolate between the physical directions of the bracketing views. Lanczos weights
  use this fractional view index and fall back to linear at gaps or weak coverage.
  The Python values are `rebin_angles="lanczos3"` and `rebin_angles="linear"`.
- `--set tile_rebin=measured` (Python `rebin_angles="measured"`) keeps both halves of the
  turn through FBP. A geometry pass computes the mean measured direction of each view across
  tiles at the slice height. Same-side tiles align there. At each detector position the stronger side keeps
  its measurement. The weaker or missing side borrows from the mirrored detector position
  interpolated at the physical opposite direction. Each completed
  projection is filtered once and backprojected with its angular quadrature weight.
  Debug output includes all view angles, one filtered row per angle, and the fraction of
  supported rays requiring opposite-side coverage (`opposite_only_fraction`).
- `manifest.json` records the mode and the largest input row-centre angle offset from its output bin
  (`angle_spread_bins`). All modes reconstruct horizontal slices. The calibrated rotation-axis pitch
  shifts each tile's detector rows to the common sample origin at each view. The variation in ray
  height with depth within the sample and the relative tile rotations remain outside this model.
- The places come from `vc_tomography_calibration geometry` run on all the tiles (see
  [Calibration and alignment](calibration.md#bm18-mosaic-tiles)). A file in which every place is 0 is refused.

From Python: `vc_tomography.mosaic.canvas_reconstruction(projections, cal, ...)` returns an object with `doable`,
`size` and `reconstruct(h_lo, h_hi, roi=None)`. From C++: `vc_tomography::MosaicReconstruction`
(`include/vc_tomography/mosaic.hpp`). On the 19 tiles of PHercParis4 at a quarter (RTX 5090) the setup takes 18
seconds and a slice of 9005 × 9005 pixels 7 seconds.

## Reconstructing at reduced resolution

Frames stored as JPEG 2000 can be read at 1/2, 1/4 or 1/8 of the detector's resolution. Reduced reconstructions
read their frames from a projection store: a Zarr v3 copy of the frames, darks and flats at several resolutions.

**1. Download the frames into a store.** The downloader fetches the frames that a slab of the scan needs, plus
1.5 turns of context on either side (`--context-turns`).

```bash
vc_tomography_download_projections \
    --scans 2.4um_PHerc-Paris4_B_HA 2.4um_PHerc-Paris4_B_TA 2.4um_PHerc-Paris4_B_TA2 2.4um_PHerc-Paris4_C_TA3 \
    --slices 128 --center-mm -256.0385 --reduce 1 2 \
    --source-root RAW_DATA --output download --zarr projections.zarr
```

- `--scans` names scan folders under `--source-root`; `--tiles` with `--tile-folder` names mosaic tiles.
- `--slices` and `--center-mm` choose the slab, in full-resolution slices and recorded motor height.
- `--reduce` lists the levels to store: 1, 2 or 3 for a half, a quarter or an eighth. More than one level
  requires `--zarr`.
- `--remote NAME:PATH` reads files that are not local over SFTP from an rclone remote, fetching only the bytes the
  reduced decode needs. Needs paramiko (the `remote` extra: `uv sync --extra remote`).
- `--plan-only` prints the selection and its size without downloading. Running the same command again resumes.

`vc_tomography_projections_zarr STORE.zarr DIR [DIR ...]` builds a store from earlier downloads made without
`--zarr`.

**2. Calibrate and reconstruct from the store.** `PROJ` then needs only the masters (the downloader copies them
into `download/metadata/`).

```bash
vc_tomography_calibration geometry META/*/*_0001.h5 --out geometry.json \
    --reduce 2 --projections projections.zarr --row-time-us -6.46
vc_tomography_reconstruct META --projection_store projections.zarr --reduce 2 \
    --set geometry=geometry.json --z 500 532 --chunk 4 --uint8 --tag quarter
```

The slices have pixels 2^R times the stored size. Only heights covered by the stored frames can be
reconstructed. `mode=legacy_reference` and `sharpness_weighting=auto` are not available from a store.

## Surface composites

`vc_tomography_composite` renders max and mean composites of layers along a tifxyz surface.

```bash
# from a reconstructed volume
vc_tomography_composite VOLUME.zarr SURFACE.tifxyz --layers 15 --start -4 --direction auto --xy_slices 8 --tag s

# straight from the projections, computing only the voxels the layers need
vc_tomography_composite --projections PROJ --cache_dir CAL SURFACE.tifxyz --z Z0 Z1 --layers 15 --start -4 \
    --set "filters=paganin(delta_beta=1000)+unsharp(sigma=1.2,coeff=4)" --tag s
```

Layer `k` lies at `direction × (start + k × step)` voxels along the surface normal; `--direction auto` steps
toward the denser side. Outputs: `max.tif` and `mean.tif` (float32) with 8-bit copies, `mask.png`,
`composite.json`, the individual layers, and `--xy_slices N` slices over the surface's z range. From the
projections, `--sweep paganin.delta_beta=250,500,1000` renders several filter settings in one pass (CUDA only).

## Python

```python
from vc_tomography.recon import HelicalRecon

r = HelicalRecon("PROJ", cache_dir="cache", device="cuda")
img = r.reconstruct_fractions([0.5])[0]                       # one slice, float32
r.set_params(filters="paganin(delta_beta=500)+unsharp(sigma=1.2,coeff=4)")   # keeps the calibration
vol = r.reconstruct_heights(h, h + 32)                        # 32 slices from view height h
```

`HelicalRecon` takes the same parameters as `--set` (`params={...}`) and the same `projection_store` and `reduce`
options. `vc_tomography.band.BandComposite` computes surface composites from the projections.

## Limitations

- Slices are reconstructed one at a time in a parallel-beam model. Effects that move a ray out of its slice (beam
  divergence along the axis, tilt of the rotation axis toward the beam) are measured by the geometry calibration
  but not applied, except the mosaic canvas's shift of each view's rows by the pitch (see above). `manifest.json`
  lists them under `provenance.geometry.measured_not_applied`.
- A mosaic is reconstructed as one slice only by FBP, and written only as TIFF.
- Parameter sweeps run only on CUDA.
- Only xy slices are produced.
