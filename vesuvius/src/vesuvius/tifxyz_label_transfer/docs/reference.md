# Reference

Complete command, option, output, and performance documentation. For the
concepts behind these tools see [theory.md](theory.md).

## Environment

- Package: `vesuvius.tifxyz_label_transfer`
- Language: Python 3.14; platforms: Ubuntu/macOS, amd64/arm64
- Inputs: TIFXYZ folders, grayscale PNG/TIFF labels, koine-style label
  OME-Zarrs, optional registration JSON
- Tests: pytest over the ported standard-library `unittest` cases
- Non-negotiable behavior: affine matrices use XYZ order; categorical
  labels are never bilinearly blended; unmapped regions receive an
  explicit validity mask

```bash
uv sync --extra label-transfer --extra tests
```

The `label-transfer` extra does not install torch, napari, or a Qt backend.

### Native rasterizer

Build the dependency-free C++17 rasterizer once to accelerate the
per-pixel stage:

```bash
uv run --no-sync python -m vesuvius.tifxyz_label_transfer.build_native
```

This uses `CXX` when set and otherwise invokes `c++`. A normal GCC or
Clang C++17 compiler is sufficient; no Python headers, CUDA toolkit,
OpenMP runtime, or additional Python package is required. The generated
platform-specific `_native_rasterizer.so` is intentionally ignored by Git
and must be rebuilt on each machine.

The native library embeds a fingerprint of its C++ source plus an ABI
version and structure-size checks. A stale or incompatible binary is
rejected instead of being used silently. `--rasterizer auto` is the
default: it uses a valid native build and otherwise falls back to the
NumPy reference. Use `--rasterizer native` to require the optimized path
or `--rasterizer python` for differential testing.

### Remote configuration

All remote reads go through `rclone`; HTTP and the Hugging Face client are
never used. `rclone` is an external binary: install and configure it yourself;
the Python package does not provision or wrap it.

- `--open-data-rclone-root` defaults to an anonymous inline rclone S3
  remote for the public `vesuvius-challenge-open-data` bucket, so no local
  rclone configuration is required for public data.
- `--ink-rclone-root` (evidence preparation, viewer) must point at your
  own rclone mirror of the private ink dataset.
- `--source-raw-rclone-root` (self-render) must point at your rclone
  remote holding the raw scan volumes; `--source-raw-volume` overrides the
  provenance-based resolution entirely.
- AWS credentials are inherited from the environment
  (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`) or your rclone config.
  `--aws-credentials-file` optionally loads shell-format exports, and is
  only consulted when `AWS_ACCESS_KEY_ID` is absent:

```bash
set -a
source /secure/path/to/aws-credentials
set +a
```

## Shape inference and overrides

The output shape is inferred from the label and TIFXYZ dimensions (see
[theory.md](theory.md)). If an existing full-canvas target surface-volume
render uses a different sampling scale, pass it with `--target-reference`;
only its dimensions are read. `--output-shape HEIGHT WIDTH` is an
equivalent explicit override. For the normal full-canvas case the only
required inputs are the three TIFXYZ folders, the label, and the
registration JSON — the original render commands are not required.

## transfer.py pipeline

```bash
python -m vesuvius.tifxyz_label_transfer.transfer pipeline \
  --old-tifxyz /data/old-2.4um.tifxyz \
  --updated-tifxyz /data/updated-2.4um.tifxyz \
  --target-tifxyz /data/native-9um.tifxyz \
  --label /data/old-2.4um-inklabels-v2.zarr \
  --additional-label /data/old-2.4um-supervision-v2.zarr \
    /data/updated-2.4um-supervision.tif /data/native-9um-supervision.tif \
  --additional-label /data/old-2.4um-validation-v2.zarr \
    /data/updated-2.4um-validation.tif /data/native-9um-validation.tif \
  --affine /data/2.4um-to-9um.json \
  --intermediate-output /data/updated-2.4um-inklabels.tif \
  --output /data/native-9um-label.tif
```

Label inputs may be ordinary 2D TIFF/PNG files or koine-style OME-Zarrs.
For a 65-plane annotation Zarr the center label plane (Z=32) is read from
level 0. Repeating `--additional-label` transfers ink, supervision, and
validation through one shared geometric pass: their categorical pixels and
sidecars are identical to separate runs, but the expensive correspondence
and per-pixel 3D verification are performed only once.

By default the affine direction is tested both ways. `forward` has the
registration repository's meaning, `p_target/fixed = M * p_source/moving`;
override an ambiguous decision with `--affine-direction forward|inverse`.

Before allocating full-resolution output rasters, every stage samples
target TIFXYZ vertices against the exact source triangles and aborts when
sampled coverage is below 1%, which normally means the surfaces use
different volume frames or the supplied affine/direction is wrong. Final
pixel coverage is checked again before any TIFF is written. For different
volume frames in stage one, supply the registration explicitly:

```bash
--stage-one-affine /data/old-volume-to-updated-volume.json \
--stage-one-affine-direction forward
```

Use `--minimum-mapping-coverage 0` only when a near-zero-overlap transfer
is deliberate. Both sampled and final coverage are recorded in the report.

Add `--dry-run` to inspect all inferred shapes and scales without
processing.

An approved canvas offset is applied with `--label-canvas-offset DY DX`:
label pixel `(i, j)` depicts source canvas position `(i + DY, j + DX)`.
Mapped pixels whose corrected label position falls outside the raster
become invalid rather than clamped.

## transfer.py single

One stage on its own, same-volume or cross-volume:

```bash
python -m vesuvius.tifxyz_label_transfer.transfer single \
  --source-tifxyz /data/old-2.4um.tifxyz \
  --target-tifxyz /data/updated-2.4um.tifxyz \
  --label /data/old-label.tif \
  --output /data/updated-label.tif

python -m vesuvius.tifxyz_label_transfer.transfer single \
  --source-tifxyz /data/updated-2.4um.tifxyz \
  --target-tifxyz /data/native-9um.tifxyz \
  --label /data/updated-label.tif \
  --affine /data/2.4um-to-9um.json \
  --output /data/native-9um-label.tif
```

For a manual second stage, pass the first stage's validity with
`--source-validity /data/updated-label.valid.tif`; the `pipeline` command
does this automatically and composes it with the second stage's geometric
validity. Use `--distance-output distances.tif` to additionally save the
surface matching distance for every mapped pixel.

## Outputs

For `--output native-9um-label.tif`, the tool writes:

- `native-9um-label.tif`: transferred categorical labels
- `native-9um-label.valid.tif`: `255` where a surface correspondence was
  accepted in both pipeline stages, `128` where this or an earlier stage
  was seam-filled, and `0` elsewhere
- `native-9um-label.report.json`: shapes, scales, affine decision,
  distance threshold, coverage and distance statistics

Output rasters are disk-backed while processing, so full label canvases do
not have to fit in RAM. The XYZ arrays, spatial index, and
stored-resolution UV correspondence fields do need to fit.

## Canvas offset workflow

### 1. Prepare evidence

`prepare_canvas_offset_evidence.py` fetches both CT comparisons through
`rclone`. A shipped annotation TIFF is copied as one object and
downsampled tile-by-tile; uncompressed open-data Zarr chunks are
byte-range read so unused depth layers are not downloaded; compressed
chunks are fetched whole and decoded locally.

```bash
python -m vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence \
  --case-dir /data/cases/pherc0139-w039 \
  --ink-rclone-root myremote:bucket/datasets/ink/ink_YYYYMM
```

When the source and updated scans have different physical Z resolution,
pass the source value explicitly with `--source-resolution-um`. This
affects only physical slab matching; it never creates or estimates a
volume affine.

### 2. Estimate and approve

`estimate_canvas_offset_evidence.py` works entirely from those local
artifacts. Every tile shift, peak, inlier decision, and fitted drift is
recorded; the consensus is approved only when at least two comparisons
converge, each supports a constant-translation model, and their
full-resolution offsets agree:

```bash
python -m vesuvius.tifxyz_label_transfer.estimate_canvas_offset_evidence \
  --case-dir /data/cases/pherc0139-w039 \
  --output /data/cases/pherc0139-w039/affines/hf-render-canvas-offset.json
```

If source and target TIFXYZ coordinates belong to different volume frames,
the same command accepts a registration with
`--stage-one-affine ... --stage-one-affine-direction forward`. It never
estimates that affine.

### 3. Raw-CT self-render check

`self_render_tifxyz.py` independently renders both TIFXYZ surfaces through
their corresponding raw volumes. Source samples are mapped onto the target
canvas through the existing 3D correspondence; only textured overlap is
rendered, so a smaller source surface inside a larger target is supported.

```bash
python -m vesuvius.tifxyz_label_transfer.self_render_tifxyz \
  --case-dir /data/cases/pherc0139-w039 \
  --source-raw-rclone-root myremote:raw-volumes
```

The report separately records the source raster-to-own-TIFXYZ offset and
the two-sided source-on-target versus target geometry residual; the
overall check passes only when both pass. It never derives or replaces a
volume affine. The source raw CT is resolved from the source surface
Zarr's `source_zarr` provenance, remapped through
`--source-raw-rclone-root`; use `--source-raw-volume` when that internal
path cannot be mapped or to override the resolution entirely.

When a transfer report does not exist yet, validate just the source canvas
first — this breaks the otherwise circular dependency between approving
the label offset and producing the target-stage report:

```bash
python -m vesuvius.tifxyz_label_transfer.self_render_tifxyz \
  --case-dir /data/cases/phercparis4-w01 \
  --source-only \
  --source-surface-zarr public-s3:sample/segment/surface.zarr \
  --source-raw-volume public-s3:sample/volumes/raw.zarr \
  --raw-cache-dir /data/shared-raw-chunks
```

`--source-only` approves only the raster-to-source-TIFXYZ translation and
sets `geometry_approved` to `null`. `--source-surface-zarr` can point at a
public mirror of the source surface metadata while `--source-raw-volume`
identifies the matching raw CT. A shared `--raw-cache-dir` avoids
re-downloading overlapping diagnostic chunks across segments of one scan.

When the dataset ships an independent annotation maximum, that physically
matched raw-CT self-render may approve the source canvas even if the exact
center comparison has no usable textured tiles; the report records
`basis: "annotation-maximum-only"` and preserves the center error instead
of silently treating it as agreement. If the ink dataset has no shipped
`*_max_FIRST_LAST.tif`, self-render falls back to the published surface
Zarr's exact center plane (`source_reference_kind` is
`surface-volume-center`) — still a direct raster-to-own-TIFXYZ check, but
not independent maximum-slab evidence.

### Single-pair diagnostic

For one diagnostic image pair, `estimate_canvas_offset.py` runs the same
band-pass, robust tile field, drift rejection, and iterative sign check
directly:

```bash
python -m vesuvius.tifxyz_label_transfer.estimate_canvas_offset \
  --source-tifxyz /data/old-2.4um.tifxyz \
  --target-tifxyz /data/updated-2.4um.tifxyz \
  --source-render /data/renders/hf-original-2.399um-level2-middle3-max.tif \
  --target-render /data/renders/2.399um-level2-middle3-max.tif \
  --output /data/affines/hf-render-canvas-offset.json
```

The output JSON records the offset in full-resolution source-canvas pixels
plus full provenance (per-iteration measured shifts, scatter, residual,
tiles, peaks, drift). Prefer the evidence workflow above for approvals;
this tool is the underlying engine and remains useful for ad-hoc
comparisons. It accepts `--stage-one-affine` with an explicit direction
(automatic direction scoring uses point-to-surface distances, which are
blind to tangential translations).

The viewer summary reports `render_residual_shift_px` for both stages as a
standing check that no constant image offset remains.

## Label Zarrs

`make_label_zarrs.py` converts transferred label TIFFs into the OME-Zarr
layout the ink-detection preprocessing expects (same conventions as
`vesuvius.ink_detection.preprocessing.create_label_zarrs`): the
2D label embedded at slice 32 of a 65-deep z/y/x volume, a 6-level
OME-NGFF 0.4 pyramid with nearest-neighbour downsampling, chunks of
`(65, 128, 128)`, Blosc zstd level-3 bitshuffle compression, and `/`
dimension separators. The store is explicitly Zarr v2 under either supported
Zarr major. Levels are derived in 2D and streamed block-wise, so
full-resolution canvases convert without materialising the 65-deep volume.

```bash
uv run --no-sync python -m vesuvius.tifxyz_label_transfer.make_label_zarrs \
  results/supervision-2.399um.tif results/supervision-9.362um.tif --overwrite
```

Each output (`<input>.zarr` next to the TIFF, or `--output` for a single
input) records `source_image`, `canvas_size`, and the sibling
`*.report.json` path in its `.zattrs` — the raw annotation Zarrs' empty
attributes are what made the canvas offset so hard to trace, so these
Zarrs carry their provenance.

## Napari viewer

The viewer expects the directory layout produced by
`scrollprize.org/scripts/download_ink_app_inputs.py` plus pipeline outputs
in its `results/` folder:

```bash
python -m pip install napari
uv run --no-sync python -m \
  vesuvius.tifxyz_label_transfer.view_alignment_napari \
  --case-dir /data/pherc0139-w039 \
  --case-dir /data/pherc0139-w035
```

It opens one window with a compact **Comparisons** sidebar. Choose a
segment, then click a preset for CT alignment, labels, supervision masks,
mapping coverage, complete TIFXYZ surfaces, or annotation-only 3D points.
The preset hides unrelated layers and switches between 2D and 3D
automatically. Small checked buttons appear below the active preset for
each constituent layer; toggle either side off to inspect the other layer
raw, then turn both on for the additive comparison. The source is blue,
the target is red, and overlap is purple. Layer toggles and 2D preset
switches preserve pan and zoom; changing cases or switching between 2D and
3D refits the view.

Repeated `--case-dir` arguments are lazy in interactive mode: startup
loads only the first selected segment, and changing the segment unloads
its layers before preparing the next one. Slow evidence loading, TIFXYZ
projections, and residual measurements run in a napari worker thread; the
case selector is disabled and the sidebar reports `Loading …` while that
work is active, and layer creation/removal stays on Qt's GUI thread, so
the window remains responsive during first-time case preparation.

Memory scales with one case rather than the entire review list.
Transferred annotations are read from their OME-Zarr pyramid at the
closest useful display level; validity and source TIFFs decode only the
strips or tiles selected by the display grid. `--validate-only` checks
every supplied case sequentially without importing napari (batch audit
and headless cache pre-warm; it prints geometry, exact Zarr shapes,
selected Z indices, and byte counts, and writes a JSON audit record beside
each cached composite).

For a faster initial visual pass over many cases, add
`--preview-factor 4`. The viewer downsamples each CT render before the
TIFXYZ projection and stores the result in a separate `*-preview4.tif`
cache, leaving full-resolution caches untouched. Preview mode is
intentionally qualitative: it skips residual measurements and diagnostic
contact sheets. Reopen a case without `--preview-factor` before making a
quantitative approval decision.

### Presets

The three `Stage` presets are native-stage inspections, not a shared pixel
canvas:

1. `Stage · original 2 µm` — native HF CT and native HF annotations;
2. `Stage · updated 2 µm` — native updated CT and transferred annotations;
3. `Stage · final 9 µm` — native final CT and projected annotations.

Within each preset, CT and annotations use that stage's matching canvas;
preserving the display camera while switching stages is a navigation
convenience, not a claim that their pixels correspond. Magenta is ink and
cyan is supervision.

`2 µm CT alignment` is a true common-canvas comparison: it projects the
original HF CT through the old and updated TIFXYZ surfaces onto the
updated canvas, then compares it with native updated CT; its `Ink` and
`Mask` toggles add the transferred annotations without leaving the
comparison. Every blue/red CT pair must already have identical
destination-canvas shape — the viewer raises an error instead of resizing
mismatched native flattenings, because a resize is not a TIFXYZ
registration. `Self-render · center/matched max` compare raw-CT samples
from both TIFXYZ surfaces on the updated canvas. `Cross evidence ·
center/matched max` show the published-render checks. `Final CT
alignment` projects updated CT through the selected volume affine onto the
final canvas. The `Updated 2 µm + ink/mask` and `Final 9 µm + ink/mask`
presets show each transferred annotation over its destination CT.
Coverage appears in two separate destination-coordinate presets
(`Coverage · updated 2 µm`, `Coverage · final 9 µm`); the viewer never
resizes one surface's validity canvas onto the other surface's flattening.

Case discovery is resolution-agnostic: the open-data TIFXYZ names
(`<resolution>um-<volume_id>.tifxyz`) determine the stages, so scrolls
whose high-resolution volume is 2.4 µm rather than 2.399 µm work
unchanged. A case may also contain only the updated high-resolution TIFXYZ
and no native-resolution target (for example PHercParis4); the viewer then
shows only the stage-one presets and disables the final/affine ones.

### Remote reads

The viewer reads the public surface-volume Zarrs through rclone: it
inspects their metadata, chooses the middle three Z indices, and
byte-range reads only those planes for uncompressed chunks — the Zarr is
never synced in full. The max composite is cached under
`CASE_DIR/renders/`; pyramid level 2 is the default and the reader uses
the selected level's actual metadata for all three dimensions (Z
downsampling differs between published surface volumes). Use
`--ink-rclone-root` for the private ink mirror (only needed when a source
render must be fetched remotely), `--open-data-rclone-root` to override
the public root, `--zarr-level N` to choose another level, or
`--skip-renders` for an offline run.

For viewer-quality annotations, generate labels at the same pyramid-level
canvas sizes by passing the cached updated and final CT composites as
`--updated-reference` and `--target-reference` to the pipeline. Forcing
output to the stored TIFXYZ dimensions discards annotation detail and
makes nearest-neighbour enlargement look blocky.

The scans use different voxel sizes and acquisition settings, so intensity
disagreement in a blue/red comparison does not necessarily indicate
geometric disagreement. The final-stage label and supervision layers are
not native annotations: they are the HF annotations transferred to the
updated surface and then projected through the volume affine.

## Matching controls

`--max-distance` rejects matches beyond a distance measured in target
volume voxels. The default `auto` threshold is 75% of the smaller median
stored-grid edge spacing after the affine. The same threshold is applied
again after interpolating each output pixel's source UV and target XYZ,
which rejects fold-seam jumps even when all four coarse endpoint matches
are close. Use a lower value (for example
`--cross-volume-max-distance 2.0`) to avoid jumping to a nearby winding.

`--nearest-vertices` controls how many nearby source vertices contribute
candidate triangles; the default of 8 favors robustness.
`--query-batch-size` controls peak temporary matching memory without
changing results. `--workers` threads the mapping batches and the output
tile loop (default: all cores); workers only write disjoint output regions
and statistics fold in submission order, so outputs and reports are
identical for any worker count.

`--uv-cache PREFIX` caches the mapped UV field to `PREFIX.<stage>.npz`.
The field depends only on the surface pair, affine, and matching
parameters — not on the label — so transferring a second label between the
same surfaces reuses the cache and skips the mapping phase entirely. The
cache key includes a fingerprint of the actual surface coordinates, so two
segments can never collide on one cache file; a cache whose recorded
configuration does not match is recomputed and rewritten, never silently
reused. Writers publish through a per-process temporary plus an atomic
rename, so concurrent jobs sharing a prefix are safe.

`--vertex-index` selects the nearest-vertex search structure. The default
`kdtree` uses scipy's parallel cKDTree and is fastest in benchmarks.
`grid` buckets the near-uniformly spaced TIFXYZ vertices into a uniform 3D
grid; it produces byte-identical outputs but only guarantees completeness
up to the mapper's `index_max_distance`, so `locate` rejects larger
`max_distance` values.

### Streaming output and performance

For TIFF destinations, the default per-pixel path writes completed tiles
directly into compressed tiled TIFFs. Compression overlaps the geometry
work through bounded queues, and complete uncompressed output rasters are
never created. Each TIFF is assembled beside its destination and
atomically published only after every encoder succeeds. Use
`--no-stream-output` only to reproduce the previous materialized-raster
path or when diagnosing TIFF writer compatibility; the two paths' decoded
arrays and mapping statistics are exact matches (byte layouts may differ).

On the PHerc0814 stage-one workload (8520×13820, three uint8 labels plus
validity, cached UV, 20 workers, 512-pixel tiles), `/usr/bin/time -v`
measured:

| output path | wall mean; min/median/max (n=3) | peak RSS mean; min/median/max | full-size temporary rasters |
| --- | ---: | ---: | ---: |
| `--no-stream-output` | 10.53 s; 10.39/10.44/10.77 | 1.90 GiB; 1.89/1.90/1.91 | 0.44 GiB |
| streamed TIFF + Python rasterizer | 9.83 s; 9.63/9.86/10.01 | 1.47 GiB; 1.47/1.47/1.48 | 0 GiB |
| streamed TIFF + native rasterizer | 1.28 s; 1.24/1.28/1.32 | 0.86 GiB; 0.86/0.86/0.86 | 0 GiB |

All six decoded outputs were pixel-identical between all paths with
identical mapping reports. The native kernel fuses target/UV bilinear
interpolation, source-triangle XYZ reconstruction, 3D distance rejection,
categorical source-index generation, and seam/provenance decisions.

The full PHerc1667 w018 stage-one workload (42380×98100, 4.16 billion
pixels, three v2 labels plus validity, approved UV cache) completed in
49.41 seconds with 22.81 GiB peak RSS and no 15.49 GiB full-raster
temporary, with outputs exactly equal to the previously approved ones.
Input label loading still dominates memory on multi-billion-pixel cases
because the source TIFFs are decoded in full; streaming source-label reads
would be the next optimization.

## Transfer methods

The production nearest-triangle implementation lives in `core.py`; it is
the correctness reference used by both the NumPy and compiled rasterizers.
The optional global approximation lives separately in `planar.py` so its
weaker guarantees cannot be confused with the production mapper.

| method | implementation | guarantee |
|---|---|---|
| nearest source triangle (default) | `core.py` | local 3D correspondence with per-pixel distance verification |
| global planar approximation (`--planar`) | `planar.py` | one fitted 2D affine; residuals quantify ignored local deformation |

`--fill-seams` is not a third registration method: it is an explicitly
marked post-process on the default UV field. `native.py` and
`native_rasterizer.cpp` accelerate the default rasterization step and are
required by differential tests to reproduce the NumPy reference exactly.

`--planar` (with `single`) replaces the per-pixel mapping entirely: a
sample of target vertices (`--planar-sample-vertices`, default 200000) is
mapped through the same 3D geometry, one global 2D affine is least-squares
fitted from those correspondences, and the whole label raster is warped at
once. Every output pixel whose mapped position lands inside the label
raster is filled, so fold seams and locally rejected geometry do not punch
holes; the cost is that any non-affine component of the true mapping is
ignored, quantified by the report's `residual_label_px` percentiles.

`--fill-seams` (with `single` and `pipeline`) keeps the per-pixel geometry
mapping but closes its holes: rejected vertices (fold seams, distance
failures) are filled by continuing the field smoothly from their nearest
measured neighbours. Filled pixels get validity `128` instead of `255`
(`seam_filled_pixels` counts them), and the provenance survives later
stages (`inherited_filled_pixels`). The fill is not width-bounded — large
unmapped regions are extrapolated too — and the continuation blurs across
genuine fold discontinuities rather than resolving them; mask
`validity == 128` downstream to recover the strictly measured transfer.

## Validation

```bash
uv run --no-sync pytest tests/tifxyz_label_transfer
```

The native differential tests factually skip when the library is not built.
After the explicit build command they treat the NumPy rasterizer as the oracle:
exact labels,
validity, float32 distance rasters, and statistics across affine and
identity mappings, canvas offsets, invalid masks, fold rejection, seam and
inherited provenance, multiple label dtypes, edge tiles, worker counts,
and a deterministic randomized surface matrix. They also verify
missing/stale native binaries fall back or fail as requested.

Before accepting transferred labels:

1. Check `mapping_coverage`, `distance_p50`, and `distance_p95` in the
   report.
2. Overlay the label and `.valid.tif` on the target surface render.
3. Inspect boundaries, holes, and regions where neighboring papyrus layers
   approach one another.
4. Inspect rejected bands near winding changes; those are deliberately
   invalid rather than filled from unrelated source UV regions.

## Current limitations

- Only complete, consistently oriented canvases can be inferred without
  extra information. A cropped/rotated/flipped label or target render
  needs to be restored to native orientation first.
- The global affine cannot correct local non-rigid differences between
  scans.
- Fine 2.4 µm label details below the 9 µm target sampling cannot be
  preserved.
- RGB/RGBA labels are intentionally rejected; use a single-channel
  categorical label image.
