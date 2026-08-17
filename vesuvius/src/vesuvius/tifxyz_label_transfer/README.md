# TIFXYZ label transfer

Transfer a 2D categorical label image (ink, supervision, or validation
masks) from one TIFXYZ surface canvas to another by matching the surfaces
in 3D.

Typical use: annotations were drawn on a render of an old flattening, and
the same sheet now has an updated flattening and a native lower-resolution
canvas that should carry those annotations:

```text
old 2.4 µm label
  -> old 2.4 µm TIFXYZ
  -> updated 2.4 µm TIFXYZ
  -> affine into 9 µm coordinates
  -> native 9 µm TIFXYZ
  -> 9 µm label
```

Label correspondence is determined entirely from TIFXYZ geometry: every
target pixel's XYZ volume coordinate is matched against the nearest source
surface triangles, verified in 3D, and the categorical source image is
sampled with nearest-neighbour interpolation — label values are never
blended. Phase correlation on CT renders is used only to audit a constant
annotation-canvas offset and to measure visualization residuals; it never
estimates the 3D volume registration and never replaces the surface
mapping.

- [docs/theory.md](docs/theory.md) — how the transfer works and why it is
  built this way.
- [docs/reference.md](docs/reference.md) — every command, option, output,
  and performance note.

## Environment

From the `vesuvius/` project, install the standalone label-transfer and test
extras. The label-transfer extra does not install torch or napari:

```bash
uv sync --extra label-transfer --extra tests
```

Optionally build the dependency-free C++17 rasterizer once per machine; it
accelerates the per-pixel stage roughly 8× and is verified against the
NumPy reference by differential tests:

```bash
uv run --no-sync python -m vesuvius.tifxyz_label_transfer.build_native
```

NumPy remains the default when this library is absent. Nothing compiles during
package installation or import. For the optional interactive viewer, install
napari separately (and the Qt backend appropriate for your platform):

```bash
python -m pip install napari
```

### Remote data

The QA tools read scan data exclusively through the external `rclone` binary
(never HTTP). Install and configure `rclone` yourself; this package neither
wraps nor auto-configures it.

- The public `vesuvius-challenge-open-data` bucket is read anonymously by
  default through an inline rclone remote; no local rclone configuration is
  needed.
- The ink annotation dataset and the raw scan volumes are private. Point
  `--ink-rclone-root` and `--source-raw-rclone-root` at your own rclone
  remotes for them (for example `myremote:bucket/datasets/ink/ink_YYYYMM`).
- Credentials come from your environment or rclone config; nothing is read
  implicitly from local files.

## Quick start

Run the full two-stage pipeline from the `vesuvius/` project:

```bash
uv run --no-sync python -m vesuvius.tifxyz_label_transfer.transfer pipeline \
  --old-tifxyz /data/old-2.4um.tifxyz \
  --updated-tifxyz /data/updated-2.4um.tifxyz \
  --target-tifxyz /data/native-9um.tifxyz \
  --label /data/old-2.4um-inklabels-v2.zarr \
  --affine /data/2.4um-to-9um.json \
  --intermediate-output /data/updated-2.4um-inklabels.tif \
  --output /data/native-9um-label.tif
```

Output shapes and render scales are inferred from the label and TIFXYZ
dimensions; add `--dry-run` first to inspect every inferred value before
processing large rasters. Repeat `--additional-label` to transfer ink,
supervision, and validation masks through one shared geometric pass. Use
`vesuvius.tifxyz_label_transfer.transfer single` for one stage on its own.

TIFXYZ XYZ values and `--max-distance` are in scan-voxel coordinates, not
micrometres. Canvas scales are stored as X/Y density values and exposed by the
tool as `(scale_y, scale_x)`; a scale near `0.05` means roughly 20 rendered
canvas pixels per stored TIFXYZ vertex.

For `--output native-9um-label.tif` the tool writes:

- `native-9um-label.tif` — transferred categorical labels;
- `native-9um-label.valid.tif` — `255` measured correspondence, `128`
  seam-filled (here or in an earlier stage), `0` unmapped;
- `native-9um-label.report.json` — shapes, scales, affine decision,
  coverage and distance statistics.

## Quality assurance

Annotations are drawn on published renders, which do not always sit
pixel-for-pixel on the TIFXYZ canvas the transfer reads. The blessed
approval workflow measures that constant canvas offset from two
independent CT comparisons and only approves when they agree:

```bash
uv run --no-sync python -m \
  vesuvius.tifxyz_label_transfer.prepare_canvas_offset_evidence \
  --case-dir /data/cases/pherc0139-w039 \
  --ink-rclone-root myremote:bucket/datasets/ink/ink_YYYYMM

uv run --no-sync python -m \
  vesuvius.tifxyz_label_transfer.estimate_canvas_offset_evidence \
  --case-dir /data/cases/pherc0139-w039 \
  --output /data/cases/pherc0139-w039/affines/hf-render-canvas-offset.json
```

Pass an approved offset to the pipeline explicitly with
`--label-canvas-offset DY DX`; the correction is never applied silently.
`vesuvius.tifxyz_label_transfer.self_render_tifxyz` provides an independent
raw-CT check of both the
source canvas and the cross-volume geometry, and
`vesuvius.tifxyz_label_transfer.estimate_canvas_offset` is the lower-level
single-pair diagnostic
behind the evidence workflow. The napari viewer
(`vesuvius.tifxyz_label_transfer.view_alignment_napari`) overlays every stage
for visual inspection.
All are documented in [docs/reference.md](docs/reference.md).

## Module map

| module | role |
| --- | --- |
| `core.py` | production geometry: surfaces, nearest-triangle mapping, per-pixel 3D verification |
| `io.py` | TIFXYZ, TIFF/PNG, and label-Zarr reading/writing |
| `planar.py` | optional global planar approximation (weaker guarantees, kept separate) |
| `native.py`, `native_rasterizer.cpp`, `build_native.py` | optional compiled rasterizer, differentially tested against NumPy |
| `transfer.py` | `single` / `pipeline` CLI |
| `make_label_zarrs.py` | convert transferred TIFFs to koine-style OME-Zarr labels |
| `prepare_canvas_offset_evidence.py` | fetch offset evidence (rclone only) |
| `estimate_canvas_offset_evidence.py` | blessed offset estimation + approval from that evidence |
| `estimate_canvas_offset.py` | single-pair offset estimator (library + diagnostic CLI) |
| `self_render_tifxyz.py` | raw-CT self-render validation |
| `view_alignment_napari.py` | interactive inspection viewer |

## Tests

```bash
uv run --no-sync pytest tests/tifxyz_label_transfer
```

Label Zarr exports are explicit v2 arrays under either supported Zarr API and
are covered by the project's Zarr 2.18.7/3.x matrix.

## Limitations

- Only complete, consistently oriented canvases can be inferred without
  extra information; a cropped/rotated/flipped label must be restored to
  native orientation first.
- The global affine cannot correct local non-rigid differences between
  scans, and fine source detail below the target sampling cannot be
  preserved.
- RGB/RGBA labels are intentionally rejected; use single-channel
  categorical images.
