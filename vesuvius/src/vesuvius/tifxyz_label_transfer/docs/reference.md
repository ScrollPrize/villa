# Reference

Transfer grayscale TIFF/PNG or label OME-Zarr annotations between TIFXYZ
surfaces. See [Theory](theory.md) for how mapping works. Run commands from
`vesuvius/` after installing the required extras:

```bash
uv sync --extra label-transfer --extra tests
```

## Single transfer

```bash
python -m vesuvius.tifxyz_label_transfer.transfer single \
  --source-tifxyz /data/source.tifxyz \
  --target-tifxyz /data/target.tifxyz \
  --label /data/labels.tif \
  --output /data/transferred.tif
```

For different scan frames, add `--affine registration.json`.
`--affine-direction forward` means `p_target = M * p_source`; by default,
the tool checks both directions.

## Two-stage transfer

```bash
python -m vesuvius.tifxyz_label_transfer.transfer pipeline \
  --old-tifxyz /data/old.tifxyz \
  --updated-tifxyz /data/updated.tifxyz \
  --target-tifxyz /data/native.tifxyz \
  --label /data/labels.tif \
  --affine /data/updated-to-native.json \
  --intermediate-output /data/updated-labels.tif \
  --output /data/native-labels.tif
```

The pipeline carries validity through both stages. For a manual second
stage, pass `--source-validity` with the first stage's `.valid.tif`.
If the old and updated surfaces use different scan frames, also supply
`--stage-one-affine`.

## Common options

| Option | Purpose |
| --- | --- |
| `--dry-run` | Inspect inferred shapes and scales without transferring. |
| `--output-shape HEIGHT WIDTH` | Override inferred output dimensions. |
| `--target-reference IMAGE` | Use an existing target image's dimensions. |
| `--label-canvas-offset DY DX` | Correct a measured source canvas shift. |
| `--max-distance VALUE` | Set the matching radius in target-volume voxels; automatic by default. |
| `--fill-seams` | Fill nearby mapping gaps, marked with validity `128`. |
| `--max-seam-distance VALUE` | Limit filling distance in target vertices; default `25`. |
| `--seam-anchor matched\|content` | Measure filling distance from all matches or matches with nonzero annotation. |
| `--uv-cache PREFIX` | Reuse geometry mappings. |
| `--workers N` | Set processing threads. |

The automatic matching radius is 75% of the target's median stored-grid
edge spacing. Preflight checks sampled coverage before transfer; the
default minimum coverage is 1%. Seam filling cannot repair mismatched surfaces.

In a pipeline, repeat `--additional-label INPUT INTERMEDIATE OUTPUT` to
transfer supervision or other labels with the same geometry. With
`--seam-anchor content`, anchors include nonzero samples from any label
in the shared pass.

Use `single --help` or `pipeline --help` for all options.

## Outputs

For `--output labels.tif`:

| File | Contents |
| --- | --- |
| `labels.tif` | Transferred categorical labels. |
| `labels.valid.tif` | `255`: measured; `128`: filled in this or an earlier stage; `0`: unmapped. |
| `labels.report.json` | Shapes, registration, matching distances, and coverage. |
| `labels.seam_anchor.tif` | With seam filling: float32 anchor distances at target stored-grid resolution, in grid cells. |

Keep only `validity == 255` for strictly measured output, applying the
policy separately to each asset, including supervision masks. Check the
report and overlay the outputs on the target render before using them.

## Other tools

Run each module as `python -m vesuvius.tifxyz_label_transfer.MODULE --help`.

| Module | Purpose |
| --- | --- |
| `prepare_canvas_offset_evidence` | Prepare CT comparisons; remote reads require configured `rclone` access. |
| `estimate_canvas_offset_evidence` | Estimate a canvas offset from the prepared comparisons. |
| `estimate_canvas_offset` | Diagnose a single render pair. |
| `self_render_tifxyz` | Check alignment against raw CT. |
| `make_label_zarrs` | Convert label TIFFs to label OME-Zarrs. |
| `view_alignment_napari` | Inspect alignment and labels; requires napari. |

Canvas offsets must be passed explicitly to the transfer command.
These tools do not estimate volume registration transforms.

## Native acceleration and tests

An optional C++17 rasterizer accelerates transfer:

```bash
uv run --no-sync python -m vesuvius.tifxyz_label_transfer.build_native
uv run --no-sync pytest tests/tifxyz_label_transfer
```

`--rasterizer auto` uses a compatible native build when available and
otherwise falls back to Python. Use `native` to require it or `python`
to select the reference implementation. Native differential tests skip
when the library is unavailable.
