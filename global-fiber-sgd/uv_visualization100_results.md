# 100-sweep fiber maps

Both completed saved solutions are exported under
`data/uv_categorical_v1/visualization100/{ccw,cw}/`.

- `layout.tif`: LZW tiled TIFF, 2000px high entirely devoted to the map;
  no title, axes, ticks, legend, padding or histogram. Equal U/V base-voxel scale.
- `layout.svg`, `layout.pdf`: full vector map, annotations, fitted winding
  boundaries and both confidence histograms. No tooltips or raster map embedding.
- `histograms.tif`, `histograms.svg`: standalone confidence distributions.
- `summary.json`: input hashes, bounds, dimensions, retained counts/confidence.

Same readout as earlier plots: omit P(broken)>0.5; draw every remaining piece at
its most probable winding, H blue/V orange, opacity equal to joint winding
probability. This is not a stricter joint-probability>0.5 filter. Boundaries remain
dashed within the raster map. Neither probabilities nor geometry were modified.
PDF preserves continuous opacity; raster opacity is necessarily 8-bit.

PDF pages are uniformly scaled to at most180in on either dimension (below the
traditional200in PDF page limit); physical U/V scale remains equal. Files are
very wide because the solution spans many windings. Zoom rather than fit-page
for detailed inspection. Viewer performance is not guaranteed for400k paths.

## Reproduce (CPU only)

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_categorical_visualization100.json
```

Uses immutable ccw_sweep100/cw_sweep100; no inference, optimization, GPU grant,
or energy reevaluation. Historical20-sweep figures remain untouched.

## Verification

| Solution | Retained pieces | Map TIFF dimensions | TIFF MiB | PDF MiB |
|---|---:|---:|---:|---:|
| CCW | 395549 | 108962 x 2000 | 19.23 | 29.71 |
| CW | 395118 | 105507 x 2000 | 18.95 | 29.68 |

Seven visualization tests passed: signed endpoints/selection, histories, SVG
completeness, cumulative boundaries, tiled pixels, map-only dimensions/LZW,
histograms, vector PDF without image objects, fresh CPU-only path, source-identity
mismatch rejection. Independent review found a PDF font-size mismatch; fixed
and regenerated exports. Raster detail and separate histogram inspected.

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_visualization.py
```
