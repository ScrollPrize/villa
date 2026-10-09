# Full U/V sweep progression

Both layouts use equal U/V base-voxel scale, dashed fitted winding boundaries
at U=[0,cumsum(widths)], and no tooltips. Raster output is tiled LZW BigTIFF at
1000px total height (including histograms/annotations), with proportional plot
width. TIFF replaces the old independently-scaled PNG preview. CPU-only rerender;
no optimizer changes. Tile-wise rendering avoids allocating the full raster.

Global expected cost includes frozen geometry constants, excludes entropy. Lower is better.
Costs belong to the state at the listed sweep; pre-update history was shifted back one sweep.
No optimization rerun. Final cost evaluated from saved states using existing GPU cost code.

| Sweep | CCW violations | CCW global cost | CW violations | CW global cost |
|---|---:|---:|---:|---:|
|0|28,760|26.412998|27,968|35.220635|
|1|25,338|30.882744|24,513|38.034340|
|2|21,307|42.838618|20,229|47.699002|
|3|16,637|69.903887|15,368|70.525235|
|4|11,384|114.123194|10,474|101.550648|
|5|6,140|140.790919|5,661|96.948521|
|6|2,439|104.569254|2,229|55.974095|
|7|570|50.443491|580|28.648793|
|8|180|25.430882|236|17.393557|
|9|127|16.642464|138|12.638057|
|10|110|12.425041|104|10.260665|

## CCW
Retained pieces:378322; rejected omitted:111341.
Conditional confidence median:0.879384; >=90%:47.4093%; >=99%:24.4802%.
Joint confidence median:0.850799.
Final negative entropy term:-0.159147; variational free energy:12.265894.

## CW
Retained pieces:378075; rejected omitted:111588.
Conditional confidence median:0.878343; >=90%:47.3023%; >=99%:24.3838%.
Joint confidence median:0.850690.
Final negative entropy term:-0.159225; variational free energy:10.101440.

## Files and reproduction

Artifacts: `data/uv_categorical_v1/visualization/`.
- `ccw/layout.svg`, `cw/layout.svg`: all retained pieces, vector lines and histograms.
- `ccw/layout.png`, `cw/layout.png`: lightweight previews.
- `progression.svg`, `progression.csv`, `results.json`: curves and exact data.

H blue / V orange. Opacity is joint selected-bucket probability. Conditional
confidence excludes broken probability and describes winding ambiguity only.
Pieces with broken probability above50% are omitted explicitly. Axes use
independent scaling; the SVG is not an equal-aspect physical map. Source/piece IDs,
source arc spans, chosen buckets and probabilities are in line tooltips.

Original optimizer state committed first: `c01fb1ecd`. No optimization rerun.
GPU used only to evaluate missing final expected energies (12GiB shared grants);
rendering is CPU-only. Both grants released. Frozen geometry constants added
at every sweep; entropy excluded from curve and final entropy diagnostic shown
separately. Nonmonotonic cost behavior is genuine, not a history indexing error.

Approved command:
```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_categorical_visualization.json
```

Completed exports refuse overwrite through the managed evaluator. Use
`run_uv_categorical_rerender.json` with the same runner for CPU-only regeneration
from saved energies; validates source hashes before replacing plots.

Validation:39tests passed, including XML/line counts and confidence histograms
of both full exports. Independent review verified cost alignment, final costs,
378322/378075vector-line counts and confidence values. Both previews inspected.

