# 151-winding continuation and contrasting maps

Status: both solves, persisted-state audits and visualizations complete. No reference labels enter optimization.

## Method

Both saved100-sweep solutions are expanded from100 to151 winding hypotheses
(indices0–150), plus broken. This is the existing **assignment-only** resolver:
widths, shared anchors and per-hypothesis geometry stay frozen. Added widths
repeat the last saved width. Old candidate physical positions remain unchanged;
broken probability remains unchanged. Nonbroken mass is redistributed100/151
to old candidates and51/151 uniformly to the additional candidates. This is a
deliberate warm-start perturbation, not equivalent to ten ordinary continuation
sweeps. Wider categorical support alone does not guarantee escaping a local fit.

Each convention runs ten full-graph sweeps, batch2048, temperature0.25, step1.
No loss weights changed. Shared12GiB GPU allocations on separate GPUs0/1.

## Reproduction

Use the existing approved command, replacing CONFIG with one filename below:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/CONFIG
```

- `run_uv_categorical_ccw_151.json`, `run_uv_categorical_cw_151.json`: solves.
- `report_uv_categorical151.json`: persisted-state and evidence audit.
- `run_reference_comparison100.json`: rerender previous solutions.
- `run_reference_comparison151.json`: render expanded solutions, reusing the
  same input-hashed physical projection cache, not redoing correspondence search.

## Visualization

Saturated HSV integer landmarks (saturation0.9, value0.95), golden-angle hue
spacing; fractional windings interpolate in OKLab. Out-of-gamut channels clip.
Opaque2px lines, stable sample order (later fibers overwrite earlier ones).
Equal physical U/V scale,1000px-high map-only LZW TIFF; zoomable SVG also retained.
Gray means unmatched reference association, not low probability. Confidence remains
available numerically in fibers.csv; it no longer fades the displayed colors.

Previous maps: `data/reference_comparison100_v1/{ccw,cw}/`.
Expanded maps: `data/reference_comparison151_v1/{ccw,cw}/`.
Map names: `fit_own`, `fit_reference`, `reference_fit`; `.tif` and `.svg`.

## Completed ten-sweep results

| Sweep | CCW retained length | CW retained length | CCW sign violations | CW sign violations |
|---:|---:|---:|---:|---:|
| 0 | 66.4410% | 66.3125% | 4 | 4 |
| 1 | 66.4085% | 66.2806% | 0 | 0 |
| 2 | 66.3548% | 66.2311% | 0 | 0 |
| 3 | 66.2502% | 66.1343% | 0 | 0 |
| 4 | 65.7966% | 65.6850% | 0 | 0 |
| 5 | 58.3342% | 58.1317% | 0 | 0 |
| 6 | 44.1517% | 43.9662% | 0 | 0 |
| 7 | 39.6551% | 39.4858% | 0 | 0 |
| 8 | 35.4083% | 35.2540% | 0 | 0 |
| 9 | 30.7531% | 30.6466% | 0 | 0 |
| 10 | 25.5162% | 25.4423% | 0 | 0 |

Both processes exited0. Optimization/audit loop:794.13s CCW /797.55s CW.
Peak allocated GPU memory7.38GiB each (observed allocator reservation ~8.4GiB).
Final rejected pieces59.9263% /60.0207%; retained events8.2050% /8.2001%.
Both MAP occupied-index ranges remain1–97: none of the new100–150 hypotheses
won. This experiment is **not promoted**. Zero violations was achieved together
with severe loss of retained coverage; the old solutions remain unchanged.

Interpretation: the broad warm-start mass redistribution introduces expected
pair costs against distant hypotheses. The update targets favored broken for
~90% of nodes throughout these sweeps. This is evidence against this expansion
initialization, not evidence that151windings cannot represent the scroll.
Geometry was frozen, so this was not a fresh joint151-winding solve.

## Verification and artifacts

35 regression tests passed. Full persisted-state audit passed for both runs:
all 6,606,100 observations retained, identities/partition intact, frozen geometry
and widths unchanged. All four original sign violations in each run disappeared
through endpoint rejection, not corrected placement. No continuation links remain
active (0/466). Detailed audit: `uv_151_audit.md`.

Both comparison renderers exited successfully. Existing100-sweep maps rerendered
in330.90s; expanded maps in158.81s, reusing the validated physical projection cache.
All six fiber maps in each comparison set report height1000. Raster is LZW TIFF;
SVG uses opacity1 and width2, with no tooltips. GPU allocations released.
Reference median absolute winding discrepancy remains large:12.6572 CCW /
12.6207 CW, measured only on the substantially reduced retained set. This is
not directly comparable evidence of improvement over the previous larger set.

