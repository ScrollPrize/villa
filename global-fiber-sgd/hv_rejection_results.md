# H/V rejection results — 2026-10-09

Completed both saved20-sweep conventions. **10000vx U-profile buckets;
3000×3000vx heatmap cells**. No new solve/inference. Exact definitions, masks
and reproduction commands: [hv_rejection_metrics.md](hv_rejection_metrics.md).

## Plots

| Plot | CCW | CW |
|---|---|---|
| Profiles, posterior placement | [SVG](data/hv_rejection20_v1/ccw/posterior/profiles.svg) / [TIFF](data/hv_rejection20_v1/ccw/posterior/profiles.tif) | [SVG](data/hv_rejection20_v1/cw/posterior/profiles.svg) / [TIFF](data/hv_rejection20_v1/cw/posterior/profiles.tif) |
| Profiles, MAP placement / integer counts | [SVG](data/hv_rejection20_v1/ccw/map/profiles.svg) | [SVG](data/hv_rejection20_v1/cw/map/profiles.svg) |
| Disabled length heatmaps | [posterior](data/hv_rejection20_v1/ccw/posterior/disabled_length_heatmaps.svg) / [MAP](data/hv_rejection20_v1/ccw/map/disabled_length_heatmaps.svg) | [posterior](data/hv_rejection20_v1/cw/posterior/disabled_length_heatmaps.svg) / [MAP](data/hv_rejection20_v1/cw/map/disabled_length_heatmaps.svg) |
| Disabled count heatmaps | [posterior](data/hv_rejection20_v1/ccw/posterior/disabled_count_heatmaps.svg) / [MAP](data/hv_rejection20_v1/ccw/map/disabled_count_heatmaps.svg) | [posterior](data/hv_rejection20_v1/cw/posterior/disabled_count_heatmaps.svg) / [MAP](data/hv_rejection20_v1/cw/map/disabled_count_heatmaps.svg) |
| Disabled piece fraction heatmaps | [posterior](data/hv_rejection20_v1/ccw/posterior/disabled_piece_fraction_heatmaps.svg) / [MAP](data/hv_rejection20_v1/ccw/map/disabled_piece_fraction_heatmaps.svg) | [posterior](data/hv_rejection20_v1/cw/posterior/disabled_piece_fraction_heatmaps.svg) / [MAP](data/hv_rejection20_v1/cw/map/disabled_piece_fraction_heatmaps.svg) |
| Rejection probability, count weighted | [posterior](data/hv_rejection20_v1/ccw/posterior/probability_piece_heatmaps.svg) / [MAP](data/hv_rejection20_v1/ccw/map/probability_piece_heatmaps.svg) | [posterior](data/hv_rejection20_v1/cw/posterior/probability_piece_heatmaps.svg) / [MAP](data/hv_rejection20_v1/cw/map/probability_piece_heatmaps.svg) |
| Rejection probability, length weighted | [posterior](data/hv_rejection20_v1/ccw/posterior/probability_length_heatmaps.svg) / [MAP](data/hv_rejection20_v1/ccw/map/probability_length_heatmaps.svg) | [posterior](data/hv_rejection20_v1/cw/posterior/probability_length_heatmaps.svg) / [MAP](data/hv_rejection20_v1/cw/map/probability_length_heatmaps.svg) |

All heatmaps show H, V, V-H and V/H in separate columns, with weak/undefined H
ratios masked. Each also has LZW TIFF. Support profiles, numeric CSV/NPZ and
support-qualified ratio peak lists are beside these figures. MAP counts are
integer optimized-piece counts; posterior counts are weighted piece-equivalents.

## Global results (each piece counted once)

| Metric | CCW H | CCW V | CW H | CW V |
|---|---:|---:|---:|---:|
| Disabled length |37.2833%|32.2703%|37.4941%|32.3329%|
| Disabled pieces |55,527|45,202|55,955|45,206|
| Disabled piece fraction |22.0505%|19.0047%|22.2205%|19.0064%|
| Mean P(broken), piece weighted |22.3315%|19.3559%|22.5031%|19.3577%|
| Mean P(broken), length weighted |37.4569%|32.5383%|37.6645%|32.6018%|
| Total pieces |251,817|237,846|251,817|237,846|

V is **less** rejected globally, both by length and piece fraction. That does not
exclude localized V rejection at a joint. In the inspected profiles/heatmaps,
however, there is still no convincing common set of height-spanning V-only
rejection bands. Most well-supported U regions have V/H rejection rates below1.
Large ratio spikes mostly appear toward low-support chart edges or where H
rejection is small; use the difference and support plots alongside the ratios.

For example, CCW posterior disabled-length ratio9.44at U=3,765,000 combines
H=1.21%and V=11.38%, with only18.83H/11.83Vpiece-equivalents. Its largest MAP
counterpart is elsewhere, U=3,355,000:ratio9.21,H=1.94%,V=17.88%,50H/48Vpieces.
This instability across localization methods is another reason not to identify
these maxima directly as kolleseis. No geometry/anatomy candidates were accepted.

## Validation and runtime

- CCW calculation33.36s, complete figures98.5s; CW calculation31.07s,
  complete figures94.9s. CPU-only, installed Python/NumPy/Torch/Matplotlib stack;
  one run each, not an isolated performance benchmark.
- Full runs check conservation of total, disabled and expected-disabled source
  lengths across both families, both placements and both grid types. MAP counts
  checked to be integer. Solution piece identities/arcs/families verified.
- 28tests passed across H/V diagnostic, pooled diagnostic, layout and anchor
  modules, including SVG/TIFF/CSV smoke and shared exact segment overlap.
- Independent plan/code review approved. Review caught a nonzero-origin endpoint
  rounding edge case that could add zero-length counts; fixed and covered by a
  regression before the completed runs. Both conventions visually inspected.
- Original pooled metrics and saved solutions are untouched. Future pooled
  recalculation now shares exact length clipping, replacing its old250vx
  quadrature; this is documented rather than silently changing old artifacts.

Source files: `fiber_uv_hv_rejection.py`, shared `fiber_uv_kollesis.py`, dispatch
in `crop_benchmark.py`, config `run_hv_rejection20.json`, tests
`test_fiber_uv_hv_rejection.py`. No new dependencies or native build required.
