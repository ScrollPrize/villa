# 151-winding resolver and high-contrast maps

User: saturated contrasting winding colors; smooth OKLab interpolation; opaque
fiber lines,1000px height,2px width, later fibers overdraw earlier ones. TIFF/LZW
only for rasters. Rerender existing reference comparisons from verified projection
cache (no repeated matching). Palette uses golden-angle hue spacing, saturated
HSV integer colors, OKLab interpolation, explicit gamut clipping on conversion.

Run both CCW/CW resolvers10sweeps with winding hypotheses0..150 inclusive (151
plus broken). Continue saved100sweep states by an explicitly logged expansion:
preserve existing hypothesis geometry/widths; append51widths equal to last saved
width; append neutral centered offsets, adjusting shared anchors to preserve old
coordinates. Preserve broken probability; allocate51/151of surviving mass to new
hypotheses uniformly, rescale old nonbroken masses by100/151. This gives new
choices finite support rather than near-zero logits. No new reference constraints,
reference-driven reinitialization or shifted winding gauge used in optimization.
Geometry stays frozen as in the current cost-target resolver; only assignments
update. Generalize categorical dimensions, q-range prior, warm validation and
report validation. Existing100defaults remain unchanged. GPUs0/1 shared12GiB
each, batch2048to bound larger K²scratch. Existing approved runner only.

Spec update: configurable categorical hypothesis count; old checkpoints expand
only when explicit expand_windings flag. Preserve all source/evidence checks.
Tests: old geometry/probability preservation, finite new support, rejection mass,
dynamic cost/updates, palette gamut/interpolation, opaque TIFF rendering. Independent
review. Docs: task/log/status/changelog, exact configs/commands and10-sweep metrics.
No promise that10sweeps with frozen geometry reaches a globally correct solution.
