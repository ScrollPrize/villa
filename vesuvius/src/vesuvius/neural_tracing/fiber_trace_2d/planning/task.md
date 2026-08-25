# Task: apply staged Fiberlet filtering before replay

Allow `vc_fiberlets fiberlet-replay` to optionally apply the existing ordered
overlapping-box Fiberlet reduction stages before graph tracing.

- Accept the staged-filter CLI arguments on `fiberlet-replay`.
- Anchor every filter box lattice globally in base-volume coordinates.
- Select complete final-stage boxes intersecting the requested replay-radius
  corridor, then expand required input coverage backward through all preceding
  stages so offsets and graph endpoint reach cannot leave a final box partial.
- Materialize every required source chunk before tracing starts.
- Keep generated/storage chunk grids independent from filter analysis-box
  sizes and offsets; map them by base-space extents.
- Keep the filtered overlays transient for now. Persistent canonical anchor and
  Fiberlet generation caches remain reusable and globally anchored.
- Preserve current unfiltered replay when no stages are supplied.
