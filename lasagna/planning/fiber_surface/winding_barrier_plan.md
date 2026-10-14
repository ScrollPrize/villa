# Other-winding backup barriers

## Task
Add weak backup constraints at every other represented winding. Keep the real
fiber XYZ and aligned inward normals; duplicate evaluations, not free variables.

## Implementation
- Convert initial attachment U to cumulative-width winding and phase; map phase
  to each other winding using its own width. Exclude evaluations outside mesh.
- Freeze coordinates, normals and topology. Inner windings must be inward,
  outer windings outward. Squared hinge with 5 base-voxel clearance.
- Experiment weight .01 relative to attachment loss. Normalize arc weights per
  source sample over its possible winding copies, then globally.
- Backup support gate: nearest initial own-winding attachment in UV; zero below
  100 base voxels, linear to full at 300. Do not renormalize after gating.
- Reuse fixed bilinear sampler. Save all barrier arrays/settings and restore
  them for audits/continuation. Reject invalid normals, widths and settings.
- Matched 800-step .1 tangential/100x UV-smooth experiment, baseline weight0 and
  backup weight .01. Shared nonexclusive GPU0 reservation.

## Tests
Unequal widths, boundary exclusion, inner/outer signs, geometry-only gradients,
clearance, support gate, V raw-position semantics, zero weight and restore.
Compare fit errors, folds, signed violations (all pairs and gated pairs), runtime.

## Spec update
Opt-in weak other-winding barriers with fixed coordinates, inward normal order,
actual unoffset fiber XYZ and own-winding support suppression. Not a global
collision guarantee; fixed correspondence may be imperfect under deformation.

## Documentation updates
Document settings/units, exact runner configs, results and limitations in docs,
task_log/status and changelog. Existing half-resolution rendering stays unchanged.

## Follow-up: registered folded-quad maps
Use the existing geometry_masks diagnostic, not a new fold definition. Render a
binary TIFF mask and red CT overlay at exactly the CT/fiber pixel grid, including
inspection cropping. Save local/global quad coordinates and counts. Support
refresh from already-rendered CT without inference, CT fetching or GPU allocation.
Test grid-cell ownership, crop offsets and consistency with geometry_stats.
