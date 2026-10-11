# Fiber density: definitions and limitations

Inputs: immutable100-sweep CCW/CW layouts, after earlier normal-sign/HV/direction
rejections. Fibers removed before the UV stage are not in this experiment.

## Density per inferred surface

H and V are separate families associated with the two sheet surfaces; do not pool
windings by angle. U is the full unwrapped coordinate. Primary metric is source
fiber arc length within each UV cell / cell area, multiplied by1,000,000. Units
are base-voxel fiber length per million squared base voxels of inferred UV area.
The existing axis-aligned UV piece model deposits exact intersected length,
renormalized to source arc length. This is not a measurement of3D surface area.

Secondary count metric: one piece per cell it intersects, divided by same area.
U profiles are computed independently of heatmaps; a long V piece is counted
only once per U bin. Counts are split-sensitive, so prefer length density.

Common bounds include every selected endpoint in all three versions, snapped to
full3000vx heat cells. They are identical across versions and H/V. A profile is
length divided by10000vx U-bin width (clipped at this rectangle's edges) times
the full common V height. Extra trailing zero-area profile bins export NaN;
positive-area cells without fibers are zero. No occupancy-derived normalization
that could manufacture density gains. Unobserved parts of this rectangle are
not proven absent material. Missing acquisition/assignment support can lower
density; folds, extraction duplicates, UV distortions and variable completeness
can raise/lower it independently of kollesis.

## Three placements

- `enabled`: P(broken)<=0.5, maximum non-broken bucket, no confidence weighting.
- `all_saved`: add all UV-disabled pieces, each at its saved maximum non-broken
  bucket. Every included piece contributes full length, not attenuated by its
  rejection probability.
- `all_refit`: enabled positions stay exactly fixed. For each disabled piece,
  evaluate all100 frozen-geometry winding candidates against connected enabled
  pieces at their saved MAP assignments. Use canonical sign/magnitude/distance/
  metric costs, degree-calibrated sign charge and continuation cost, originalN/E
  normalization, unit neighbor weights. No disabled peers, no node or endpoint
  unary priors, no geometry refit. Select minimum cost; exact ties keep saved
  MAP. No enabled links => flagged fallback. This is a conditional diagnostic,
  not a new converged solution or re-enabling rejected fibers.

The cost-softmax (saved solver temperature) is only descriptive. Strong residual
conflicts can persist even at the best candidate, because disabled fibers may
connect incompatible regions. Support, cost gap, confidence, before/after sign
conflicts and choices are exported in placement.npz and summary.json.

## Files

`length_profiles.svg/.tif`: all3 versions overlaid, H/V rows, absolute density
and disabled-added density columns. `count_profiles.*` uses piece density.
Heatmaps are `<variant>_<length|count>_heatmaps.*`, including
`all_saved_added_disabled` and `all_refit_added_disabled`. Eight consecutive U
panels at equal U/V scale. Same color limits across all3 versions and H/V within
a convention; 99.5th-percentile positive values clipped for display only.

`*_profiles.csv`, `<variant>_<heat|profile>.npz` contain unmodified numeric
length/count/area values. `profile_peaks.json` is a ranking, not anatomical labels.
`summary.json` records hashes including shared helpers, geometry, inputs/cost
configuration, timings, placement diagnostics and family counts/lengths.

## Reproduce

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_density100.json
```

CPU only, no inference or GPU reservation. The conditional placement evaluates
Kx1 and1xK costs in batches instead ofKxK, reusing the same solver functions.
