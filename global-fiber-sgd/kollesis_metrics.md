# Kollesis candidate maps: saved20 CCW/CW

These are exploratory evidence maps, **not detected kolleseis**. No endpoint
density, segmentation, overlap fitting, model update or new inference is used.
Both frozen20-sweep solutions are used to match the previously inspected layouts
and surfaces. The100-sweep jobs continue independently.

## Metrics

All coordinates/lengths are native base voxels. U and V bins are1000vx. Each
profile is the ratio of full-height numerator and denominator sums, not the
average of heatmap pixels. All100winding hypotheses contribute. Heatmaps are
split into8contiguous U panels with equal physical U/V scale, without wrapping U.

| Metric | Definition / denominator |
|---|---|
| Rejection (%) | Expected UV-stage rejection mass / total mapped fiber length |
| Uncertainty (%) | Conditional winding entropy divided by log(100), weighted by retained fiber length |
| Distance disagreement (turns) | Exact posterior expectation of absolute difference between absolute model layer gap and Lasagna measured distance; pair retention-weighted |
| Normal-distance bands (%) | Fractions of eligible extracted same-family parallel encounters in0–8,8–16,16–32,32–64,64–128vx |
| Close encounter rate | Eligible0–32vx encounters per1000retained fiber voxels; supplements the band fractions with a density measure |

Distance disagreement is also plotted separately for parallel and perpendicular
events. Model layer gap reuses the solver's fractional-angle correction and
H/V half-layer offset. Thus it tests the **current model**, including that model's
known inability to explicitly represent an overlapping sheet within a winding.
Unsigned measured distances are compared to absolute model gaps, not signed ones.
This is a diagnostic absolute error, not the optimizer's Gaussian-mixture loss.

### Crowding eligibility

Different original source fibers, same baked H/V family and existing extracted
parallel event. Both sheet-normal axes must be finite/nonzero with absolute dot
product>=0.9. Project the connector on each unit normal:

```
normal_distance = (abs(projection_a) + abs(projection_b)) / 2
transverse_distance = sqrt(length_squared - (projection_a^2 + projection_b^2)/2)
```

Require transverse<=32vx and normal<=128vx. All but the final normal-distance
band are half-open; the final includes128vx. Band fractions sum to100% wherever
the common denominator has support. Normal signs do not affect these distances.
There is no new exhaustive spatial pair search. Multiple extracted encounters
between a pair remain multiple local observations, and duplicate source traces
are not anatomically deduplicated by this diagnostic.

### Probability weighting and support

Each event splits half its weight to each endpoint's UV distribution. For an
endpoint hypothesis k, weight is0.5*p(endpoint,k)*P(other retained). The distance
numerator uses its conditional expected discrepancy against **all** hypotheses
at the other end, not a MAP approximation. Weighted prefix sums evaluate the
full pair expectation without storing100x100arrays. No line is drawn between
potentially remote endpoint charts.

Fiber metrics use uniform arc quadrature with spacing<=250vx, conserving total
length exactly (bin allocation is quadrature, not exact clipped segments).
That describes the saved original kollesis20results. The later H/V-specific
diagnostic refactor shares exact segment/cell overlap for future pooled
recomputation; old artifacts are unchanged. See hv_rejection_metrics.md.
Rejected mass is localized with the conditional winding distribution; those
locations are uncertain model estimates. Uncertainty uses that conditional
distribution and weights by retained length. Fibers excluded before the global
UV stage are absent; rejection refers only to this stage.

Heatmap cells require>=1000weighted fiber voxels or>=5weighted encounters,
depending on the metric. Other cells are gray/missing, not zero. Profiles require
the same minimum on their full-height support. Raw numerators and denominators
remain in NPZ, so support thresholds can be changed without recomputing evidence.
The supplementary density uses retained-length support; its encounter support
must also be inspected to avoid interpreting lack of observations as absence.

V-coherence plots show the percentage of **supported** V bins exceeding a
metric-specific, support-weighted global75th-percentile threshold, and separately
the percentage of the full V range that is supported. This is an exploratory
correlation indicator, not a calibrated significance test.
A U profile may pass aggregate support even if no single V cell passes its
threshold. Peak lists therefore additionally report those with>=25%of V bins
supported, separated by at least10000vx in U. Spearman profile correlations use
only buckets where both metrics have>=25%V support; they are descriptive, not
independent significance tests (neighboring buckets are spatially correlated).

## Caveats

Crowding can reflect same-sheet fibers, duplicates, compression or folds. Distance
disagreement can rise from diffuse assignment distributions, not only physical
layer anomalies. Retention weighting suppresses pair evidence in rejected areas.
Rejection/uncertainty are consequences of the current model, not independent
anatomical observations. Cross-height agreement is useful but not proof of a
kollesis. CW and CCW charts have different fitted coordinate gauges; don't compare
their numeric U positions as if they referred to the same physical location.

## Run and artifacts

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_kollesis20.json
```

Output: `data/kollesis20_v1/{ccw,cw}/`.

- `profiles.svg`, `profiles.tif`: all four requested diagnostics plus supporting
  subtypes, normal-distance bands and close-encounter rate.
- `*_heatmap.svg/.tif`: same metrics over U/V, eight equal-scale U panels.
- `*_support.svg/.tif`: mapped length/encounter support in equal-scale panels.
- `v_coherence.svg/.tif`: elevation across supported V bins and coverage.
- `profiles.csv`, `maps.npz`: numeric profiles and raw sums/denominators.
- `accumulation.npz/.json`: costly accumulation and input hashes, saved before
  plotting; `summary.json`: provenance, timing and mass-conservation checks.

CPU only. Eight bounded independent event batches share read-only input arrays;
histogram reduction order is fixed. No GPU-manager allocation, no dependencies.
The read-only database geometry scan is cached by database hash and event IDs.
Use `run_kollesis20_replot.json` through the same command to regenerate profiles
from validated accumulation files, without recalculating events. Remove its
`plot_names` selector to regenerate all figures. Inspection PNGs are written only
under `/tmp`; delivered figures remain SVG and LZW TIFF.

## Validation

```bash
/home/hendrik/.venv_las/bin/python -m pytest -q --noconftest -o pythonpath=/home/hendrik/villa2/vesuvius/src vesuvius/tests/test_fiber_uv_kollesis.py
```

Tests compare sorted-prefix expectations and actual endpoint-map deposition to
brute force, including unequal widths and phase/HV offsets; check length mass,
entropy/rejection, reversed direction, missing support, distance-band partition,
ratio-of-sums profiles, SVG/CSV/NPZ output and LZW TIFF encoding.
