# Local constraint events and clean midpoint splitting

> Historical broken-graph experiment: post-split support filtering discarded certified constraints. These results do not evaluate the intended fine-splitting algorithm. Superseded by the certified-splits correction and rerun; retained for provenance.

## Outcome

Implemented and tested, **not promoted to defaults**. Strict local certification
followed by very fine splitting loses most geometric evidence. Parallel factors
help relative to orthogonal-only, but both variants are substantially worse than
legacy on independent reference winding accuracy.

| Method | Removal score ↑ (0% best) | Initial independent accuracy ↑ | Conditioned reference accuracy ↑ | Initial defect pieces | Wall time |
|---|---:|---:|---:|---:|---:|
| Legacy fixed 512/128 pieces | −110.6766% | 38.4615% | 100.0000% | 15.5965% | 263.4 s |
| Local clean splits + parallel | −90.5016% | 15.3846% | 57.6923% | 3.8167% | 453.7 s |
| Local clean splits, orthogonal only | −74.7873% | 11.5385% | 34.6154% | 3.7549% | 318.2 s |

Removal score is the existing `−100 × problematic / retained-fulfilled` metric.
Do **not** interpret its improvement as a successful result: the graph changes
radically and both local conditioned solves hit the 500-message limit without
recovering all reference windings (15/26 and 9/26, versus 26/26 legacy). The
benchmark's existing accept-message-limit policy was unchanged. No ordinary
measurement-disable alternative was enabled. Initial independent readout never
feeds reference assignments back into the solve.

[Plot](2026-09-27-local-events.svg) ·
[Complete results, commands, hashes and ladders](2026-09-27-local-events-results.json).

## What was implemented

- Discover bounded local encounters on original source traces, before piece
  construction, with the existing spatial index/polyline projection.
- Independently check incoming/outgoing directions. Retain locally parallel or
  orthogonal relations only when the side checks agree.
- Repeat parallel correspondence and winding-distance checks along short spans;
  require reciprocal correspondence and matching distance category/sign.
- Sample more finely around local classification/distance changes (two bounded
  refinement rounds). Transition cuts are bracket midpoints, not exact detected
  transition locations.
- Split at midpoints between consecutive orthogonal event arcs, plus parallel
  relation transition brackets. Full geometry is preserved; no overlapping arc
  length. Existing continuation/defect semantics remain.
- Retain a factor only if independently certified support survives in both
  assigned pieces. Paired halves are required for parallel correspondences.
  Reference cross-extraction preserves already-cut pieces and accepts valid
  available one-sided support.
- Normalize repeated parallel finite costs/defect incidence by represented arc
  length / 128 base voxels. Hard signs remain hard. Empty represented intervals
  are dropped. Shared BP and least-squares models carry fractional incidence.
- Remove parallel factors only after common segmentation. Factor CSVs expose
  source arcs, surviving support and evidence weights. Pre-BP piece CSVs allow
  exact geometry auditing.

Defaults for this experiment: event spacing 64, per-side tangent span 16,
parallel probe span 64, reciprocal tolerance 8 base voxels; angular tolerance
30°; per-side winding tolerance 0.25 with the same rounded distance category.
Near-identical encounters are deduplicated at 8-base-voxel arc-pair proximity;
parallel repeats preserve distinct A arcs. These are first-test parameters, not
an optimized configuration.

## What failed

Both variants have **identical 68,851 pieces before BP**, verified by comparing
the complete exported geometry. Their length percentiles in base voxels are:

| P5 | Median | P95 |
|---:|---:|---:|
| 1.0269 | 22.8043 | 96.8391 |

Extraction creates 58,700 orthogonal midpoint-cut candidates and 7,838 parallel
transition-cut candidates (coincident cuts are deduplicated). There are 36,377
refinement candidates. Consistency rejects 521,744 candidate attempts; that count
includes repeated discoveries, so it is not a unique-event percentage.

After cuts, 48,626 events are rejected for insufficient/empty represented
support, leaving 4,854 ordinary events in the parallel-enabled graph: **90.92%**
of the post-consistency event population is lost at this stage. Many pieces are
too short for the certified support. Orthogonal-only omits 542 parallel
events before represented-width filtering; 33 of those would otherwise be
dropped for empty represented support.

The existing benchmark subsequently keeps the largest active winding component.
Adding/removing parallel factors can change this component; that is why the
saved final piece lists differ despite identical pre-BP splitting.

| Method | Retained BP pieces | Continuation constraints | Ordinary parallel | Ordinary orthogonal |
|---|---:|---:|---:|---:|
| Legacy | 6,790 | 4,012 | 61,595 | 95,529 |
| Local + parallel | 62,777 | 61,049 | 398 | 4,229 |
| Local orthogonal only | 62,559 | 60,832 | 0 | 4,237 |

The new score is dominated by continuation links, not preserved independent
geometric evidence. Lower defect percentages also do not imply better winding
quality. A sensible follow-up is to cluster nearby orthogonal events or constrain
cut placement by usable support, before tuning BP costs. That follow-up is **not**
part of this implementation and was not silently enabled.

## Reproduction

Input: the frozen Paris4 2048 crop's beam8, coverage40/80, 45° angle,
forward/backward-consistency artifact:

`/ephemeral/hendrik/fiber_crop_bench/20260923_beam2048_v1/beam-crop-new2048-coverage40x80-angle45-reverse-v2.zarr`.

Same normal manifest, reference fiber directory/tag, scale, original solver
budgets and weight settings as the preceding retention benchmark. No inference,
retrace, exhaustive predictor/model search, or new dependency installation.
Build: RelWithDebInfo, 64 CPU threads per trial, one timed run each; trials
overlapped, so these are observed wall times rather than isolated speed claims.
Legacy reproduces the prior score and independent accuracy exactly.

Approved build:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
```

Run each of the three configs with this same approved command:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/crop_benchmark_local_events_local-parallel_20260927.json
```

Sibling config names: `crop_benchmark_local_events_legacy_20260927.json`,
`crop_benchmark_local_events_local-orthogonal_20260927.json`,
`crop_benchmark_local_events_tests_20260927.json`, and
`crop_benchmark_local_events_report_20260927.json`.

Artifacts:
`/ephemeral/hendrik/fiber_crop_bench/20260927_local_events_v1/`.

The two `audit-local-*` configs repeat extraction with unchanged 500-iteration
normal alignment/orientation settings, no reference-conditioned solve, and only
one winding iteration. They export pre-BP geometry and rejection/cut counts;
their BP outputs are **not** benchmark scores. Earlier audit attempts shortened
the shared BP setting, which also shortened normal alignment; those were
discarded and remain under `attempts/`. The corrected audit geometries were
checked against every retained piece of the full runs.

The implementation review caught/fixed zero-support hard factors, unweighted
finite hard-sign slack, reference endpoint support, and transition brackets
spanning rejected neighborhoods. Regression coverage includes clean/full
geometry, endpoint support, H→V turns, distance/sign inconsistency, parallel
ablation geometry, and fractional evidence/finite-cost invariance.

Validation: 112 native test cases and 30 Python tests passed. The final retained
signed-event counter correction is diagnostic-only; benchmark artifacts preserve
the exact binaries, shared-library hashes and source patch used for measurement.

Portable C++/existing OpenMP; no new platform-specific dependencies or reduced
precision. Local event mode is opt-in. Very short reference pieces with no
complete one-sided probe still correctly provide no constraint. Sampling and
transition localization are bounded approximations, not exhaustive continuous
geometry proofs.
