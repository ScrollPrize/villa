# Minimum-cost-flow winding experiments

Run whole-fiber or independently split-fiber solves from existing AFV geometry,
umbilicus and tensor caches. This runner uses the production `Paths`, `graph`,
`Pieces` and certified circulation solver. It does not regenerate measurements.
It runs on CPU with the project's existing NumPy/SciPy dependencies.

From the repository root, with the project Python environment active:

```sh
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_experiment --config /path/to/local-experiment.json
```

Keep experiment JSON, dataset references and generated outputs outside the repository.
No dataset configuration is bundled with this runner.

## Configuration fields

The external JSON object has `output_root` and a nonempty `cases` array. Output
must be absent or empty; reruns require a fresh output directory. Case names must
be unique and contain only letters, digits, underscores and hyphens, starting
with a letter or digit.

| Field in each case | Meaning |
|---|---|
| `name` | Output subdirectory name |
| `dataset` | Dataset object, or path to a JSON dataset object |
| `target_mode` | `fractional` (default) or `rounded` |
| `sign_only` | Boolean, default false; use one-sided ordering penalties |
| `split` | Optional object; omission preserves whole fibers |
| `evaluation` | Optional GT scoring object |
| `baseline` | Optional path to a saved `solution.npz` |

A dataset object requires `afv`, `umbilicus` and `tensor_cache`. It may provide
`tensor_sha256` (enforced), `lasagna_winding_scale` (default 1),
`perpendicular_offset` (default 0.5) and `perpendicular_relative_weight`
(default 1). Existing solver config files may be supplied as dataset objects;
other fields in those legacy files are not solver parameters for this runner.

Relative paths are resolved against the JSON file declaring them. For a dataset
loaded from a separate JSON file, its paths are relative to that file. Inline
datasets, evaluation paths, baselines and output paths are relative to the
experiment file. There are no implicit project-directory conventions.

The `split` object accepts `minimum_length_vx` (default 1000),
`minimum_perpendicular` (default 4), and `protected_sources` (default empty).
Protected source IDs remain intact; their winding labels are still optimized.
Minimum length zero gives identity splitting. All pieces remain enabled, with
zero continuation penalty. Split thresholds operate on cached perpendicular
observations; they do not guarantee that every faulty fiber can be split.

## Evaluation

`evaluation` requires `mode` and a `reference` JSON path. The reference contains
`fiber_offsets`, mapping source IDs to integer offsets relative to cached
unwrapped phase. Its optional `excluded_source_ids` list documents exclusions;
excluded IDs must be omitted from `fiber_offsets`. Only IDs present in
`fiber_offsets` are scored. No reference labels are fixed during optimization.

- `relations`: whole-fiber tests with known offsets for both endpoints. Compare
  local winding differences to reference relations, including fiber transport.
  Connector transport cancels between prediction and reference. Relations between
  disconnected components are excluded as unidentifiable; their count is reported
  separately from missing-reference events.
- `cross_events`: additionally requires `events` (JSON path) and `dataset`
  (the separate GT dataset object or JSON path). Supports whole and split fibers.

Cross events accept either a bare list or a generated object containing an
`events` list. Each record has `gt_source`, `test_source`, `gt_piece`
(index into GT cache), `gt_arc`, `test_arc` (original polyline arc lengths),
`signed_distance`, `family_delta`, and boolean `parallel`. Additional event fields
are preserved in the exported event table. Every GT piece index must be a
nonnegative integer within the GT cache and identify the declared GT source;
inconsistent event/cache pairs fail before scoring. The scorer interprets each distance
with the existing integer same/next convention; event distances must already be
in winding units, with the standard perpendicular calibration. Missing reference
IDs and ambiguous unsigned targets are excluded and counted in coverage.

Scoring transports both fibers and the connecting segment across the image-up
seam. It aligns each connected component by an integer median offset and reports
both this score and a single-global-gauge score. L1 and exact correctness are
measurement-based reference consistency, not direct truth labels for every fiber.
Known-sign violations are reported separately for parallel and perpendicular
constraints. Empty coverage produces zero counts and null accuracy/L1 values.

Split runs with cross-event scoring also solve the whole-fiber graph to check
that identity piece lookup gives exactly the same scores as direct lookup. This
extra solve is included in total case time, not the split solver's certificate
time. Connectivity diagnostics count child components of that whole-fiber graph.
Boundary jumps across disconnected components are unidentifiable and are counted
separately, not interpreted as fiber errors.

## Outputs and baseline comparisons

- `manifest.json`: resolved settings, SHA-256 of consumed inputs, flow module
  hashes, Python/NumPy/SciPy versions.
- `results.json` and per-case `metrics.json`: certificate, internal signs, GT
  scores if requested, split diagnostics and timings.
- Per-case `solution.npz`: labels, graph, flow and seam transport; split cases
  also store piece bounds, parents and endpoint-to-piece maps.
- Cross-event cases write `gt_events.csv` and `gt_violations.json`.

Baseline comparisons first require identical source ordering, graph arrays and
piece topology, then equivalent component partitions. They report raw label
changes and changes after removing per-component integer gauge shifts. A
mismatched graph is reported as non-comparable, not silently aligned. Comparisons
are diagnostic; differences do not fail the run. The solver's own certificate
checks still enforce optimality for the configured objective.

Input hashing verifies reproducibility but does not recompute the original AFC
measurements. This runner does not add geometry fitting, fiber disablement,
uncertainty optimization or GT forcing.

## Tests

```sh
PYTHONPATH=vesuvius/src python -m pytest -q --noconftest -o pythonpath=vesuvius/src vesuvius/tests/test_fiber_min_flow_evaluation.py vesuvius/tests/test_fiber_min_flow.py vesuvius/tests/test_fiber_min_flow_split.py
```

The evaluation tests exercise seam transport, component alignment, empty coverage,
unknown signs, gauge-independent boundary diagnostics, baseline compatibility,
relative paths, output protection, and a solve without GT or baseline inputs.

## Optional historical audits and exports

All historical flow diagnostics are available through an optional `audit` object
on each case. They do not change the primary solve or overwrite saved solutions:

```json
"audit": {
  "details": true,
  "sign_analysis": true,
  "rounded_baseline": "/path/to/rounded/case/solution.npz",
  "measurements": {
    "database": "/path/to/original.afc.sqlite",
    "stage1": "/path/to/normal-orientation/solution.npz",
    "sign_cosine_min": 0.1
  }
}
```

Every field is optional; omit the object to retain the standard runner behavior.
Paths are relative to the experiment config. These fields apply independently:

- `details` exports internal violation CSV/JSON, per-source counts and rankings,
  reversal/missing-next-step counts, seam involvement, excluded unsigned-target
  diagnostics, GT violation CSV/JSON and all scored GT events. GT summaries include
  affected ordinary/GT fibers, violation steps, and normal-cosine ranges where
  supplied. Split runs also export every boundary with source/piece identity,
  arclength, raw jump and an explicit identifiability flag, plus GT piece counts.
  Fractional measured targets and residuals are retained without integer truncation.
- `sign_analysis` runs the historical hard-sign constrained L1 optimization and
  the lexicographic optimization minimizing total sign deficit at the original
  L1 optimum. The first reports additional L1 cost or an explicit contradictory
  cycle with source, piece and event IDs. The second preserves the original L1
  cost. Both handle fractional-expanded graphs, one-sided penalties and split
  endpoints. Integer potentials and objective agreement are checked; an invalid
  rounded LP result raises an error. Feasible alternatives are saved separately
  and scored against the configured reference, with optional detailed exports.
  Their NPZs contain labels/topology, not the primary solution's flow certificate.
  These extra optimizations can take substantially longer than the flow solve.
- `rounded_baseline` cross-compares a saved rounded result with this fractional
  case. Both assignments are evaluated under both objectives, checking each
  optimum against the other assignment. For a fractional target `m=floor(m)+a`,
  edges with targets `floor(m)` and `ceil(m)` carry weights `w*(1-a)` and `w*a`;
  for integer labels their combined penalty equals `w*abs(difference-m)`.
  The comparison checks that equality on both assignments. Changed source IDs
  (and piece bounds for split runs) mean labels changed between these runs after
  subtracting a modal offset per connected component. Reports include objective
  values, reference scores, internal signs, arc counts, available certificates
  and timings, and solution hashes. The baseline must match the configured input
  and rounded graph; unlike the standard baseline comparison, the two graphs
  need not have the same expanded arcs.
- `measurements` restores the source-AFC audit: join cached event IDs to original
  rows, verify source endpoints, recompute parallel/perpendicular classification,
  select the corresponding distance, and derive the sign using the original
  normal solution converted to the inward convention. Compare those values to
  the cache. A mismatch raises an error. Missing source files are explicitly
  reported as unavailable with zero verified rows; hashing is not treated as
  measurement verification. This checks the original raw measurements, before
  experiment-specific winding scale and perpendicular calibration.

To audit existing results without rerunning the primary solve:

```sh
PYTHONPATH=vesuvius/src python -m vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_audit \
  --config /path/to/local-experiment.json \
  --source /path/to/existing-results \
  --output /path/to/new-audit-results
```

The external config supplies the datasets, split settings, evaluation inputs and
optional audits for each named case. Saved geometry transport, graph, topology
and component partitions are validated against it. Historical rounded snapshots
without `flow_event_index` are supported; the graph and transport still must match.
Input hashes are checked against a source manifest when available, and all consumed
inputs are hashed before and after standalone auditing. Output must be new or empty.
The config's `output_root` is unused by this standalone command. No machine-local
config, dataset or generated result belongs in the repository.

Fresh solves write `case/audit/summary.json` and optional exports; standalone
runs write `case/summary.json` and exports in their separate output directory.
Internal diagnostics preserve saved component gauges and report cross-component
comparisons as unidentifiable. GT diagnostics retain the standard per-component
and single-global-gauge scores. The primary solver and default metrics are unchanged.

### Source-verification validation

The source-AFC audit was exercised on the preserved Paris4 `bbox_gt_solve_v3`
oriented constraint database and normal solution. Its tensor cache is identical
to the cache used by the five small benchmark cases. All five configured CLI
audits verified 46 rows: signs and parallel/perpendicular classifications matched
exactly; maximum distance roundoff was `1.0734044453286629e-07`, within the existing
cache tolerance. Temporary copies with independently altered signs, classification,
distance, endpoint identity, or a deleted event were all rejected. Original input
hashes were unchanged. This validates the feature on real source data; it does not
establish provenance for the separate large-case AFC database that is unavailable.
