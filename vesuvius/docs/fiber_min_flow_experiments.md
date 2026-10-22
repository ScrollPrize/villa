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
  Connector transport cancels between prediction and reference.
- `cross_events`: additionally requires `events` (JSON path) and `dataset`
  (the separate GT dataset object or JSON path). Supports whole and split fibers.

Cross events are a list of records with `gt_source`, `test_source`, `gt_piece`
(index into GT cache), `gt_arc`, `test_arc` (original polyline arc lengths),
`signed_distance`, `family_delta`, and boolean `parallel`. Additional event fields
are preserved in the exported event table. The scorer interprets each distance
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
