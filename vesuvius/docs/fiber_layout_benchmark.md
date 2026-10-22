# Replaying the three-arm fiber layout benchmark

The versioned implementation is
`vesuvius.neural_tracing.fiber_trace_3d.fiber_layout_benchmark`. It owns validation,
workflow, managed solver launches, scoring, and old/new comparisons. It reuses
the existing optimizer, geometry, reference scorer, and GPU-manager launcher;
it does not implement another solver or change model defaults.

This is a **prepared-input replay**, not parameter tuning or native constraint
extraction. The three arms are `free` (no GT in training), `joint` (GT forced
enabled), and `selectable` (GT enablement optimized). They are evaluated against
the same frozen, independent GT solve. Each arm starts from its saved *initial*
state, never its historical fitted result.

## Portable entrypoint

With the repository package and its existing dependencies available:

```sh
python -m vesuvius.neural_tracing.fiber_trace_3d.fiber_layout_benchmark \
  --config benchmark.json --runtime runtime.json --action prepare
```

Actions:

- `prepare` (default): validate/configure inputs and write a provenance manifest;
  no GPU probing or allocation.
- `validate-reference`: re-score the three historical checkpoints, checking
  signs/counts, L1/correctness, checkpoint hashes, objective and GT retention
  against recorded reports. No GPU allocation.
- `run`: validate historical scoring, launch three fresh solves concurrently,
  score each result, and write `comparison.json` and `report.md`.
- `score`: re-score completed results with the same manifest, without optimizing.

`run` refuses existing solve directories. Use a new output root for a new run;
there is no automatic resume or historical-result reuse. Failures are surfaced
in `status.json` and per-arm logs. Already-running sibling solves drain through
their managed launcher on an arm failure; their grants are not abandoned.

## Configuration versus local execution options

`benchmark.json` contains dataset/config/reference bindings and the explicit
metric protocol, as illustrated in `examples/fiber_layout_benchmark.json`.
`runtime.json` contains the output directory, one distinct device per arm, CPU
scoring threads and GPU-manager settings. No host paths are baked into Python.
Top-level bindings are resolved relative to their JSON file. Prepared solver
configs retain their existing API and must contain absolute tensor/initializer
paths; relocate those bindings explicitly when moving data to another host.

The prepared solver config can be plain JSON or wrapped under
`fiber_implicit_winding`. Only `output_root`, `devices`, and `gpu_manager` are
replaced. Scientific settings, iteration budgets, batch sizes, seeds, tensors,
samples, and initial parameters are preserved. GT source identities and forced
enablement policy must agree with the supplied solver config.

The benchmark requires nonexclusive grants. The solver's existing VRAM estimate
is used when configured, otherwise runtime settings must supply a VRAM budget.
The launcher acquires resources before CUDA work and releases them afterwards.
There is no benchmark option to bypass the manager. Busy GPUs wait for a grant;
the benchmark does not modify other jobs' reservations.

Input/config/source hashes are recorded and checked again at completion.
Outputs include resolved configs, a manifest, per-arm logs/reports, live status,
the full metrics JSON, and a compact old/new table. Nonexclusive timings are
operational solve times, **not isolated throughput measurements**.

## Metric compatibility

Protocol `frozen-gt-map-half-up-v1` preserves the original experiment's integer
label tie convention (`floor(N+.5)`), frozen reference, enabled-logit >=0 cutoff,
and single global integer-offset correction. It never rounds a constraint into
its own target. This explicit historical protocol is distinct from the generic
solver's lower-label tie convention and from the minimum-flow reference audit.
The reference includes whichever fibers the supplied frozen GT contains; none
are silently excluded.

Reported: objective and all saved loss terms, enabled percentages, parallel and
perpendicular GT L1/correctness/sign counts and percentages, GT retained arc
length, width comparison, and solve time. Ranking is summed perpendicular sign
violations divided by selectable-GT retained length fraction, with summed
perpendicular L1 as tie-breaker. Zero retention yields an invalid (`null`) score.
No new optimum is selected automatically. Exact-meeting/visualization metrics
are not part of this prepared fixed-length replay.

## Thin local approval wrapper

`/local/` is ignored by Git. A local script may set repository import paths and
default config filenames, then call the versioned module's `main(argv)`. Keep
solver calls, scheduling, manager logic, scoring, and reporting out of it.
Local choices do not need to be committed or imposed on another machine.

```python
from pathlib import Path
import os
import sys
repo = Path(__file__).resolve().parents[2]  # local/scripts/runner.py
roots = [str(repo / 'vesuvius/src'), str(repo / 'gpu_manager/src'), str(repo)]
sys.path[:0] = roots
os.environ['PYTHONPATH'] = os.pathsep.join([*roots, os.environ.get('PYTHONPATH', '')])
from vesuvius.neural_tracing.fiber_trace_3d.fiber_layout_benchmark import main
if __name__ == '__main__':
    main(['--config', str(repo / 'local/configs/benchmark.json'),
          '--runtime', str(repo / 'local/configs/runtime.json'), *sys.argv[1:]])
```

Approve that specific local script path, not an unrestricted Python prefix.
Approving its launcher is not blanket permission for unrelated workloads.

## Tests

From the repository root, using the existing environment:

```sh
python -m pytest -o pythonpath=vesuvius/src -q \
  vesuvius/tests/test_fiber_layout_benchmark.py \
  vesuvius/tests/test_fiber_gt_constraint_score.py
```

Tests mock GPU launches. Actual data replay requires the prepared tensors,
initializers/samples, frozen GT, cached cross-events, and historical reports;
these large artifacts are not bundled. No installation or bootstrap is done.

Real-data compatibility check (2026-10-22): replaying the historical accepted
candidate `a838956dd140d329e78f` through the new scorer reproduced all three
recorded perpendicular/parallel scores and sign counts (L1/correctness tolerance
1e-10). Summed perpendicular signs: 38; selectable GT retained length: 95.315808%;
rank score: 39.867469; summed perpendicular L1: 1.023098. This checks the scoring
API against real checkpoints, **not** convergence of new solves on main.
