# Task Log: direct replay visualization manifests

## Findings

- Version-2 publication already writes one complete local manifest below each
  failure visualization directory, but the viewer unnecessarily requires the
  aggregate root plus `--index` before it will parse that manifest.
- The viewer's prior strict version-1 reader was removed rather than retained,
  making existing single-visualization artifacts unreadable.
- A no-`--vis` aggregate contains complete evaluator traces and failure records
  but not failure-local anchor/path extraction, so it cannot provide the local
  diagnostic layers after the fact.

## Plan Review

- Direct local manifests already contain the required visualization data, but
  their immutable generation paths cannot support useful reload after a rerun.
  Stable atomically replaced direct aliases are required.
- Direct parsing must validate local identity, sources, prediction binding,
  failure/tube containment, crop, hashes, geometry, and path safety without
  depending on aggregate-root cross-checks.
- Restoring version-1 compatibility requires a dedicated schema parser and
  normalization into the current segmented display model, including failed,
  nonfailure, and optional graph-route cases.
- Tests must cover standalone direct manifests, stable alias publication and
  reload, aggregate rejection guidance, path/hash escape protection, and the
  version-1 variants.

## Deviations

- The repository suite does not contain a preexisting version-1 nonfailure
  artifact fixture. The restored parser handles the nonfailure shape, but the
  compatibility validation used the user's real failed version-1 artifact,
  which is the visualization workflow at issue, rather than adding a large
  synthetic copy of the retired schema.

## Validation

- Independent plan review completed and drove stable alias/reload semantics and
  standalone local validation.
- Built with 32 jobs:
  `cmake --build volume-cartographer/build -j32 --target vc_fiberlets test_fiber_replay`.
- `test_fiber_replay`: 4 cases passed, including direct stable alias publication
  and obsolete-alias cleanup.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest -q vesuvius/tests/test_view_fiber_presence.py`:
  57 tests passed.
- `ruff format --check` and `ruff check` passed for the viewer and its tests.
- Loaded the real
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/data/workdir3/fiber_replay.json`
  through the default strict version-1 path: status `failure_with_postroll`, five
  anchor stages, one greedy segment, and one fiberlet segment.
- Viewer help exposes direct `--replay REPLAY` and no `--index` argument.
