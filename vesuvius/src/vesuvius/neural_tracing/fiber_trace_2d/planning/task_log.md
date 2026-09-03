# Task Log

## Scope

- Continue each directed endpoint replay after failures until the full in-crop
  reference run has been evaluated.
- Retain every failure event and report failure-free traced spans.
- Reuse the existing deterministic replay restart implementation and shared
  anisotropic threshold measurement.

## Decisions

- The first seed remains restricted to the endpoint seed window.
- A directional case is whole-run complete once replay reaches its reference
  end, even if it contains failures; failure-free status is reported separately.
- Ordinary replay defaults remain unchanged.

## Independent Review

- Define credited length from actual seeded segment intervals rather than the
  replay completion cursor; missing seed gaps must not become successful length.
- Distinguish whole-run evaluation completion from a failure-free direction.
- Report failure density, failed-span length, seeded-span length, restart
  settings, and complete per-failure location/threshold diagnostics.
- Bump the incompatible JSON schema to version 2 and retain the historical
  first-failure run record.
- Keep direction-ablation reference handling unchanged and create the canonical
  external-data record only from a committed implementation.

## Validation

- Release `vc_fiber_trace_chunk` and
  `test_fiber_reference_replay_benchmark` build successfully.
- Focused benchmark tests: 4 test cases passed.
- The broad `test_fiberlet_paths` executable still reports the pre-existing
  bit-exact prepared-scoring failures at line 414; its continuation replay
  regression is compiled, and the benchmark uses the unchanged ordinary reset
  path already covered by that suite.
