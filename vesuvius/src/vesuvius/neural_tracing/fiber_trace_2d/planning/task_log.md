# Task log: arbitrary staged Fiberlet graph reduction

## Decisions

- Stage specifications are ordered and repeatable; offsets are relative to the
  selected bbox minimum in base XYZ coordinates.
- Every stage uses the initial storage layout. Analysis boxes and storage chunks
  are independent geometric concepts.
- Missing upper chunks fall through, while explicit empty upper chunks shadow
  lower data.
- Updates are monotone and limited by canonical Fiberlet ownership inside each
  processed half-open box.
- Existing macro merging remains an in-memory diagnostic. Persisting macros as
  ordinary Fiberlets would violate the established route-format contract.
- The independent review recommended persistent completed layers, but the
  current user requirement explicitly makes every derived stage cache
  temporary. Layers therefore use unique invocation roots, are never reopened,
  and are removed after reporting; no completion protocol is needed yet.
- Cold canonical source generation remains allowed. Staging never prunes or
  rewrites the initial cache.

## Deviations

- Macro-Fiberlet merging remains an in-memory simplification diagnostic. The
  ordinary Fiberlet route payload cannot encode a concatenated macro without
  resampling, so temporary overlays persist exact physical removals and unused
  anchor removal only. This preserves the established storage contract.
- The staged layers are intentionally deleted after the report. They therefore
  do not yet have a completion marker or cross-invocation reuse protocol.

## Validation

- The first Paris4 staged run exposed a storage-boundary case: an eligible
  Fiberlet's first endpoint was geometrically inside the analysis box, but its
  canonical owner was an adjacent storage chunk. The box writer now unions
  those canonical owners with geometrically intersected chunks before writing.
- Built with 32 jobs:

  `cmake --build volume-cartographer/build/ci-tests-clang-systemdeps --target vc_fiberlets test_fiberlet_storage test_fiberlet_paths -j32`

- `test_fiberlet_storage`: 28 test cases passed.
- `test_fiberlet_paths`: 87 test cases passed.
- `git diff --check`: passed.
- A hot Paris4 512-base bbox run used stages `256,0,0,0`,
  `256,128,128,128`, and `512,0,0,0` with the existing 128-base storage
  chunks and 32 threads. Cache preparation reused 512/512 anchor chunks and
  216/216 Fiberlet chunks.
- Stage wall times in the current Clang system-dependency build were 42.7 s,
  2.9 s, and 25.4 s. Counts were:

  | Scope | Initial | Stage 1 | Stage 2 | Stage 3 | Joint reduction |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Anchors | 4,383 | 4,184 | 4,143 | 3,368 | 23.16% |
  | All Fiberlets | 79,301 | 48,992 | 46,026 | 35,028 | 55.83% |
  | Interior Fiberlets | 48,415 | 18,406 | 15,440 | 4,470 | 90.77% |

- The invocation-local stage directory was absent after normal completion,
  confirming cleanup while the initial caches remained unchanged.
