# Trace recovery log

2026-10-22: Created trace-recovery from origin/main 862142652; recovery source fiber-exp5 at 8ab5ec0cc. Untracked console log preserved; caches untouched. Main already includes CPU tracing, reverse checks and chunk route/cache APIs, so wholesale replacement is inappropriate. Legacy dependency script absent; use current build documentation. Historical data remains on source branch, outside recovery commits.

## Implementation

- Retained main's shared replay core, route extraction, case queue, live bridge,
  cache configuration/identity and Python lifetime ownership. No old monolithic
  evaluation runner substituted for these APIs.
- Recovered five-point angular initial match, custom ellipse, joint joins,
  bounded seed fitting, normal-aware conflict filtering, optional CUDA chunk
  construction and online fixed-plane refinement, native RAM pooling, and
  portable shared tensor/scoring helpers. New behavior is opt-in.
- Extended existing callable evaluator and native replay-set for chunk settings.
  Chunk CLI uses the full requested shared worker budget even for few reference
  cases, so seed jobs are not artificially capped by reference count.
- Added reusable uncached single-chunk worker scaling and isolated filter timers,
  plus provenance/hash-checked legacy-cache import into a new directory.
  Existing replay_metrics remains the single shared historical metric implementation.
- Independent review caught late GPU admission after host pinning; moved admission
  before all host chunk preparation. Cancellation/error releases its token.
  Also fixed output dtype checks and early GPU/refinement validation.
- Tests caught an overly strict GPU device-count check on CPU defaults; corrected
  to permit zero only when no GPU backend is selected. No main defaults changed.

## Validation

Existing build: `volume-cartographer/build/fiber-evaluation`, RelWithDebInfo,
GCC13; CUDA12.8 uses GCC12 host compiler and architecture80. No installations.

- Native suites: trace3d, prediction_channels, reference_replay_benchmark,
  replay_metric, width_scoring, reverse_consistency, reference_replay_core,
  chunk_trace and case_queue passed. Pooling tests passed (concurrent channel
  bundle reads, explicit level mapping, readonly behavior and dtype rejection).
- Python recovery/evaluation/chunk/trace suites: 20 passed, 1 optional real-data
  test skipped. Local-socket tests rerun outside sandbox after sandbox denial.
- CUDA shared plugin and standalone benchmark compiled. Host-only shared
  refinement regression command: 7 tests passed; no GPU resources used.
- Build-tree Python test bootstrap removes this machine's older editable package
  finder using temporary `/tmp/vc-trace-recovery-test/sitecustomize.py`, with
  PYTHONPATH pointing to build/python. No installed package changed.
- CUDA OFF configuration/build checked separately. `git diff --check` clean.

Representative commands:

```sh
cmake --build volume-cartographer/build/fiber-evaluation --target test_fiber_chunk_trace test_prediction_pooling vc_fiber_trace vc_fiber_trace_metric -j 8
ctest --test-dir volume-cartographer/build/fiber-evaluation --output-on-failure -R '^test_fiber_(trace3d|chunk_trace|replay_metric|reference_replay_benchmark|width_scoring|reverse_consistency|prediction_channels|reference_replay_core|case_queue)$' -j 4
PYTHONPATH=/tmp/vc-trace-recovery-test:/home/hendrik/villa3/volume-cartographer/build/fiber-evaluation/python /home/hendrik/.venv_las/bin/python -m pytest -q volume-cartographer/python/tests/test_fiber_recovery.py volume-cartographer/python/tests/test_fiber_evaluation.py volume-cartographer/python/tests/test_fiber_chunk_trace.py volume-cartographer/python/tests/test_fiber_trace.py
volume-cartographer/build/fiber-evaluation/bin/benchmark_fiber_chunk_cuda --host-refinement-tests
```

## Explicit limitations / exclusions

- GPU device execution and the two historical zero-error cohort reruns are NOT
  completed by this integration task. Build/synthetic tests do not establish
  historical numerical parity. No current quality/performance result claimed.
- Old caches require explicit provenance attestation before copy-import; no
  cache compatibility bypass or mutation of historical inference/path files.
- CUDA remains limited to its recovered fixed beam/lattice/capacity configuration;
  unsupported settings fail. Online refinement is GPU-only. No embeddings or
  CPU seed-fit option in GPU construction.
- Pooling degenerate eigenspaces can differ with LAPACK vs Eigen-only builds;
  document solver policy in source identity. macOS/arm64 runtime not tested.
- Historical experiment configs, sweeps, reports, data, model paths and launchers
  remain on fiber-exp5/external storage, not restored or committed on this branch.
  Unrelated experimental BP/skeleton families are not included in tracing recovery.
  Existing untracked console log remains untouched.
