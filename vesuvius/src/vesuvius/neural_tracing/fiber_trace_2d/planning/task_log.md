# Task log: concise replay progress and latest fiberlet optimizations

## Initial findings

- `fiberlet-replay` currently prints setup, extraction, cache-open, every cache
  chunk, every failure, both evaluator streams, cache statistics,
  visualization, and publication details unconditionally.
- `--stats` exists but is accepted only for the standalone `paths` command.
- `fiber-lets2` has three unmerged commits: path-search optimization, anchor
  extraction optimization, and their durable documentation.
- The current branch already requires byte-identical eager and float-cache
  replay, so that remains the merge correctness gate.

## Independent review

- Accepted: make concurrent evaluator accumulation explicitly monotone; keep
  100% reserved until requested output is published; test quiet/statistical
  output and final-line handling; measure actual CPU as well as wall time; merge
  cumulative specs/docs semantically; compare cache populations as well as the
  final replay.
- The review's shared-stage/sample-count findings describe an earlier benchmark
  task and do not apply to this output-and-merge task. The merged optimizations
  will still be checked for duplicate work through their existing diagnostics.

## Progress implementation

- Default replay now displays one 24-column bar. Concurrent fractions are
  clamped and accumulated independently; their minimum drives tracing progress.
- Tracing reserves the final one percent for publication. Visualization runs
  reserve twenty percent for the overview and per-failure artifacts, preventing
  a displayed 100% while requested output is still being written.
- `--stats` is now accepted by `fiberlet-replay` and restores all prior detailed
  rows. Default eager/local visualization extraction also suppresses its nested
  anchor and path progress callbacks.
- Validation: `cmake --build volume-cartographer/build -j32 --target
  vc_fiberlets` passed. A cold 500-base-voxel default replay printed only the
  progress display and terminal result; a 128-base-voxel `--stats` replay
  retained stage, chunk, evaluator, cache, and terminal rows.
