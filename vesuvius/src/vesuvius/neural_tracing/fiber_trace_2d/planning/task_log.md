# Task log: integrate current main

## 2026-09-08

- Worktree has no tracked modifications before integration; unrelated
  untracked scratch/build artifacts are left untouched.
- Fetched `origin/main` at `d8c5f488a`.
- The histories contain 48 main-side and 202 branch-side commits after their
  common ancestor `8b59dde58`.
- Chose a merge rather than rebasing 202 commits because the branch is already
  published and contains merge history; this avoids unnecessary history
  rewriting while putting it on the current main basis.
- Committed the active planning reset separately as `c84f25860` because all
  four active planning files also differ on `origin/main`; carrying them as
  local edits would obstruct the merge.
- Independent plan review required a complete overlap audit, two-parent
  semantic diff checks, explicit CMake/test targets, a deterministic Fiberlet
  smoke check, dirty-worktree protection, and deliberate handling of durable
  versus active planning files. The plan now includes those checks.
- The merge produced four textual conflicts. Additive Python entry points and
  changelog histories were both retained. FiberTrace uses upstream's
  span-bounded endpoint configuration throughout target acceptance and fusion.
  ChunkCache combines branch-side shared-buffer pin protection with upstream's
  parked-blocking-reader protection.
- The combined cache contract required adapting four new upstream assertions:
  upstream expected a returned blocking-read chunk to leave the cache
  immediately, while the branch intentionally keeps externally leased byte and
  typed payload chunks resident. The merged tests now verify successful reader
  delivery, reader-registration cleanup, and lease residency; the existing
  lease-pressure test verifies later eviction after release.
- Regenerated the existing Release build with
  `cmake -S volume-cartographer -B volume-cartographer/build`.
- Successfully built `vc_fiber_trace_chunk`, `vc_fiberlets`, `VC3D`, nine
  focused core/Fiberlet test targets, and six focused VC3D test targets.
- Focused core results: `test_fiber_trace3d`, `test_chunk_cache` (99 cases),
  Lasagna normal/alignment/optimizer/view tests, `test_fiberlet_storage`, and
  `test_fiberlet_crop_trace` pass. `test_fiberlet_paths` retains its pre-merge
  295 failures, all at unchanged line 414 in the legacy prepared-scoring
  bitwise comparison helper; no other assertion site failed.
- The six upstream VC3D focused binaries pass 94 Qt test cases in total.
- `git diff --check` reports only whitespace already present in `origin/main`
  (`CVolumeViewerView.cpp/.hpp` and `test_post_process.cpp`); the branch-side
  merge delta introduces no additional whitespace errors.
