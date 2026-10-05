# Joint branch / upstream integration

- Branch `fiber-normal-width-hv`: collector `4dcf278e2`, evaluation records
  `afe60bc6e`, villa3 merge `330207081`. Villa2 tip `29b2eb56f` was already
  ancestral; villa3 tip `be3b45d36` is included. Other worktrees untouched.
- Merge `origin/main` at `df50aa186` instead of rebasing already-merged local
  histories. Retain local tracer callbacks/history and upstream fixed-start
  annotation support; keep new native V4 tags and Python serialization roundtrip.
- Upstream's arclength-weighted meeting fusion is included intentionally as
  part of main, not claimed numerically equivalent to historical tracing.
- Preserve bootstrap native installation/compiler selection plus upstream
  portable interpreter discovery and CUDA kernel smoke check.
- Shared upstream remote settings retain the explicit legacy `cache.json`
  override. Rendering checks that its requested cache matches the now-global
  Volume cache rather than silently dropping the request.
- Python validation: 130 passed / 6 subtests (bootstrap, inference preprocessing,
  fiber loader/parser, collector). Additional broad rerun was interrupted after
  stalling; no success claimed for that rerun. Tagged serializer regression
  rerun separately. No GPU jobs launched, no installation performed.
- Approved `run_fiber3d_cooperative_tests`: `merge-native-build` and
  `merge-native-tests` passed (4 executables, 12.08s); annotation build/tests
  passed (4 executables, 0.17s). Build is RelWithDebInfo, existing dependencies,
  no downloads, CPU only. First build exposed/then fixed the changed NewFromUrl
  signature. Build/test logs under cooperative-tests/20261005_120411_149917,
  20261005_120449_195776 and 20261005_120521_058398.
- Full GUI/application build, full repository suite and GPU validation are not
  part of this focused merge validation. Raw downloaded data/perf.data remain
  untracked. No new training launched.
- Next: bake width-annotated normals with shared core `fiberDisplayField`, add
  normal six-channel / width input-voxel / inherited binary H/V training heads
  and consistent geometric augmentation. These features are not implemented
  by the integration merge itself.
