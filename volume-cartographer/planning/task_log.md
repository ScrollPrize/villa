# Task Log

2026-09-14: Created fix/quadsurface-cache-invalidation-race using git checkout
-b from origin/main at 3b398f7cc. Previous read-only diagnosis on PR #1596
reproduced a normal-cache reader SIGSEGV under GDB, and 2/20 full smoke runs
failed during volume.open with four CPUs. The exact CI C3-only timeout has not
been reproduced or attributed conclusively. No external dependency installs.

Scope: derived-cache invalidation versus rendering. Concurrent geometry writes,
point eviction and arbitrary channel I/O remain externally synchronized duties.

Independent plan review requested explicit lock ordering and matching mask/flag
snapshots, cloned render baselines, and concurrent-operation exclusions; all
incorporated. Independent code review found no implementation defect and caught
a test hole outside the cropped view. The test now covers the full source grid,
both components and interior invalid points, with explicit NaN assertions.

Implemented cache resets under _cacheMutex and reference-counted normal/mask
snapshots. Existing public signatures, numerical operations and rendering
scheduling are unchanged. Snapshot acquisition does not copy full matrices or
hold the mutex through pixel sampling.

Validation (Linux GCC QuickBuild, existing system dependencies, -j32):

- Built VC3D and seven focused QuadSurface test targets.
- ctest --test-dir volume-cartographer/build -R
  '^test_quadsurface_(cache_concurrency|basics|components|more|fixtures|extras|final)$'
  --output-on-failure -j32: 7/7 passed (OMP_NUM_THREADS=2).
- ctest --test-dir volume-cartographer/build -R
  '^test_quadsurface_cache_concurrency$' --repeat until-fail:100
  --output-on-failure: 100/100 passed, 7.67 seconds total.
- Negative control: linked the same regression test against the previously
  compiled pre-fix QuadSurface object. It exited with SIGSEGV (139), including
  after correcting the test viewport. This is schedule-dependent evidence,
  not a guarantee that every unpatched run will fail.
- Reused smoke_offscreen.main() through a temporary diagnostic wrapper that
  captures process exit/log information. OMP_NUM_THREADS=4, taskset -c 0-3,
  20 sequential runs: 20/20 passed after the fix versus 18/20 before.
- New test includes the committed PHerc0172 fixture 20241113090990, generated
  all-valid components, an interior hole with strict validity, optional normals
  and depth offsets. Mat row bytes (including NaN payloads) match cloned serial
  baselines during concurrent eviction. Existing tests cover sequential geometry
  invalidation and point unloading.
- No-PCH test compilation caught a missing opencv2/core.hpp include; added it.
  Recompiled without PCH at -O2, linked against vc_core and ran both test cases:
  passed. Final seven-target focused CTest run also passed after the include fix.

No dependency installation, full-scroll replay, platform-wide testing, or
performance claim. The observed CI C3-only timeout remains unattributed; this
fix targets the separately reproduced cache-reader crash, not every RPC timeout.

Independent final review confirmed the test framing correction; no outstanding
implementation findings. PR draft prepared for approval; publication is held.
