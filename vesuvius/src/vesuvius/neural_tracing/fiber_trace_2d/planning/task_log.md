# VC3D GitHub CI Acceleration

## Goal And Constraints

- Target GitHub pull-request feedback latency: ideally less than one minute.
- Every configured target must compile in CI even if it is not executed by the
  test job. This coverage may be split into separate compile-only jobs.
- Preserve Ubuntu/macOS and amd64/arm64 portability and do not use numeric
  shortcuts such as `-ffast-math` or reduced precision.
- Measurements use the exact Linux CI dependency image with a four-CPU
  container limit and Ninja `-j4`, matching the public GitHub runner CPU count.

## Existing GitHub CI

- Workflow: `.github/workflows/vc3d-ci.yml`.
- Image:
  `ghcr.io/scrollprize/vc3d-deps/linux:sha-0c371b1d472c5281b703d65517e980d945da693f`.
- GCC and Clang matrix jobs each configure, run an unrestricted `ninja` build,
  execute CTest, and run the VC3D offscreen agent-bridge smoke test.
- A prior GitHub GCC job spent about 10m46s in the combined configure/build/test
  step; the complete job took about 12m13s.
- `ci-release-tests-gcc` uses Release `-O3`, GCC `-flto=auto`, PCH, debug info,
  frame pointers, and asynchronous unwind tables.

## Cold Release/LTO Baseline

- Container CPUs: 4; Ninja jobs: 4; compiler cache: cold/disabled locally.
- Build suffix: `-docker-bench-gcc`.
- Configure time: approximately 32 seconds.
- Clean `ninja all`: approximately 780.5 seconds (13m00.5s), 1,812 Ninja
  build edges, successful.
- CTest: 117/117 passed in 121.50 seconds.
- VC3D offscreen smoke: passed in 1.52 seconds.
- LTO linking dominates the late critical path. The VC3D executable link took
  approximately 174 seconds; individual test and benchmark links commonly took
  9-30 seconds, with at least one test link around 46 seconds.
- The build compiles applications, CLI tools, tests, flatboi/PaStiX, benchmark
  executables, and an E2E executable that is not registered in default CTest.
- Local Docker image/container metadata operations are unusually slow due to
  the host overlay2 configuration. These outer Docker delays are excluded from
  the in-container configure/build/test measurements.

## Current Hypothesis

- Use the existing `QuickBuild` mode (`-O1`, PCH, no LTO) as the first
  candidate configuration for executables used by the test job.
- Preserve full-target GCC and Clang compile jobs. A later test-dependency-only
  build may reduce test feedback latency without reducing compile coverage.
- Keep a production Release/LTO job outside the fastest pull-request feedback
  path if its compile/link cost remains dominant.

## Cold QuickBuild Full-Target Result

- Candidate: existing `QuickBuild` mode (`-O1`, PCH, no LTO), with
  `VC_TESTING=ON` and `VC_USE_SCCACHE=OFF`.
- Container CPUs: 4; Ninja jobs: 4; compiler cache: cold/disabled.
- Build suffix: `-docker-quickbuild-gcc`.
- The generated graph remained 1,812 Ninja edges, matching the Release/LTO
  baseline and retaining the same configured-target compile coverage.
- Clean `ninja all`: 584.86 seconds (9m44.9s), successful.
- Improvement over the 780.5-second Release/LTO baseline: 195.64 seconds,
  approximately 25.1%.
- The LTO link tail disappeared, but compiling the full VC3D GUI, GUI-heavy
  tests, applications, PaStiX/flatboi, and all other tools still dominates.
- A cold full-target build cannot approach the one-minute feedback goal.
  Subsequent experiments should preserve the full build as a parallel required
  gate while measuring smaller test dependency closures and warm sccache.
- CTest validation: 117/117 passed in 120.72 seconds.
- The three explicitly live network tests consumed 75.83 seconds:
  `test_pherc0172_live` 0.77 seconds, `test_volume_live_s3` 48.29 seconds, and
  `test_normal_grid_live` 26.77 seconds. The remaining 114 tests therefore
  consumed approximately 44.9 seconds in the existing serial CTest run.
- VC3D offscreen agent-bridge smoke validation: passed in 1.76 seconds.
- Candidate correctness validation succeeded. The build remains unsuitable as
  a single sub-minute cold job, but deterministic test execution is already
  below one minute once live network checks are assigned elsewhere.

## QuickBuild Compile-Cost Findings

- Ninja recorded approximately 2,164 CPU-seconds of C++ compilation work.
  Against 584.86 seconds wall time on four CPUs, ordinary C++ compilation is
  already close to saturating the available cores; raising Ninja parallelism
  alone is unlikely to produce a large improvement.
- QuickBuild still applies full GCC `-g3` debug information and
  `-fno-eliminate-unused-debug-types` to every translation unit. It produced
  approximately 8.06 GiB of object files.
- `LineAnnotationController.cpp.o` is 69,077,312 bytes with the existing
  QuickBuild flags but 2,640,744 bytes after stripping debug sections, a 96.2%
  reduction for that representative large translation unit. This is not yet a
  compile-time benchmark, but it establishes that DWARF generation and object
  I/O are substantial avoidable work for a test-only build.
- PCH is already enabled. The VC3D PCH took about 15.4 seconds to build, while
  expensive individual C++ translation units still took 10-42 seconds.
- The compile database has 1,085 commands for 1,015 unique source paths; only
  44 sources are compiled more than once. Extracting duplicated VC3D test
  sources into shared targets remains worthwhile engineering cleanup but is
  not large enough to explain most of this cold-build time.
- The next direct compiler experiment should be a test-only `-g0`, no-LTO
  configuration. The selected first experiment instead uses `-g1` to retain
  function and line information for VC3D crash reports while omitting locals,
  unused types, and macro debug data. A following isolated experiment can use
  `-O0`; test runtime must then be remeasured because the current non-network
  suite has only about 15 seconds of margin below one minute.
- The GitHub workflow already enables the GHA sccache backend. Cache-hit timing
  is a separate warm-build experiment, not a substitute for reducing cold
  compiler work.

## Reduced-Debug QuickBuild Experiment

- Change QuickBuild debug generation from GCC/Clang full `-g3` plus
  compiler-specific type retention to `-g1` while retaining frame pointers and
  asynchronous unwind tables.
- Compare a clean, no-sccache, four-CPU `ninja all -j4` build against the
  584.86-second full-debug QuickBuild using the same source graph and image.
- Configure: 36.37 seconds.
- Clean `ninja all -j4`: 365.08 seconds (6m05.1s), successful, with the same
  1,812-edge graph.
- Improvement over full-debug QuickBuild: 219.78 seconds (3m39.8s), a 37.6%
  reduction and 1.60x speedup.
- Improvement over the original Release/LTO baseline: 415.42 seconds
  (6m55.4s), a 53.2% reduction and 2.14x speedup.
- Recorded C++ compilation work fell from approximately 2,164.1 CPU-seconds to
  1,311.7 CPU-seconds, a 39.4% reduction.
- Object payload fell from approximately 8.06 GiB to 0.377 GiB (95.3%). The
  complete build directory fell from approximately 13 GiB to 1.6 GiB.
- Representative `LineAnnotationController.cpp.o` fell from 69,077,312 bytes
  to 10,904,584 bytes while retaining `-g1` function and line information.
- Validation: 117/117 CTest tests passed in 119.55 seconds; the VC3D offscreen
  agent-bridge smoke test passed in 1.46 seconds.
- Result: keep the reduced-debug QuickBuild change. It materially improves cold
  compilation without changing the target graph or test behavior.

## Remaining Low-Effort Candidates

1. Benchmark `-O0` instead of `-O1` in the reduced-debug test build. This is
   the next likely cold-compile reduction, but all tests must be timed again
   because runtime will increase.
2. Benchmark omitting `-ffunction-sections` and `-fdata-sections` for the
   test-only build. Fine-grained linker garbage collection is not important for
   disposable test binaries, and emitting thousands of ELF sections adds some
   assembler and linker work. Expected impact is smaller than reduced DWARF.
3. Benchmark modest Ninja oversubscription (`-j6` on four CPUs). The reduced-
   debug build averaged about 3.6 utilized cores, so the theoretical remaining
   gain is limited, but extra ready work may reduce uneven-TU and link-tail
   idle time.
4. Benchmark Clang QuickBuild. Existing GitHub Release/LTO history favored
   Clang over GCC, but that does not establish its reduced-debug cold-build
   result.
5. Inspect Qt AUTOGEN parallelism. VC3D AUTOGEN was a visible serialized edge;
   explicitly matching AUTOGEN parallelism to the worker may remove a smaller
   setup bottleneck if CMake is not already doing so.

- `-g0` could remove more work, but is not the next recommendation because
  `-g1` preserves actionable function/line crash reports.
- Unity builds are not classified as low effort: they risk source collisions,
  high peak memory, and materially coarser incremental rebuilds.
- The larger gains still require workflow structure: build test dependencies
  separately, run full compile-coverage shards concurrently, and exploit warm
  GHA sccache entries.

## Five-Candidate Benchmark Batch

- Baseline: GCC QuickBuild, `-g1`, `-O1`, PCH, no LTO, no sccache, four-CPU
  container, `ninja all -j4`: 365.08 seconds.
- Benchmark five isolated clean configurations with separate build suffixes:
  `-O0`; no function/data sections; Ninja `-j6`; Clang; and
  `CMAKE_AUTOGEN_PARALLEL=4`.
- Compiler-launcher scripts in the ignored build tree append the experimental
  override after project flags. They are benchmark harnesses, not proposed
  repository tooling.
- Run configurations sequentially in one persistent container to avoid
  cross-build CPU/I/O contention and exclude Docker lifecycle delays.
- Deterministic test selection excludes exactly `test_pherc0172_live`,
  `test_volume_live_s3`, and `test_normal_grid_live` because local network
  timing is unreliable. Baseline: 114/114 tests in 44.46 seconds.

### Results

| Candidate | Clean compile | Change vs baseline | 114 tests | Smoke | Decision |
|---|---:|---:|---:|---|---|
| GCC `-g1 -O1 -j4` baseline | 365.08 s | - | 44.46 s | pass, 1.46 s | baseline |
| GCC `-O0` | 322.80 s | -42.28 s (-11.6%) | 45.68 s | **fail**, 11.03 s | reject globally |
| GCC without function/data sections | 372.52 s | +7.44 s (+2.0%) | 51.13 s | pass, 1.39 s | reject |
| GCC `-j6` on four CPUs | 376.42 s | +11.34 s (+3.1%) | 44.17 s | pass, 1.39 s | reject |
| Clang `-g1 -O1 -j4` | **293.93 s** | **-71.15 s (-19.5%)** | **40.18 s** | pass, 1.64 s | keep candidate |
| GCC with AUTOGEN parallel 4 | 343.80 s | -21.28 s (-5.8%) | 44.89 s | pass, 1.48 s | keep candidate |

- Configure times were stable: GCC candidates 34.67-35.74 seconds; Clang
  37.61 seconds. They are reported separately from clean compile time.
- `-O0` passed every unit test but the VC3D smoke workflow timed out after 10
  seconds in `viewer.set_overlay` and then lost the process. Do not weaken the
  smoke timeout to accept slower test code. A future hybrid could keep VC3D at
  `-O1`, but that is no longer a single low-effort global flag.
- Removing function/data sections regressed both compilation and deterministic
  test startup, so retain fine-grained sections and linker garbage collection.
- Ninja `-j6` initially advanced generators faster but ultimately lost to CPU
  contention under the four-core quota. Retain `-j4`/native four-way scheduling.
- Clang is the strongest valid individual candidate: it improves clean compile
  and deterministic test runtime while passing smoke.
- Explicit AUTOGEN parallelism is independently useful. The next benchmark
  should combine Clang with AUTOGEN parallel 4; the individual gains cannot be
  assumed to add linearly.

## Clang Follow-Up Batch

- Benchmark Clang `-g1 -O1 -j4` with `CMAKE_AUTOGEN_PARALLEL=4` to measure the
  combined valid candidates.
- Repeat the combined configuration with Ninja `-j6` under the same four-CPU
  quota.
- Inspect Clang's effective debug mode and existing frontend/linker choices.
  Benchmark only materially distinct low-effort options, including Clang
  `-O0`; do not repeat aliases or flags the current configuration already uses.
- Clang baseline objects total approximately 0.227 GiB, compared with 0.377
  GiB for GCC reduced-debug QuickBuild. Representative
  `LineAnnotationController.cpp.o` is 7,389,296 bytes under Clang versus
  10,904,584 bytes under GCC.

### Results

| Candidate | Clean compile | 114 tests | Smoke | Decision |
|---|---:|---:|---|---|
| Clang `-O1 -j4`, default AUTOGEN | 293.93 s | 40.18 s | pass, 1.64 s | prior baseline |
| Clang `-O1 -j4`, AUTOGEN 4 | **286.48 s** | **39.56 s** | pass, 1.37 s | keep |
| Clang `-O1 -j6`, AUTOGEN 4 | 296.70 s | 40.77 s | pass | reject `-j6` |
| Clang `-O0 -j4`, AUTOGEN 4 | 207.20 s | 42.49 s | **fail** at 10/30/120 s | reject `-O0` |

- Combining Clang with AUTOGEN parallel 4 saves 7.45 seconds (2.5%) over
  Clang alone. The gains overlap but the combination is the fastest valid
  full-target build so far.
- Clang `-j6` regresses by 10.22 seconds (3.6%) under the four-core quota.
  Retain `-j4`.
- Clang `-O0` reduces compilation by 79.28 seconds (27.7%) and all 114 unit
  tests pass, but VC3D exits during `viewer.set_overlay`. The smoke harness
  times out at the same call with 10-, 30-, and 120-second per-RPC deadlines;
  this is not acceptable as a merely slower test binary.
- Clang 21 maps `-g1` directly to `-debug-info-kind=line-tables-only`, so
  `-gline-tables-only` is an alias rather than another optimization. The
  configuration already uses PCH, Clang's integrated assembler, and `ld.lld`.
  No additional distinct low-effort Clang frontend/linker flag was identified
  that preserves line traces and program semantics.
- The valid combined build records approximately 1,024.5 CPU-seconds of C++
  compilation work. Every one of its 25 slowest Ninja edges is compilation or
  Qt AUTOGEN/PCH work; no executable/shared-library link appears there. Linker
  substitution is therefore not a useful next cold-build experiment.
- Added a `--rpc-timeout` smoke-test option with the existing 10-second default
  so deliberately slow configurations can be tested without weakening the
  Release/default guard. Default Clang `-O1` smoke passes in 1.37 seconds;
  non-positive values are rejected by argument validation.

## Clang O0 Smoke Failure Diagnosis

- Improved the offscreen smoke harness so an unexpected VC3D exit reports the
  child return code and the useful prefix of its crash report immediately,
  instead of continuing with secondary RPC failures against a dead socket.
- The first `viewer.set_overlay` request did not run slowly. VC3D aborted with
  `SIGABRT` in `CChunkedVolumeViewer::setOverlayWindow` at
  `CChunkedVolumeViewer.cpp:2995`.
- The request clamps threshold 300 to 255. The controller and `ViewerManager`
  correctly represent the upper-edge window as `(255, 255)`, but the viewer
  redundantly evaluated `std::clamp(high, low + 1, 255)`, giving invalid bounds
  `(256, 255)`. `std::clamp` requires its lower bound not to exceed its upper
  bound, so the old expression had undefined behavior.
- The CI image's libstdc++ 15 explains why the bug appeared only at O0:
  `bits/c++config.h` automatically enables `_GLIBCXX_ASSERTIONS` when
  `__OPTIMIZE__` is absent. O0 caught the violated precondition; O1 omitted the
  runtime assertion but did not make the expression valid.
- Fixed the viewer to clamp both values independently to `[0, 255]`, then use
  the same guarded `min(255, low + 1)` adjustment already used by the
  controller, `ViewerManager`, and base-volume window setter.
- Local `cmake --build volume-cartographer/build --target VC3D -j32` passed.
  The host smoke run was not usable because one sandboxed attempt could not
  listen on the Unix bridge socket and a subsequent host run hung for unrelated
  environment reasons; the exact CI container is the authoritative validation.
- Rebuilt the affected target in the exact CI image and original Clang O0
  build tree. The complete VC3D offscreen smoke suite now passes in 1.9 seconds
  with the original 10-second per-RPC timeout.
- The deterministic non-network CTest selection also passes 114/114 in 15.57
  seconds on the warm incremental tree.
- Revised decision: Clang O0 plus AUTOGEN 4 is now a valid fast-test candidate,
  with the previously measured 207.20-second cold full-target compile and
  42.49-second cold-build test runtime. The earlier `reject O0` entries record
  the pre-diagnosis benchmark result and are superseded by this section.

## Dependency And Shard Measurements

- The exact Linux CI image already supplies the normal Ubuntu dependencies.
  The remaining source-built dependencies are libbacktrace, header-only libigl,
  and PaStiX. The image has no standalone Ubuntu `libbacktrace` development
  package, and Flatboi documents its dependency on the project's exact PaStiX
  6 configuration. The highest-return image change is therefore to install the
  pinned libbacktrace and PaStiX builds into `vc3d-deps` and make CMake prefer
  those installed targets. Baking libigl mainly removes fetching because its
  headers are still compiled in each consumer.
- All shard measurements below are clean Clang 21 QuickBuild builds using
  `-O0 -g1`, PCH, no LTO, `CMAKE_AUTOGEN_PARALLEL=4`, Ninja `-j4`, the exact
  four-CPU CI container, and no reused compiled objects. Existing downloaded
  source trees were used offline so the measurements isolate compilation from
  unreliable network time.
- Full all-target reference: **207.20 seconds** compile.
- All applications and tests with `VC_BUILD_FLATBOI=OFF`: **192.36 seconds**.
  Moving Flatboi out of the monolithic lane therefore removes only 14.84
  seconds (7.2%) from its critical path.
- Isolated Flatboi/PaStiX target: **30.11 seconds** compile, plus 9.00 seconds
  configure in this fresh tree. It is a clean independent compile-coverage
  lane and remains shorter than every proposed complement.
- Tests-only closure with applications and Flatboi disabled: 2.74 seconds
  configure, **103.16 seconds** compile, and **20.66 seconds** for all 114
  deterministic non-network tests, or **126.56 seconds** total.
- VC3D-only closure with tests and Flatboi disabled: 1.99 seconds configure,
  **92.51 seconds** compile, and approximately **2.2 seconds** for the complete
  offscreen RPC smoke suite. The smoke suite passed.
- The tests-only graph reaches `vc_core` at 17.40 seconds and the reusable test
  PCH at 22.59 seconds. Their measured compilation work is approximately 54.4
  and 3.4 CPU-seconds respectively. Two independent test build trees therefore
  duplicate at least approximately 57.8 CPU-seconds of base/PCH work, with a
  roughly 22.6-second wall-time readiness floor per lane. Advanced Lasagna,
  Atlas, fiber-tracer, and tracer libraries add another approximately 54.1
  CPU-seconds in the specialized lane.
- The useful test split follows ownership/dependencies rather than dividing the
  CTest list numerically: a core/data lane for `vc_core` tests and a specialized
  lane for Lasagna, Atlas, native fiber tracer, neural tracer, and tests that
  compile VC3D sources directly. Realizing the split without compiling unrelated
  targets requires explicit CMake aggregate targets (and matching CTest labels
  or fixture lists); the existing global `all` target always builds every test
  and the advanced core libraries.
- The largest individual test targets are GUI/source-heavy rather than test-run
  heavy: `test_ink_detection_overlay` uses 24.26 CPU-seconds of compile work,
  `test_viewer_overlay_surface_primitives` 22.12, and
  `test_segmentation_lasagna_panel_ui` 21.20. They belong with the VC3D/specialized
  ownership lane, not the base core/data lane.
- Path-filtered triggering is viable once every source area maps conservatively
  to its build and test closures. Flatboi changes can trigger only its lane;
  VC3D sources trigger VC3D compile/smoke plus affected GUI tests; Lasagna,
  Atlas, and fiber-tracer sources trigger the specialized test lane. Shared
  `core` headers/sources, PCH, top-level CMake/toolchain files, and dependency
  image changes must fan out to all applicable lanes to avoid false negatives.

## GitHub Hosted-Runner Capacity

- GitHub standard hosted-runner concurrency is scoped to the repository
  owner's account or organization plan, not separately budgeted per repository:
  Free 20 concurrent jobs, Pro 40, Team 60, and Enterprise 500. The standard
  macOS subset is capped at 5 for Free/Pro/Team and 50 for Enterprise.
- Standard GitHub-hosted runner usage is free and unlimited for public
  repositories, but the concurrency ceiling still applies. A 3--6 job VC3D
  split is below the Free ceiling; jobs can still queue if other repositories
  owned by the same account/organization consume the shared capacity.
- The effective live allowance is visible under Settings -> Actions -> Runners
  -> GitHub-hosted runners -> All jobs usage. Sources:
  <https://docs.github.com/en/actions/reference/limits>,
  <https://docs.github.com/en/actions/reference/runners/github-hosted-runners>,
  and <https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/monitoring-your-current-jobs>.

## Implemented Linux CI Shards

- Added configure-time test registration in CMake. Every CTest from the core,
  VC3D unit-test, and agent-bridge test directories must be registered exactly
  once. CMake compares registered names with the actual directory inventories
  and fails configuration on an omission or stale registration.
- Added `vc_test_core` and `vc_test_specialized` aggregate build targets and
  matching `vc-core` / `vc-specialized` CTest labels. The specialized closure
  contains consumers of Lasagna, Atlas, native fiber tracing, and neural
  tracing. Qt/VC3D-source tests needing only `vc_core` remain in the base
  closure to balance clean compilation while preserving dependency separation.
- Added `vc_cli_all`, with configure-time validation against every `vc_*`
  executable created by `apps/CMakeLists.txt`. A new CLI cannot be silently
  omitted from compile coverage.
- Added `VC_QUICKBUILD_OPT_LEVEL`, defaulting to the existing `1`. Fast Linux
  CI explicitly selects the validated Clang `-O0 -g1` build; ordinary developer
  QuickBuild remains `-O1` unless explicitly overridden.
- Replaced the monolithic Linux GCC/Clang Release matrix with parallel Clang
  QuickBuild lanes for base tests, specialized tests, VC3D compile/smoke, all
  CLI tools, and Flatboi/PaStiX. The existing aggregate `CI` job remains the
  required result and accepts only success or legitimate path-filtered skips.
- Path filters skip unrelated lanes. Shared core, build-system, dependency, or
  workflow changes fan out conservatively; VC3D, CLI, Flatboi, and MCP-local
  changes can avoid unrelated compilation.
- The three live tests remain labeled `network` but are enabled by default in
  ordinary CTest, `check-core`, and GitHub CI. `check-core-offline` and local
  benchmark commands exclude them explicitly so local network latency cannot
  affect compiler/shard decisions.

### Clean Four-CPU Results

All measurements use the exact CI image, Clang 21, QuickBuild `-O0 -g1`, PCH,
no LTO, AUTOGEN parallel 4, Ninja `-j4`, and clean target closures. Downloaded
source trees were present, but compiled dependency and project outputs were
cleaned.

| Lane | Configure | Compile | Tests/smoke | Measured total |
|---|---:|---:|---:|---:|
| Base tests | 2.6 s shared configure measurement | 57.81 s | 7.54 s, 96 offline tests | about 67.9 s |
| Specialized tests | same configuration | 59.36 s | 14.85 s, 18 tests | about 76.8 s |
| VC3D | 1.99 s | 92.51 s | 2.2 s smoke | about 96.7 s |
| All CLI tools | incremental configure 0.5 s | 56.46 s | compile coverage only | about 57.0 s |
| Flatboi/PaStiX | 9.00 s | 30.11 s | compile coverage only | about 39.1 s |

- Base and specialized CTests pass for the measured offline selection. The
  three network tests are deliberately absent from the benchmark result but
  remain enabled in GitHub CI; their timing must be read from the first pushed
  workflow run.
- The complete VC3D offscreen RPC smoke suite passes after the split, and the
  CLI aggregate builds every registered command-line executable.
- Expected raw Linux critical path is now VC3D at approximately 96.7 seconds.
  GitHub checkout, container startup, sccache setup, dependency fetching, and
  live-test latency are outside these compile/test measurements.
- The workflow now uses Clang for the fast Linux PR gate. It no longer provides
  the previous independent GCC Release/LTO PR compilation; retaining a slower
  GCC gate should be decided separately rather than hidden in the fast-path
  timing claim. The existing macOS Homebrew-Clang compile gate is unchanged.

### Remaining Source-Built Dependencies

- Ubuntu/image packages provide almost all dependencies, but the build still
  creates libbacktrace and the pinned PaStiX configuration from source in every
  clean shard. Libigl is header-only but is also fetched as source.
- Install only pinned libbacktrace in `vc3d-deps` and teach CMake to prefer the
  installed target. PaStiX configuration/patching, the custom libigl solver
  integration, and Flatboi are Villa-owned development and remain in Villa.
- `vc3d-deps` currently contains only platform dependency manifests, install
  scripts, restore helpers, and publishing workflows. Its Linux workflow builds
  an Ubuntu image from `linux/Dockerfile`; it neither checks out Villa nor
  contains Villa/Flatboi source.
- Moving the Flatboi executable itself into that image would require pinning or
  checking out Villa in the dependency-image build and would stop ordinary CI
  from compiling the PR's Flatboi source. Since the new path filters already
  skip the Flatboi lane on unrelated changes, this saves little on ordinary
  PRs. Keep PaStiX, `flatboi.cpp`, and the custom libigl integration built from
  Villa when their lane is relevant; only libbacktrace remains a candidate for
  the separate dependency image.
- In the measured clean VC3D lane, source-built libbacktrace completed at about
  3.4 seconds while the complete compile took 92.51 seconds. Its work overlapped
  with utilities, PCH generation, and other compilation; it completed before
  the core compilation became the lane bottleneck. Preinstalling it therefore
  removes repeated work and a network fetch from every core-consuming shard,
  but is unlikely to reduce the roughly 96.7-second VC3D critical path by more
  than a small amount.

## Current-Main Transplant Validation

- Created `ci-sharding` from GitHub `origin/main` commit
  `99eb131500885eb78e638f77430ec61b440d904e` and transplanted the CI change
  without source conflicts.
- Reconfigured and rebuilt from clean lane-specific directories in the exact
  Linux dependency image with four CPUs, Clang 21, QuickBuild `-O0 -g1`, PCH,
  no LTO, AUTOGEN parallel 4, and Ninja `-j4`.
- Built both test target closures and passed all 114 locally selected
  non-network CTests: 96 base and 18 specialized. The three network tests remain
  enabled in the GitHub workflow.
- Built the VC3D target and passed the complete offscreen agent-bridge smoke
  suite.
- Built the `vc_cli_all` aggregate and the Flatboi/PaStiX target successfully.
- Workflow YAML parsing and `git diff --check` passed on the transplanted branch.

## Deviations And Limitations

- Independent plan review was not delegated because the active policy forbids
  subagents unless the user explicitly requests them; the plan was reviewed
  locally.
- The first baseline had no local sccache, so it measures a truly cold raw
  build rather than GitHub's possible cross-run cache reuse.
