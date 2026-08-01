# VC3D GitHub CI Acceleration Status

- [x] Define the interactive CI-acceleration task.
- [x] Write and locally review the measurement/workflow plan.
- [ ] Independent subagent review (unavailable under the active no-delegation
      policy unless the user explicitly requests subagents).
- [x] Measure the current cold four-core Release/LTO build.
- [x] Run baseline CTest and VC3D offscreen smoke validation.
- [x] Measure a clean four-core QuickBuild without LTO.
- [x] Validate CTest and VC3D smoke against the QuickBuild candidate.
- [x] Measure a clean QuickBuild with reduced `-g1` debug information.
- [x] Benchmark reduced-debug QuickBuild with `-O0`.
- [x] Benchmark reduced-debug QuickBuild without function/data sections.
- [x] Benchmark reduced-debug QuickBuild with Ninja `-j6` on four CPUs.
- [x] Benchmark reduced-debug Clang QuickBuild.
- [x] Benchmark reduced-debug QuickBuild with Qt AUTOGEN parallelism set to 4.
- [x] Benchmark Clang combined with AUTOGEN parallel 4.
- [x] Benchmark combined Clang/AUTOGEN with Ninja `-j6` on four CPUs.
- [x] Inspect and benchmark remaining distinct Clang fast-build flags.
- [x] Diagnose and fix the Clang O0 VC3D smoke failure.
- [x] Profile the remaining build critical path.
- [x] Benchmark test-dependency-only target closures.
- [ ] Benchmark warm GitHub Actions sccache behavior after pushing the workflow.
- [x] Implement the selected build-system and workflow changes.
- [x] Verify every Clang CI shard locally in the exact dependency image.
- [x] Record the deliberate removal of the slower GCC Release/LTO pull-request
      gate as a workflow tradeoff.
- [ ] Update developer documentation and changelog.
- [ ] Validate network-test timing and all aggregate results on GitHub Actions.
