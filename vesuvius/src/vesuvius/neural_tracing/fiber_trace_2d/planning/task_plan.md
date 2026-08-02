# Plan: VC3D GitHub CI Acceleration

## Measurement

1. Establish the current four-core cold-build, test, and VC3D smoke baseline
   in the exact GitHub CI dependency image.
2. Benchmark a clean `QuickBuild` (`-O1`, PCH, no LTO) `ninja all -j4` build
   with the same image and source tree.
3. Run all registered tests and the VC3D offscreen smoke test on the candidate
   build before considering it usable for test execution.
4. Use build logs and Ninja timing data to identify the remaining critical
   path, then test one isolated change at a time.

## Workflow Design

1. Separate the requirement to compile every configured target from the
   requirement to execute tests.
2. Split Clang QuickBuild coverage into independently scheduled base-test,
   specialized-test, VC3D, CLI, and Flatboi target closures.
3. Use configure-time registration checks so new tests and CLI targets cannot
   silently escape all shards.
4. Retain network tests in GitHub CI while excluding them only from local
   performance comparisons.
5. Preserve sccache across workflow runs and measure warm behavior from the
   first pushed workflow runs.
6. Record that the fast pull-request workflow replaces the previous GCC and
   Clang Release/LTO matrix with Clang-only QuickBuild coverage. A production
   GCC/Release gate, if required, belongs outside the sub-minute feedback path.

## Tests And Validation

1. Compare clean builds with identical four-CPU limits and `-j4` Ninja
   parallelism.
2. Record configure time, build time, test time, smoke-test time, target/build
   edge count, and cache state.
3. Require all registered CTest tests and the VC3D offscreen smoke test to pass
   for each candidate test configuration.
4. Confirm that the combined Clang shards build every configured tool, test,
   benchmark, and application target represented by the selected options.
5. Validate the workflow YAML and run `git diff --check` before publishing.

## Spec Update

- No product behavior specification change is currently expected. If CI
  coverage contracts become durable project requirements, document them in
  the developer/CI documentation rather than the fiber algorithm spec.

## Docs Updates

- Document the final local reproduction command, CI job responsibilities, and
  cache behavior in the relevant Volume Cartographer developer documentation.

## Changelog

- Add a changelog entry only when workflow or build-system changes are adopted;
  measurement-only iterations remain in `planning/task_log.md`.
