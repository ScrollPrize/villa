# Task Log

## Findings

- GitHub Actions reached the matrix after the Valgrind-version fix, then every
  collector failed while importing `passive_event_model.py` because the pinned
  dependency image has no NumPy.
- Local validation used the host Python environment, where NumPy was installed,
  and therefore did not cover the actual container dependency surface.
- Dependency replay is native C++, but per-thread event feature extraction and
  cost calculation were still performed by the NumPy calibration helper before
  costs were sent to the native engine.
- `run_render_valgrind_ci.py` also imports DRD parsing from
  `run_thread_sync_replay.py`, whose top-level calibration imports pull in
  NumPy even before a command runs.
- The local Docker socket is not accessible to this user, so exact pinned-image
  execution requires either another rootless runtime or a GitHub Actions run.

## Decisions

- Do not install NumPy in the CI workflow.
- Move runtime event-cost calculation into the existing shared C++ replay
  implementation and expose it through the persistent protocol.
- Lazy-load calibration-only Python dependencies.
- Add a `python3 -S` smoke check to prevent undeclared site-package imports.

## Plan Review

- The resulting gate has native model computation and replay, not literally
  zero Python; Python remains a standard-library coordinator.
- Extract the DRD parser structurally instead of relying only on lazy imports.
- Bump the native protocol for the scoring command.
- Define and test overflow safety, deterministic accumulation, finite input,
  malformed profile/model rejection, and tight parity with the prior model.
- Exercise parser and real native scoring under `python3 -S`; the full workflow
  matrix remains the end-to-end test of all three commands.

## Validation

- Configured a clean Release benchmark build with the workflow's GCC 15
  toolchain and `VC_MARCH_NATIVE=OFF`, then built the benchmark, replay engine,
  and native replay tests with 32 jobs.
- Five focused CTests passed: fixture, native replay, CI driver, no-site import
  and scoring, and replay-model compatibility.
- The no-site test runs with `python3 -S`, imports the complete CI driver and
  dependency-free DRD parser, and performs a real event-cost request against
  the C++ replay process.
- The complete fresh eight-case Callgrind/DRD gate passed. Serial ratios were
  exactly 1.000; parallel ratios were 1.000, 1.006, 0.995, and 1.033.
- All 102 benchmark Python unit tests passed with one expected skip.
- Ruff formatting and lint, Clang formatting, workflow YAML parsing, and
  `git diff --check` passed.
- A direct pinned-container attempt could not access this session's Docker
  daemon socket. The repository was therefore not claimed as container-tested;
  the next GitHub Actions execution remains the exact-image confirmation.
