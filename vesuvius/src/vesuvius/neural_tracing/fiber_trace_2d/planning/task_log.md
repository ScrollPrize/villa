# Task Log: Opt-In Fiber 3D Hang Diagnostics

## Findings

- The existing diagnostics object was created on every rank and immediately
  opened a per-rank file and registered `SIGUSR2`.
- Diagnostic CUDA synchronization was unconditional for CUDA tests even if no
  file was available, so merely suppressing file creation would not have made
  disabled mode side-effect-free.
- Independent review required strict JSON-boolean validation, enabled-only
  timeout validation, explicit effective-config fields, and rejection of the
  normal-training flag in auxiliary modes.

## Implementation

- `training.test_hang_diagnostics_enabled` now defaults to `false`; normal
  training can also enable diagnostics with `--test-hang-diagnostics`.
- Disabled `_TrainingHangDiagnostics` is a no-op object: it opens no file,
  registers/cancels no signal or timer, polls no resources, and causes
  `_diagnostic_cuda_synchronize` to return before a GPU barrier.
- The config key must be a JSON boolean. Config and CLI activation are ORed and
  their config/CLI/effective values are recorded in TensorBoard's effective
  config. Rank 0 prints the effective state at startup.
- `training.test_watchdog_seconds` retains its 480-second default and validation
  only when diagnostics are active. Invalid/irrelevant values are ignored while
  disabled.
- Prefetch, benchmark, and Trace2CP visualization reject the diagnostics flag.

## Validation

- A direct no-op lifecycle smoke test mocked file-global faulthandler calls and
  CUDA synchronization; disabled mode invoked none and created no log.
- Resolver smoke coverage passed for default, config, CLI, invalid type, and
  disabled invalid-timeout behavior.
- The auxiliary-mode CLI rejection path passed.
- Added pytest regressions for the disabled lifecycle and resolution rules;
  existing subprocess enabled-watchdog/manual-dump tests remain intact.
- Python compilation and `git diff --check` passed. The environment still lacks
  pytest, so the pytest suite was not executed and no dependency was installed.
