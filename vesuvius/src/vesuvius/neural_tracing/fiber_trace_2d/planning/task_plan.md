# Plan: Opt-In Fiber 3D Hang Diagnostics

## Implementation

1. Add `training.test_hang_diagnostics_enabled`, defaulting to `false`, and a
   `--test-hang-diagnostics` normal-training CLI flag. The effective setting is
   config OR CLI so either explicit request enables diagnostics.
2. Give `_TrainingHangDiagnostics` an enabled/no-op mode. When disabled it must
   not open `hang_diagnostics_rank_*.log`, register `SIGUSR2`, arm a
   `faulthandler` timer, collect resource fields, or perform diagnostic CUDA
   synchronizations.
3. Validate `training.test_watchdog_seconds` only when diagnostics are enabled;
   preserve the existing 480-second default and `<600` constraint when active.
4. Print the effective enabled/disabled state at startup and record CLI/effective
   state in TensorBoard's effective config. Preserve all existing diagnostic
   markers and cleanup when enabled.
5. Strictly require the config setting to be a JSON boolean and reject the
   training-only CLI flag when combined with prefetch, benchmark, or Trace2CP
   visualization modes.

## Testing

- Add a regression proving disabled diagnostics create no file and do not arm
  watchdog/signal state, query resources, or synchronize CUDA.
- Retain and run enabled automatic-watchdog, cancellation, re-arm, and manual
  dump tests.
- Test config default/config opt-in/CLI opt-in resolution and CLI forwarding.
- Test invalid config types and enabled-only watchdog timeout validation.
- Run focused tests where available, Python compilation, and diff checks.

## Spec update

Change diagnostics from unconditional to explicit opt-in and define both
activation mechanisms and disabled side-effect guarantees.

## Docs update

Document the config key, CLI flag, default disabled state, output files, and
timeout behavior.

## Changelog

Record that Fiber 3D hang diagnostics are now opt-in.
