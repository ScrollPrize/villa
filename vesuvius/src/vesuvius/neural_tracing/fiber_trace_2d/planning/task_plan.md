# Plan: promote the best 1024 winding tune

## Implementation

1. Change the shared winding class defaults to `0,0,0.5,4,1` and sign defaults
   to `1,0.5`; keep explicit CLI overrides unchanged.
2. Update CLI help, solver documentation, specification, and changelog.
3. Update the focused regression that locks the production defaults.
4. Commit only the files belonging to this promotion task.

## Validation

- Build `vc_fiber_trace_chunk` and `test_fiber_trace_winding_bp` in the existing
  optimized build.
- Run `test_fiber_trace_winding_bp`.
- Check CLI help displays the promoted defaults.
- Run `git diff --check`.

## Spec update

Replace the shared/CLI default tuples and record the 1024 selection result.

## Docs update

Update `volume-cartographer/docs/fiber_chunk_tracing.md` with the new defaults
and the benchmark that selected them.

## Changelog

Record the promoted defaults and 1024 result.
