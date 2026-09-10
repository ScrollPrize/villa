# Plan

Default run_training to cooperative mode; add explicit --unmanaged testing/debug opt-out and reject contradictory socket/auxiliary flags. Use one shared default socket helper in gpu_manager CLI and training settings; explicit CLI socket overrides config socket, which overrides the default. Keep errors fatal (no unmanaged fallback). Make the named profiling harness explicitly opt out. Update production docs to omit socket flags except for overrides. Include gpu_manager source in PYTHONPATH; report the current embedding config's missing request/calibration. No GPU runs.

## Spec update
Managed-by-default production entry point, explicit testing opt-out, shared default socket. No protocol or numerical changes.

## Docs updates
Update local_fiber_embeddings, training_s1a_8gpu_retrain, cooperative_training, code_structure and historical profiling notice. Keep task log/status accurate.

## Testing

Global-path refinement: prefer /run/villa-gpu-manager/manager.sock if a bounded
connect-only probe succeeds or the nearest existing parent is writable/searchable;
otherwise choose the home path. Do not create directories during selection. Keep
explicit socket overrides and never retry another manager after authorization or
protocol failure. Retain server ownership/group protections. Test live sockets,
stale/non-socket entries, writable global creation location and private fallback.
CPU tests for default dispatch, explicit opt-out, invalid combinations, error propagation and socket precedence; existing cooperative and embedding tests. Check CLI help, snippets and diff whitespace.

## Changelog
Document corrected manager-only production launch and outstanding embedding calibration.
