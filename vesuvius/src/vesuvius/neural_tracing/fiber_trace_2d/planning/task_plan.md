# Plan: Require Lasagna Normals For Native Trace2CP

## Scope

- Enforce explicit Lasagna normal data for normal-aware native 3D Trace2CP.
- Keep existing isotropic/no-normal helpers available only for explicit
  lower-level configurations that disable normal-aware weights.
- Do not change trace scale handling or the existing trace-control defaults.

## Implementation

1. Python native Trace2CP
   - Add a small helper that determines whether the native config has active
     normal-aware smoothing terms.
   - Change `_native_trace_cfg_with_effective_smoothness()` to raise when those
     terms are active and no normal sampler is provided.
   - Change lower-level tensor smoothness helpers to raise on missing candidate
     normals only when the requested term actually needs them.
   - Update focused tests so normal-aware trace calls pass a normal sampler, and
     add a regression test for the hard failure.

2. C++ metric CLI
   - Make `--normal-manifest` required in `vc_fiber_trace_metric` usage and
     validation.
   - Remove the CLI fallback that attempts to create a normal sampler from the
     fiber prediction manifest.
   - Keep remote-cache validation covering both fiber and normal manifests.

3. C++ tracer core
   - Add a guard for normal-aware smoothing requests with a null normal sampler.
   - Update synthetic tests to pass a constant normal sampler when using the
     default normal-aware config.
   - Add a regression test for the null-normal-sampler failure and for an
     explicitly non-normal-aware no-sampler path.

## Spec Update

- Update the native 3D Trace2CP spec to state that Lasagna normals are required
  for normal-aware smoothing and missing samplers are hard errors.
- Update the C++ metric spec to require explicit `--normal-manifest` and state
  that prediction-manifest normals are not used.

## Docs Update

- Update `docs/code_structure.md` where native Trace2CP smoothness and
  `vc_fiber_trace_metric` invocation are described.

## Changelog

- Add a 2026-07-29 entry for the normal-sampler requirement.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. python -m pytest vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d_trace2cp"`
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- `cmake --build volume-cartographer/build --target test_fiber_trace3d`
- `cmake --build volume-cartographer/build --target vc_fiber_trace_metric`
- `volume-cartographer/build/bin/test_fiber_trace3d`
- `volume-cartographer/build/bin/vc_fiber_trace_metric --help`
- `git diff --check`

## Deferred Explicitly

- Real Lasagna normal manifest I/O is not added to C++ unit tests; those use a
  synthetic normal sampler and the CLI/build smoke tests cover argument/help
  behavior.
