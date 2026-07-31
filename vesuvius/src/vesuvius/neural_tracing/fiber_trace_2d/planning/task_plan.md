# Plan: VC3D Native Tracing For New Fibers

## Dispatch And State

1. Detect whether the active project has configured fiber-inference data
   without opening a file picker. A selected dataset or attached fiber dataset
   counts as configured.
2. During native-mode seed creation, open the configured inference dataset
   before starting the seed solve. If none is configured, use the current
   Lasagna seed path and log the explicit fallback reason.
3. Run the existing Lasagna seed solve as the reference-line/tangent baseline.
   Mark the session so completion chains into the existing single-control-point
   `optimizeFiberWithNativeFallback` path.
4. Apply the baseline to session state without materializing generated views,
   invoking success callbacks, or saving the fiber. Immediately start native
   fiber-mode optimization with full retracing.
5. Only the native result (including its established invalid-data termination
   semantics) becomes the visible and persisted new-fiber geometry. If the
   configured inference dataset cannot be opened, report the existing error and
   leave the seed unstarted rather than silently claiming native tracing.

## Tests And Validation

1. Add focused tests for the configured-data/native-mode decision and for the
   default native mode.
2. Build `VC3D` and the focused line-annotation test with 32 threads.
3. Run the focused test binary and `git diff --check`.

## Spec Update

- State that new-fiber native mode controls initial geometry, not only the GUI
  and saved mode, when fiber inference is configured.
- State that the seed Lasagna solve is an internal reference baseline and is
  not an accepted/persisted interpolation result.

## Docs Updates

- Update the VC3D fiber annotation documentation with initial seed dispatch and
  the no-config Lasagna behavior.

## Changelog

- Record that new native-mode fibers now run native extrapolation immediately
  after seed placement when inference data is configured.
