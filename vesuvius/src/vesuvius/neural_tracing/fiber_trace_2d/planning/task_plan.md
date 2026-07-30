# Native Fiber Trace Meeting Search And Persisted Diagnostics Plan

## 1. Trace Termination Contract

- Convert the configured endpoint threshold from base voxels to trace voxels
  and pass it into both CP-pair one-way requests.
- Preserve all target-local planes and require the existing multi-plane
  crossing condition plus the selected in-plane threshold before early
  termination.
- Retain the full stepped path when a trace exhausts its budget. Do not snap a
  failed/out-of-threshold trace back to an earlier plane crossing, because its
  later samples are required by moving-plane fusion.
- Run forward and reverse independently even if one succeeds first. Preserve
  each direction's reason and crossings for final diagnostics.

## 2. Symmetric Moving-Plane Meeting Search

- Arc-length-resample both finite, deduplicated traces at a frequent,
  deterministic interval derived from the configured trace step.
- At each sample on trace A, fit its local tangent from adjacent resampled
  points and construct the plane through that sample with the tangent as its
  normal.
- Intersect every segment of trace B with that plane, including deterministic
  handling for endpoints and coplanar segments. Record interpolated positions
  and arc lengths on both traces.
- Repeat with A/B reversed so curved and asymmetric traces receive symmetric
  coverage.
- Also add endpoint-plane crossing candidates against the opposite CP when the
  crossing is within the configured endpoint threshold.
- Select by smallest finite 3D/in-plane gap. Break exact ties
  deterministically by balanced progress, then combined progress and stable
  indices.

## 3. Acceptance And Fusion

- Compute candidate traced length as forward arc length to the candidate plus
  reverse arc length to the candidate.
- Accept only a finite candidate with positive traced length and
  `meeting_error / traced_length <= 0.10` by default.
- Replace the obsolete gap-plus-arc selection factor with a persisted maximum
  meeting-error ratio setting.
- Port the Python fusion geometry into the shared C++ implementation: cut both
  partial traces at interpolated meeting positions, warp each partial toward
  their midpoint by cumulative arc-length fraction, reverse/concatenate, and
  arc-length-resample at the configured native trace step.
- Restore both original CPs exactly and expose raw trace/base-voxel error,
  normalized ratio, selected traced length, candidate source, and failure
  reason in `FiberTraceSegmentResult`.
- Do not require either one-way trace to reach its endpoint planes when an
  acceptable moving-plane meeting exists.
- Use stable combined-result codes with deterministic precedence:
  `invalid_trace_path`, `no_trace_plane_intersection`,
  `meeting_error_ratio`, `fusion_failed`, and `ok`. Caller-side exceptions use
  `trace_exception`. Keep forward/reverse one-way reasons and numeric context
  in optional detail; do not turn arbitrary exception text into the stable
  code.

## 4. CP-Owned Outcome Persistence

- Generalize `FiberTraceSegmentMetadata` into an explicit versioned outcome:
  accepted native geometry or Lasagna fallback after a native attempt.
- Store manifest provenance, trace scale, effective config, optional meeting
  error/ratio, and failure reason in the one existing `segment_to_next` object.
- Treat only the accepted-native outcome as protected geometry. Update every
  marker, protected-range, constraint-derivation, revert, retry, and
  invalidation path that currently treats object presence as native success.
- Persist a failed outcome from mixed-mode fallback and from the direct GUI
  segment action while leaving/reconstructing Lasagna geometry.
- Increment the metadata schema. Read the previous accepted-only schema as an
  accepted outcome, ignore its obsolete gap-selection factor, and map its
  endpoint error into the accepted diagnostic, while all writers emit the new
  explicit schema. The new schema persists and validates
  `meeting_accept_max_error_ratio` in `[0, 1]`; no conversion from
  `fusion_gap_factor` is implied.
- Update strict C++ core, Python, merge-script, and atlas fixture readers so
  saved fibers remain consumable throughout the repository.
- Define lifecycle transitions explicitly: adjacent CP move/insert/delete and
  adjacency-changing reorder clear either outcome; retry replaces fallback
  with its new accepted/fallback outcome atomically; ordinary Lasagna-mode
  rebuilding clears native-attempt outcomes; explicit native-to-Lasagna revert
  clears an accepted outcome only after the Lasagna result succeeds; scaling
  clears both outcomes; unrelated edits preserve both.

## 5. Generated-Strip Diagnostics

- Generalize the span-label data model so each span can display one of:
  Lasagna normal-alignment state, accepted native meeting error, or persisted
  native failure reason.
- For an accepted native span, display the meeting error in base `vx` and put
  the ratio/source in the tooltip.
- For a native fallback, display a compact failure reason below the strip and
  the complete persisted reason in the tooltip.
- Populate native diagnostics directly from session CP metadata, independent
  of asynchronous Lasagna metric availability, so reload reproduces them.
- Continue showing existing normal-alignment metrics for spans without a
  native attempt.

## 6. Tests

- Add shared C++ tests for threshold-aware continued tracing, symmetric moving
  planes, interpolated crossings, curved/asymmetric meetings, endpoint-plane
  candidates, no-intersection failure, 10% acceptance/rejection, deterministic
  selection, exact endpoints, and Python-style arc-length warp behavior.
- Add VC3D tests for accepted/fallback outcome serialization, previous-schema
  reading, protection based on outcome, retry/invalidation behavior, and
  generated-strip diagnostics after a simulated reload.
- Add a deterministic synthetic CP-pair corpus containing endpoint-plane
  misses that have valid trace-to-trace meetings. Record the old endpoint-only
  accepted/fallback count in the fixture and require the new search to reduce
  fallbacks while preserving exact CP endpoints.
- Update strict Python metadata-reader and merge-script tests for both outcome
  variants and malformed combinations.
- Add a non-unit trace-scale test proving raw trace error converts exactly to
  base voxels, the ratio is scale-independent, and both values survive JSON
  save/reload.
- Build with `-j32` and run `test_fiber_trace3d`,
  `test_line_annotation_generated_views`, `test_lasagna_line_optimizer`, the
  focused Python reader tests, and merge-script tests.
- Build `VC3D` and `vc_fiber_trace_metric` from the main build tree with
  `-j32`.

## 7. Spec Update

- Replace gap-plus-arc CP-pair fusion selection with symmetric moving-plane
  intersection selection, 10% traced-length acceptance, and arc-length-warped
  fusion.
- Specify threshold-aware one-way continuation, full-path retention, failure
  reasons, base-voxel diagnostics, and the fact that moving-plane success can
  accept a pair without endpoint-plane success.
- Generalize CP-owned metadata from accepted-only protection to explicit
  accepted/fallback outcomes and define protection semantics.
- Define generated-strip display precedence for native outcomes versus
  Lasagna alignment.

## 8. Docs Update

- Update `docs/code_structure.md` for target-plane termination, moving-plane
  search, fusion geometry, acceptance, result fields, and GUI persistence.
- Update `volume-cartographer/docs/line_annotation_fibers.md` for explicit
  segment outcomes, protection checks, fallback retries, and strip labels.

## 9. Changelog

- Add a 2026-07-30 entry for symmetric native trace meeting search,
  arc-length-warped fusion, reduced Lasagna fallback, and persisted VC3D
  diagnostics.

## 10. Review Risks

- Verify that changing optional-object presence to explicit outcome does not
  accidentally protect failed Lasagna spans or remove protection from older
  accepted records.
- Verify that an exhausted one-way trace keeps samples after an earlier bad
  target-plane crossing.
- Verify moving-plane intersection handles trace endpoints, coplanar segments,
  duplicate samples, and interpolated cut locations without zero-length joins.
- Verify error units are trace voxels in the core, base voxels in persistence
  and GUI, and the normalized ratio is scale-independent.
- Verify direct GUI rejection persists diagnostics without replacing the line
  with partial native geometry.
- Verify stable failure-code precedence and ensure verbose one-way/exception
  details never become the GUI's persisted machine-readable discriminator.

## 11. Workflow Records

- Update `planning/status.md` incrementally.
- Replace and maintain `planning/task_log.md` with review findings,
  implementation discoveries, deviations, commands, and validation results.
