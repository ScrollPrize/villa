# Plan: classify crop traces by principal fiber direction

## Contract

- Classification is a post-trace visualization operation. It must not alter
  seed ordering, tracing, coverage suppression, accepted geometry, or the
  existing complete OBJ.
- Treat local fiber directions as unoriented axes: reversing a trace must not
  change its classification.
- Estimate global direction 1 and direction 2 from consecutive local trace
  steps, not from each fiber's endpoint chord.
- Classify a fiber from the arc-length support assigned to each direction so
  irregular point sampling does not bias the result.
- Preserve deterministic output and emit valid empty group OBJ files when a
  crop has no members of a group.

## Implementation

1. Extract the existing axial two-line refinement kernel used by anchor fitting
   into a reusable helper, without changing the anchor caller's arithmetic or
   iteration accounting.
2. Add a reusable Fiberlet crop-direction classifier to `vc_fiber_tracer`.
   Accumulate the length-weighted unoriented second moment `sum(length*u*u^T)`
   of normalized consecutive steps for deterministic seed generation, then
   fit two independent (not forcibly orthogonal) axial directions by maximizing
   `sum(length*max((u dot d1)^2,(u dot d2)^2))`. Try deterministic farthest-axis
   seed pairs and retain the first maximum-objective fit.
3. Assign every nonzero step to the axis with the larger absolute dot product.
   Accumulate its length for that axis. A fiber is direction-1 or
   direction-2-dominant when that direction owns at least 75% of its valid
   local arc length; otherwise it is mixed. A trace with no valid step is
   mixed. Label the two fitted directions by descending assigned length, then
   canonical axis order on ties; equal per-step alignment goes to direction 1.
   Zero-support input uses canonical X/Y axes and classifies every line mixed.
4. Keep writing the requested complete OBJ. Partition the same named
   polylines, without modifying their points or order, into sibling
   `<stem>_dir1.obj`, `<stem>_dir2.obj`, and `<stem>_mixed.obj` outputs.
   Capture each trace's actual seed-anchor position and write matching
   `<stem>_anchors.obj`, `<stem>_dir1_anchors.obj`,
   `<stem>_dir2_anchors.obj`, and `<stem>_mixed_anchors.obj` point artifacts.
5. Report the two axes, analyzed step count/length, and group fiber counts in
   the final command output.

## Tests

- Add synthetic non-orthogonal, axis-aligned, reversed, unevenly sampled, and
  mixed polylines proving fitted modes, deterministic labels, sign invariance,
  length-based classification, 75% boundary behavior, and handling of a line
  with no valid step.
- Add an artifact partition test proving the complete and three sibling OBJ
  files are independently valid, preserve names/point order, partition every
  accepted fiber exactly once, preserve the actual seed position in matching
  point artifacts, and permit empty groups.
- Build `vc_fiber_trace_chunk` and `test_fiberlet_crop_trace` with the existing
  optimized build, run the focused test binary, and run `git diff --check`.

## Spec update

Extend the anchor-seeded crop-tracing specification with the non-orthogonal
axial local-step two-line fit, arc-length classification rule, fixed dominance
threshold, deterministic degeneracy rules, and the three additional
visualization artifacts.

## Documentation updates

Document the output filenames, classification math, threshold, console
summary, and that the operation cannot affect tracing.

## Changelog

Add one crop-tracing entry describing principal-direction classification and
the independently displayable OBJ groups.

## Follow-up: irrelevant partial halo tuples

- Filter replay-geometry halo prefixes by exact in-crop endpoint identity
  before collecting endpoint anchor owners.
- Keep strict partial-tuple rejection for every retained fiberlet dependency.
- Add a stored combined-dataset regression proving an irrelevant partial halo
  tuple is ignored while directly requiring that tuple still fails.
- Rebuild with GCC and Clang, run the storage and crop tests, and reproduce the
  reported 1024-base-voxel crop through 500 attempts.

### Spec update

State the endpoint filtering order and distinguish irrelevant halo tuples from
required partial tuples.

### Documentation updates

Document the same crop materialization rule in the Fiberlet crop-tracing guide.

### Changelog

No durable changelog entry is needed for this narrow correction to already
specified sparse-crop behavior.
