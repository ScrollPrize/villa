# Plan: Preserve Version-3 Fiber Spans During Sync

## Merge Model

1. Keep the existing file-level S3 conflict detection, three-way base lookup,
   conflict stashing, confirmation, and manual local/remote/skip workflow.
2. Apply new semantics only when base, local, and remote are version-3 fibers.
   Leave version-1/version-2 geometry behavior unchanged.
3. Split each valid fiber line at its ordered control points. A span atom is
   its start/end CP positions, dense line slice, and complete
   `segment_to_next` descriptor.
4. Reject automatic v3 merging when CPs cannot be located in order on the
   dense line with VC3D's loader-equivalent lookup.

## Three-Way Resolution

1. Base-align CP topology by position and derive changed runs bounded by
   unchanged base CP anchors.
2. Treat a run as unchanged only when its CP objects, dense line slice, and
   segment descriptors equal the base result.
3. Take a run changed on only one side as one atomic result. If both sides
   changed the same run identically, take either copy.
4. Reject different two-sided changes to the same run.
5. Permit changes from both sides only when at least one complete unchanged
   base span separates their affected runs. Reject adjacent changed runs even
   when they happen to meet at an unchanged CP.
6. Concatenate selected dense slices at exact shared CP endpoints. Preserve
   each selected side's CP objects and descriptors without field-level mixing.
7. Do not synthesize a control-point-only line or discard v3 descriptors.
8. Merge `optimization_mode` with ordinary three-way scalar rules; different
   changes on both sides are a manual conflict.

## Existing Metadata

1. Retain existing base-aware tag, branch, and manual-HV-tag behavior.
2. Re-anchor branch links only after a clean geometry merge and retain the
   existing reciprocal-peer consistency pass.
3. Continue taking non-policy opaque/scalar fields from the newer generation
   after the explicit v3 geometry and `optimization_mode` merge.

## Tests

1. Add v3 fixtures with strict segment descriptors and independently shaped
   dense spans.
2. Cover one-sided span changes, identical two-sided changes, separated local
   and remote changes, adjacent changes, same-span conflicts, topology edits,
   line-slicing failures, and base-aware `optimization_mode` changes.
3. Assert clean outputs remain loadable and retain complete descriptors and
   exact selected dense slices.
4. Run the fiber merge and vc_sync helper suites with third-party pytest plugin
   autoload disabled if the local environment's optional zarr plugin is absent.

## Spec Update

- Specify atomic v3 span merge semantics, the unchanged-span separation rule,
  base-aware global mode handling, and mandatory manual conflicts for
  ambiguous cases.

## Docs Updates

- Document how sync detects file conflicts and how v3 span conflicts are
  resolved without generation-based overwrites.

## Changelog

- Record lossless conservative v3 sync merging.
