# Trace2CP Z-Search Layer TIFF Export

Add an opt-in Trace2CP z-search debug export that writes all z layers inferred
and used by the z-search cache as a multilayer TIFF.

- The flag should only be valid with `--trace2cp-z-search`.
- The TIFF layer order must be non-interleaved: first all sampled slice images
  in sorted z-layer order, then all corresponding presence maps in the same
  sorted z-layer order.
- The export must use the already inferred z-search layers and must not
  re-sample the volume.
- Single-pair Trace2CP should write one multilayer TIFF. Whole-fiber Trace2CP
  should write one multilayer TIFF per valid pair because pair-local segment
  strips can have different shapes.
