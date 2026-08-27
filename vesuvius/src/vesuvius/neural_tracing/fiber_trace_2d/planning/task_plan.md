# Plan: Napari winding-fiber viewer

## Scope and behavior

1. Add a standalone Python module under `vesuvius.scripts`; do not change the
   C++ winding artifact format or the existing presence/replay viewer.
2. Accept an output base such as `/path/to/fibers` or `/path/to/fibers.obj`.
   Discover only exact nonnegative integer state suffixes and ignore valid
   aggregate `<base>_w_N.obj` and winding CSV siblings. Require at least one
   winding and a complete `{h,v,err,tie}` quartet for every discovered winding;
   empty state files are valid and winding labels need not be contiguous.
3. Extract the ordered-polyline OBJ parsing currently embedded in
   `view_fiber_presence.py` into a shared helper and keep that caller's custom
   header, quality-metadata, and crop validation in place. The shared parser
   must handle configurable `g`/`o` containers, global one-based indices,
   finite XYZ vertices, adjacent `l` chains, and singleton `p` objects. Reject
   duplicate names, out-of-range or cross-object indices, unused vertices,
   mixed point/line records, branching/cycles, disconnected chains, and
   unsupported records. Convert winding XYZ geometry to Napari ZYX and accept
   valid empty state partitions without creating useless layers. Winding
   fibers themselves must remain polylines; a singleton in a state artifact is
   malformed viewer input.
4. Create one Napari Shapes path layer per nonempty winding/state artifact in
   an explicit 3D viewer. Assign deterministic, bright, distinct RGBA colors
   as a pure mapping from `(winding,state)`, so adding another artifact cannot
   recolor existing layers. Keep Napari and Qt imports inside GUI construction
   so discovery and parser tests need no GUI extra.
5. Add a dock widget with category presets for H, V, Broken, All, and None.
   Broken controls both `err` and `tie`. Add previous/next winding controls;
   each navigation action selects one winding and shows its H and V layers,
   hiding all other winding/state layers. Navigate only windings with nonempty
   H or V geometry, show the selected winding in a compact label, and wrap at
   the endpoints.
6. Keep the visibility calculation as a pure helper so controls never depend
   on Napari internals beyond setting each layer's `visible` attribute.

## Spec update

Document the winding visualization artifact contract, Broken grouping, and
category/winding visibility semantics in `planning/specs.md`. No inference or
serialization contract changes are required.

## Docs updates

Add a usage section to `volume-cartographer/docs/fiber_chunk_tracing.md` with
the exact `python -m` command, accepted base-path forms, layer layout, and
control behavior.

## Testing

1. Add focused Python tests for OBJ parsing, empty partitions, singleton point
   handling, malformed/out-of-object indices, orphan vertices, duplicate
   names, unsupported records, branching/disconnected chains, artifact quartet
   completeness, aggregate-file exclusion, and numeric winding order.
2. Test stable deterministic distinct colors with bright opaque bounds and
   pure category/single-winding visibility selection, including `err`/`tie`
   grouping, nonempty navigation eligibility, and wraparound.
3. Run the focused pytest module, compile the Python modules, run a headless
   discovery/load smoke test against the existing 1024 crop artifacts, and run
   `git diff --check`. Verify module import without importing Napari/Qt.

## Changelog

Add a dated entry for the winding-layer Napari viewer and its grouped controls.
