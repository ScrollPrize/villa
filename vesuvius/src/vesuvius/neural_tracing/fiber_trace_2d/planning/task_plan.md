# Plan: regular-tracer fiberlet loss and napari colormap selector

## Shared local alignment loss

1. Extract the regular tracer's ordered float calculation of presence times its
   six positive-clamped alignment dots into `FiberLocalScoring`. Keep the exact
   multiplication order and make both native tracer scoring paths call it so
   the regular tracer's numerics and behavior remain unchanged.
2. Replace fiberlet `presence + direction` cost components with one `alignment`
   component. Remove the unshipped presence/direction weights and CLI/JSON
   fields rather than retaining compatibility aliases.
3. For each valid integer-lattice transition, sign-align the current prediction
   to the incoming step and the next prediction to the outgoing step, evaluate
   the shared multiplicative loss, and multiply it by outgoing edge length.
   This preserves comparable integration for axial and diagonal lattice edges.
   This explicitly removes the old per-voxel alignment quantization floor; the
   45-degree free angle remains only in direct smoothness. A nonzero source
   attachment uses the fitted source axis for both previous step and current
   prediction, the attachment as outgoing step, and the dense destination
   prediction. A nonzero sink attachment uses the incoming step and current
   dense prediction, the attachment as outgoing step, and the fitted target
   axis at presence one. Both are followed by direct smoothness. Zero-length
   attachments add no length-weighted alignment but consistently establish or
   constrain the corresponding fitted endpoint axis.
4. Preserve the finite invalid-prediction bridge: a transition whose destination
   prediction is invalid pays only the configured invalid cost per edge length
   plus independently valid smoothness, with no alignment cost on that arrival.
   When leaving an invalid current voxel for valid data, use the incoming step
   as the unavailable current prediction; the first valid destination then pays
   ordinary multiplicative alignment.
5. Preserve shared direct normal/tangent smoothness and its integer-lattice free
   angle. Do not add cumulative-history smoothness because that was explicitly
   excluded and would require a materially larger DP state.

## Napari colormap selection

6. Remove the loss-density endpoint fields, mapping helper, reconciliation
   helper, spin boxes, and associated tests/docs. Remove C++ MTL generation,
   material names/RGB serialization, OBJ `mtllib`/`usemtl` records, and the
   corresponding Python MTL parser. Actively delete stale `fiberlets.mtl` during
   publication. Keep the independent textured central-slice MTL bundles. There
   is no compatibility path because the artifacts are experimental and
   unshipped.
7. Add one `QComboBox` for path quality color. Its first option uses a napari
   custom red-yellow-green colormap. Other options come from napari's public
   available-colormap registry in deterministic sorted order. All options map
   the existing continuous `relative_quality` Shapes feature with fixed
   contrast limits `[0,1]`. Changing the selector sets only `edge_colormap`;
   napari owns the recolor and the stored feature remains unchanged.

## Tests and validation

8. Add focused shared-scoring tests covering perfect alignment, presence,
   multiplicative collapse, axis sign handling at the caller, and invalid
   inputs where applicable. Add exact native sample-path and corner-path
   regression coverage so helper extraction cannot change their losses. Update
   fiberlet tests to verify multiplicative alignment affects route choice, a
   valid-invalid-valid bridge has exact components, and diagnostic/artifact
   schemas contain `alignment` rather than removed additive components or
   materials.
9. Retain strict metric-aware OBJ parsing without MTL, explicitly reject old
   material records, and add small pure tests for colormap-option ordering and
   custom-map definition without requiring a GUI session. Run focused C++
   suites with 32 build jobs, focused Python tests, Ruff, compile checks, and
   diff hygiene.

## Spec update

- Replace the additive fiberlet presence/direction equations with the shared
  regular-tracer multiplicative alignment equation, while documenting the
  integer-edge and invalid-gap adaptations.
- Replace selectable density endpoints with a napari colormap selector over
  fixed stored relative quality and remove MTL/material output.

## Docs updates

- Update the fiberlet objective, component schema, CLI options, metric caveats,
  and napari viewer controls.

## Changelog update

- Record the corrected shared multiplicative DP loss and napari colormap
  selector under the current fiberlet work.
