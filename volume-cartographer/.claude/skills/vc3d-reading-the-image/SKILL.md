---
name: vc3d-reading-the-image
description: Interpret VC3D CT slices and prediction overlays without confusing fibers, sheets, inclusions, empty data, or coordinate levels. Load with vc3d-visual-evidence when choosing a point or making a visual claim.
---

# Read VC3D imagery

Use `vc3d-visual-evidence` to create a trustworthy capture. This skill governs
what can be inferred from it.

## Interpretation loop

1. Record the selected volume, pyramid level, plane/surface, display window,
   overlay source, colormap, opacity, threshold, and resolution cap.
2. Inspect the base CT without an overlay before judging a prediction.
3. Toggle the intended overlay on, capture after render quiescence, then toggle
   it off and compare. A checksum or pixel change proves the overlay rendered;
   it does not prove semantic correctness.
4. Check spatial alignment at multiple landmarks and more than one nearby
   slice. Misregistration can resemble a plausible prediction.
5. State what is visible and what remains inferred.

## Guardrails

- A fiber is one tubular cell; a sheet/segment is a surface through a bundle.
  Do not trace a bright sheet edge as if it were one fiber.
- Mineral inclusions and reconstruction artifacts can be brighter and more
  regular than papyrus. Use continuity, wall/lumen structure, and neighboring
  slices rather than brightness alone.
- Prediction channels have different semantics. Presence, normals, and surface
  predictions are complementary; no single channel proves a seed.
- Zero may mean missing prediction support rather than air. Compare coverage
  with the base CT and representation bounds.
- Pyramid levels change what features are resolvable. Convert coordinates back
  to L0 before passing them to tools that use full-resolution voxels.
- Cut, strip, and surface panes answer different questions. A strip can show
  along-fiber continuity but cannot supply an arbitrary off-centerline volume
  coordinate.

The detailed field guide, measured scale examples, and channel-specific visual
heuristics are preserved in
[`references/image-interpretation.md`](references/image-interpretation.md).
