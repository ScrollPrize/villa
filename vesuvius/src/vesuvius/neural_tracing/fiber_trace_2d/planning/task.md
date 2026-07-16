# 3D Multi-Dir TensorBoard Presence And Oblique Slice Visualization

Update the 3D multi-direction training/test TensorBoard sample sheet:

- Remove the normal-weighted closer-branch presence column.
- Add raw aggregate branch-presence columns for max, min, and average
  presence.
- Keep the close-to-slice-normal branch and other-branch raw presence columns.
- Fix the two additional oblique rows so GT line overlays and target/context
  panels are projected/rasterized in the actual oblique slice frame.
- Ensure dense-line/NML samples carry the transformed CP tangent needed to
  construct the GT-tangent and perpendicular/cross oblique rows.
