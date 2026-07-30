# Native 3D Trace2CP Refined Presence Visualization

Change native 3D Trace2CP visualization so refined/regenerated presence slices
show presence scaled by how well the sampled predicted fiber direction lies in
the displayed strip plane.

Requirements:

- Only change visualization. Do not change tracing, metrics, candidate scoring,
  or model inference outputs.
- Apply the modulation to refined/fused/regenerated presence panels. Keep the
  original/input presence panels as raw presence for comparison.
- Do not use a straight signed dot product against one tangent vector. The
  comparison is between an ambiguous fiber direction axis and the slice's
  tangent plane, so the scale factor should be sign-invariant and based on the
  direction's projection into the displayed strip plane.
