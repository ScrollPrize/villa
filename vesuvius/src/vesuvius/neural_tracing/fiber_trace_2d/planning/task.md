# Trace2CP Side Z-Axis Correction

- Fix Trace2CP side-strip z-search so the z axis is the normal of the
  side-strip surface, not the side-strip row/y axis.
- In side-strip coordinates, x follows the fiber/tangent and y follows the
  Lasagna mesh-normal/row direction. Z-search must move along the remaining
  out-of-plane side axis, roughly orthogonal to side-strip y.
- Apply this consistently to all Trace2CP side z-layer users:
  - regular stepwise combined z-search;
  - the joint/combined side search path;
  - explicit side DP z-search;
  - the side/top z experiment where it uses side z offsets;
  - z-corrected debug visualizations and exports derived from those traces.
- Do not patch the visualization to hide the issue. The sampled side z-layers
  themselves must be generated with the correct 3D axis.
- Keep Lasagna normal handling and ambiguous normal sign alignment unchanged.
- Preserve existing non-z Trace2CP behavior.
