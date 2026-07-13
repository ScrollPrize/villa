# Trace2CP Side Z-Axis Correction Task Log

## Planning Notes

- Current side strips use the Lasagna mesh-normal field for the image row axis.
- The current z-plane cache shifts side strips through the same field, so
  Trace2CP z-search moves along side image y instead of the side-strip
  out-of-plane axis.
- Regular stepwise z-search and the explicit side DP backend share
  `_Trace2CpZPlaneCache`, so a central layer-construction fix should cover both.
- The side/top z experiment has a separate local top-patch path and must be
  updated to use the same side-z interpretation.

## Validation

- Not run yet; this is the planning stage.
