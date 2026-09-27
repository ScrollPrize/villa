# Tracing Documentation
(a starting point)

## apps/src/vc_grow_seg_from_seed.cpp

- starting point for patch tracing - the seeding logic is here (and might need improvements/debugging)
- calls space_tracing_quad_phys from surface_helpers.cpp to run actual patch tracer

### Post-growth acceptance check (#1675)

After tracing, the tool samples the input prediction at each valid grown mesh
vertex (native frame) and reports the **on-prediction support** fraction. A
surface that follows its prediction scores ~100%; one that cut across windings
instead of following a sheet scores ~6-10% (indistinguishable from random
points in the volume). Alongside the fraction, the tool reports the **background
rate**: the same fraction measured on 2000 uniform random points in the
surface's neighborhood (bounding box of valid vertices, dilated by 64 voxels,
fixed seed). A surface at or below the background rate follows the prediction
no better than chance and triggers its own warning; this catches
dense-prediction cases where even a random surface would clear the absolute
threshold. Sampling is confined to the neighborhood so its chunks are the ones
the growth already loaded -- volume-wide scattered reads would cost up to
~2000 extra chunk fetches on remote (http/s3) volumes. Params:

- `min_on_prediction_support` (default 0.5): warn when the fraction is below this.
  Must be within [0, 1]; anything else is rejected before tracing starts.
- `require_on_prediction_support` (default false): when true, a surface below
  the threshold (or no better than background) is discarded and the tool exits
  non-zero instead of just warning.

The check is warn-only by default so sparse but legitimate predictions are not
rejected; opt into strict mode once the threshold is validated for a pipeline.

## space_tracing_quad_phys() (surface_helpers.cpp)

- general process: optimize a surface from a thresholded surface prediction (using CachedChunked3dInterpolator<uint8_t,thresholdedDistance> interp(proc_tensor))
- cv::Mat_<uint8_t> state(size,0) - maintain a state of the current surface corners 
- general tracing loop:
    - outer loop:
        - loop: add corners greedily (for several iterations)
        - optimize globally / optimzed windowed (large "active" edge area of the trace)
- we use a bunch of heuristics to decided when to accept some solution and go on and when to skip

## How losses operate
    - loss generation functions are somewhat "region aware" - functions get supplied with global state array as well as the corner idxs and global corner array and operate on that. 
    - check out emptytrace_create_missing_centered_losses - recurses into various losses
    - unconditional losses: e.g. gen_straight_loss() -> generates a straightness loss for o1,o2,o3 three points, based on the supplied data and state
    - conditiona loss: conditional_straight_loss() -> generates the straightness loss only if the loss position is not marked as in-use already - and marks the location as used

## Where next

- look at the code and comments in surface_helpers.cpp
- ask in https://discord.com/channels/1079907749569237093/1243576621722767412
