# Tracing Documentation
(a starting point)

## apps/src/vc_grow_seg_from_seed.cpp

- starting point for patch tracing - the seeding logic is here (and might need improvements/debugging)
- calls space_tracing_quad_phys from surface_helpers.cpp to run actual patch tracer

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
