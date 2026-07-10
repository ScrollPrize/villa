# v0.1 initial fiber tracing

- two ways to generate/refine a fiber line
    - straight extension from the direction determined from the network (choose the alignment from the ambiguous dir encoding that minimizes the angle against the last step)
    - refinement of the existing line points
        - weight samples down by distance from the cps - so closer to cp the dir is more important
- we start with just a tool to run these two on singe points / a point and its next neighbor similiar to the augment-vis too, so initiall just for inspection
    - tool1: cp patch line tracing
        - this should work on a side-strip patch
        - and simply in both direction trace the dir based line by stepping, sampling (using bilin interpolation) the direction the network estimated at the point, then stepping again ...
        - given the receptive field we do not perform this right until the edge but only where the receptive field would touch the edge
        - for now we only want to vis one of such refinment on a sample given by the passed sample idx (which shall be the same deterministic ordering as the training/prefetch - that should also be the same as augment vis sample ordering
        - vis should show the original line (which without augs is just the centerline) as well as this dir-traced line (which might be slightly off from that)
    - tool2: segmeent refinement
        - load whole side-strip between two consecutive cps
        - 
