# overview

2d slice based fiber refinement/interpolation between long distance controlpoints in the 3d volume

# general
- work on 2d slices only (maybe 2.5D by moving a bit in z on the 2d slices?)
- use lasagna normals to extract aligned 2d slices (fiber side strips as defined in vc3d)
- very small CNN (10 layer resent)
- data is streamed+cached from s3 (compare the existing fiber tracer code vesuvius/src/vesuvius/neural_tracing/fiber_trace/)

# V0 - semi-supervised direction & curvature pretraining

a pretraining stage that does not require control points at all - but does benefit from lasagna normals
- use vc3d strip based extraction to get the side-strip views of cps
- network output: just dir (in the lasagna 2-value ambiguous format)
- slices +- some number of voxels along the z dir of the side strip as additional batch input
- losses:
    - loss on reproducing the gt direction from input fiber
- augmentations (read/slice integrated)
    - affine: rotate, skew, scale, flips
    - image: (constrast, brightness, gamma, blur, noise)
    - curve distortion: given the baseline fiber-strip coords - offset columns along y with some smooth field - e.g. a random offset every N rows, cubic interpolation between of the offsets

# V1 - short refinement self-supervision

# V2 - embedding based refinement
