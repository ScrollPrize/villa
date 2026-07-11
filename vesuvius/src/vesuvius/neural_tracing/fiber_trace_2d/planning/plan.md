# overview

2d slice based fiber refinement/interpolation between long distance controlpoints in the 3d volume

# current overview plan
- add trace2cp metrics in training (full test fiber pairs) - minus the ones that go out of the FOV (or increase strip height?)
- add gt dir refinement
- add actuall embedding similarity
- add to vc3d?
- gt positions refinement (from emdedding) 

# general
- work on 2d slices only (maybe 2.5D by moving a bit in z on the 2d slices?)
- use lasagna normals to extract aligned 2d slices (fiber side strips as defined in vc3d)
- very small CNN (10 layer resent)
- data is streamed+cached from s3 (compare the existing fiber tracer code vesuvius/src/vesuvius/neural_tracing/fiber_trace/)

# various details

## trace2cp error
- load the strip between two cps (with some margin to spare
- run TTAs on the strip (leave out y shift and scales augs as thoase are hard to handle for long strips)
- then run a trace starting at one point (initial dir should be aligned to point towards the second cp of course)
    - this should share the code with the existing tracer, just some flags/args for the differences
- the score is the distance (in y) against the second cp when reaching the second cp columns
- 0 means we hit it exactly, 1.0 means we are at the strip edge (or we hit the edge before we could reach the tgt column in which case we should also stop the trace and assign error 1.0)

# V0 - semi-supervised direction 

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
    
# V0.1 initial fiber refinement and extrapolation tooling
- adding tooling to trace using the trained network dir output
- or refine the segment between to cps

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
    - tool2: segment refinement
        - this is already partially implemented by trace2cp metric
        - just extent the relevant code (it MUST be shared) to give us a full cp to cp line init like this:
            - trace both dirs
            - find the point with the smallest vertical distance between the two traces
            - correct both traces linearly by warping them so their start point is unchanged and we linearly warp the line more the closer (in x) we get to that closest point
            - at the cloest points both lines should be warped to reach the center betweent the two lines
            - the warping should only be vertically - linearly blending between zero correction at the cps and full correction at the cloest point
            - given the steps do not necessarily aling in pixels resample the lines before this so we have one line point per horizontal pixel
            - then afterwards resample the new combined line so we use the configured step again (or the closest given it might not fit perfectly) - note that this step is along the line not x only
        - trace2cp metric should actually show additional rows with:
            - the partiall lines only going until the cloest point (both in one row)
            - the newly constucted fused line going from cp to cp
        - finally we re-optimize the line by minimizing the dir errors of the sampled points while at the same time applying a loss the distributes the step evenly
            - visualize the opt result in a fourth row in trace2cp
        - the trace2cp metric calculation should actually be changed to be based on the min distance between the two traces instead of the distance at the cps
            
            
# V0.1 gt self-refinement

- the augmentations gives us a bit of room to try self-refinement of the gt because the gt itself is quite errorneous
- we want to allow to correct the gt using inferred values then supervise a larger weight against the refined gt and a smaller against the original gt (basically its just the calibration)
- implementation:
    - use two dir losses - gt and gt_mod
    - gt_mod is initialized as gt
    - at each training step we adjust gt_mod towards the inferered direction (by some small step - lets do 1 degree for now)
    - vis additionaly the gt_mod as another short line (slightly longer than model output and behind it - and diff color of course)
    - the rotation we apply is to the whole line as a rotation (in 2d) centered on the cp
    - per sample we store the current gt_mod rotation - so it will always be applied after loading the line and before applying the geometric augmentations
    - dir supervision of gt and gt mod should then for both different losses by calculated from the transformed lines (after augmenations) - this should already be the case?
    - this should also support working together with multiple per-patch cp:
        - if strip_z_offset_count is > 1 then we avg the angle offset of each result vs gt_mod so we get a cleaner correction signal
        - we specifically also suport strip_z_offset_step of 0.0 so we get multiple agumentations of the exact same cp without any z offset
        
# V1 - short refinement self-supervision

# V2 - embedding based refinement
