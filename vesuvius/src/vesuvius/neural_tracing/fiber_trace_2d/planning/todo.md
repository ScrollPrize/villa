# v0.1 initial fiber tracing

- [x] profiling and speed!
    - lets add bechmark mode - --benchmark should run the training skipping testing and only over the first 100 batches then report samples/s (thats "images" in the cnn - e.g. individual patches)
    - then add --profile which should measure timings by stage: coord gen, coord aug, loading (zarr read/sampling), image augs, fw, bw+step
    - should output summary after running the 100 batches on the time each stage takes avg pe sample
    - also output per sample the values (print table-like so its easy to read)
- [x] cache strip coord
    - add a separate cache (with it own config key) that caches the strip patch coords around cps
    - the cache should key by cps 3d coordiante + size-scale-step-/zarr-remote-path
    - if the patch size is changed that should still be counted as a hit if the cache entry has a >= patch size
    - if a larger patch is generated and stored in the cache that replace the smaller patch size
    - the cache should be used by training, the visualizations and prefetch - basically all strip reading should go through the same functions and they should by default use the cache as configured in the config
- [x] median of TTA for tracing
    - TTA for the tracing shows the truth mostly somewhere within the the middle
    - so lets use that to use TTA within one trace
    - if not yet existing extend the coord aug process so we can move coords fw and bw thorugh the augs
    - then we use that to implement a TTA trace - it should run in the references (unagumented) space but at each step warp the point into all TTA spaces, sample the dir there, warp it back (note the fw/bw warp needs to support warping both coords and orientations!) then in the reference space we use the median dir. note we need try both ambigurous dirs from each TTA sample (and reference) and simply discard all those that go in the other direction relative to the last step (e.g. that are more than +-90 deg off from the last step).
    - lets add a separate med-tta flag that triggers this mode - it should then be visualized as a third columne vs reference only and the flock of traces
- [ ] trace2cp vis should paint both tta (if enabled) and the reference trace (no augmentations)
- [ ] augmentations: overlay angled strips! see trace2cp_vis_3.jpg
- [ ] trace2cp its not a score its an error (hence we choose btest result as 0.0!) so fix this label in the vis
- [ ] make bidir trace2cp metric in train/test loop
- [ ] check augmentations for training: all flips and orientations should be equally likely
- [x] group norm: with a single sample per patch we seem to get a more global direction per patch! either remove the norm or do more batch-norm style normalization!
- [ ] contrastive embedding using cosine similarity
    - change cp sampling on training so that each batch does not independet cps but N cps from the same fiber M times (N=8 for now, M adjusted to fit the batch)
    - learn very simple contrastive loss with cos similarity - 8 samples around the cp shall be the same, all otheres shall be different - given the true fiber is a small subset of the whole patch this should give us some similarity metric (with the right weights)
    - weight so the few positive samples are euqally weighted to the negative ones
    - for now just apply the loss on training and visualize by showing similatiy (between 0 and 1) all points in a patch aginst that patches cp embedding (interpolated)
    - the losses should be applied positive: all points around the cps against each other within the same fiber (across batch patches, thats what the same fiber cp sampling is there for) - and negative across against potentially all other points from all patches (with lower weight it does not matter if we exclude the relevant cps) - choose the pairs in a way that scales well with the batch size - e.g. each output px is compared against a single other px from anywhere in the batch (for negative) and for positive all postive pxs of a fiber are compared against all other postive pxs (the number is probably still very small)
    - augmentations: geometric distortions should be _independent_ across fiber patches but image augmentations should be synced!
- [ ] try less skew
- [ ] we might be off in z (wrong plane in the 2d slice
    - can we additionally add uncertainty and/or optimize for maximum embedding similaity by shifting in z?
- [ ] better tracing using dijkstra or similar
- [ ] do a separate sheet/fiber recognition target
    - [ ] force embeeding similarity to be different to that?
- [x] lets try disabling skew and scale loss (maybe throws off embedding?)
- [x] for embedding similarity try best along some small z range (and short y as well)
- [ ] integrate presence in side-z along x,z axis gaussian
- [ ] 3d augmentations special cases
    - [ ] smooth distortion (dist field in 1 dim, 2 dims and 3d
    - [ ] blur in isotropic but also anisotropic (directional - 1 dir stronger two others small), and arbitrarily rorated
    - ([ ] ringing artifact?)
    - [ ] no skew?
    - [ ] special load/aug config key - round training patch to chunk boundaries
- [ ] check the various 3d augmetnations with a contact sheet
- [ ] multi-dir output for 3d fiber
    - two fiber dirs can be close together: output two dirs and two presence values, loss selects a choice (for both) that minimizes (something sensible)
- [ ] multiple laybers per patch : use some roi structure for fibers so we can supervise all fibers within some patch
- [ ] when evalutationg multi-dir& multi-presence outputs: can just do dot product against the reference angel (e.g. when tracing, slicing)
- [ ] if fibers in dense areas are randoly on top of each other we can still confuse them - use more defined fiber-center & edges to improve tracing
- [ ] cp dirs are wrong especially in tangent so beam out at additional angles in tangent initially -> can enable stronger tangent loss!
    - [ ] also test slight cp offsets along tangent - given that it might first need centering for optimal trace
- [ ] currently need significant margin (20) for artifact free processing (from patch size 64!)
- [x] for cp start: lets actually _not_ use the cp dir but the sampled dir that most closely aligns with the cp dir. The do apply smoothness and dir supervision directly as applicable
- [ ] conditioned decoder: add also from-dir conditioning (makes this closer to neural tracer decoder!) - can we maybe even unify this into a single dir condition?
- [x] fiber inference - how do we store and signal validity?
- [ ] inference size - test larger inference size quality for the pretrained model (then update defaults)
- [ ] fiber_trace_3d_inference.json ->should change the name similar to lasagna - or even use lasagna naming just with additional information?
- [ ] switch to wand from tb
- [ ] training should include the full json config used and the current git commit hash + diff to that commit of the running/current code
    - [ ] and the command run
    - [ ] same for inferences
- [ ] augment_anisotropic_blur_orientation "fiber"  - that does not sound correct ...
- [ ] dir result dependent presencec supervision? supervise with dot between gt and estimated dir?
- [ ] when we try embedding again (or other alignement ways) - focus on the left/right positioning on the fiber - that seems to be the biggest hurdle right now
- [ ] multi-fiber tracer discrete optimization/dp/...
    - trace multiple fibers (greedily)
    - they make mistakes which depend on where they come from - possibly introduce additional randomness
    - optimize sets of fibers for best explanation of the found predictions
- [ ] threshold/skeletonize fiber preds and explore the multiple discretet options for scoring!
- [ ] check training augmentations again
- [ ] --vis for fiber trace changes the quality?
- [ ] fiber jsons should from now on save the volume size they are based on (for scale reference)
    (vc_open_data_source_original_resolution and vc_open_data_source_coordinate_scale_factor)
- [ ] try a way smaller tangent angle range!
- [ ] when we look at the top strip it should basically only contain h or v fibers - can we use this to help tracing?
- [ ] if we follow a differnt dir fiber (turning not parallel) then the local h/v classification should change
- [ ] can we do a local h/v classification?
- [ ] if we turn wrong somewhere can we detect this by looking at the fiber normal?
- [ ] new fiber should not only in the gui default to the new fiber tracer but also actually use it (if available!)

# beam-search
- [ ] beamsearch
- [ ] short brute-force lookahead
- [ ] substep evaluation

# multidir
- [ ] test a loss on multi-dir outputs being perpendicular
    - [ ] and the cross product being the surface normal?
    - [ ] maybe even do three direction, perpendicular, at least one presence should be zero (that should be supervised with surface normal!)
- [ ] estimate normal (with orthongal req above?) then penalize normal curve less than tangential in the tracer!

## z-search training
- [ ] Now lets do a modification that re-introduces multiple z-slices again int the training - we see that some difficulty in the embedding is mostly caused by the sample shifting in z
  so we enable z-steps again in the config lets do for a step of 4vx and the pos loss is now not all in-fiber cps against each other but for each pair we only require the best similarity between any of the pairs offsets to be supervised. in addition given the set of cps for each fiber (8 per batch?) we also choose just the best over all cps - so the only positive supervision per cp in the batch is the most similar other cp + z slice from any other cp in the fiber in the batch (this way we should have a higher chance of actually loooking at similiar points not very different ones)
  
## z-search
- [x] z-search: after changing z layer also use that layers dir for the next step (e.g. dir should be always read from the "current" z layer
- [x] tracing: use last point + point candidate dir to evaluate dir score

    
# ideas
- [ ] short-strip self supervision?


# later optimization ml experiments
- ghost norm, seqnorm and other normalization techniques

# vc3d fiber tracing
- support extrapolation and interpolation with normal based optimizer and with trained fiber model
- global switch per fiber, which is also stored per fiber
- if the interpolation fails for the fiber model fall back to a lasagna optimize segment
- any lasagna normal based segment neighboring a trained fiber one should use that fiber based ones cp dir (calc from the next point from the cp in the fiber line) as the cp direction which should be used for optimization
- when switching the fiber-global mode from lasagna to trained fiber or beck we want to run a full opt (the existing one for lasagna) per-segment for the fiber tracing
- if there isnt yet one please add a spinbox or similar to change the extrapolation distance
