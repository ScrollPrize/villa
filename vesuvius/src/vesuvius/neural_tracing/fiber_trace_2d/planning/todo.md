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
- [ ] try less shift (?)
- [ ] we might be off in z (wrong plane in the 2d slice
    - can we additionally add uncertainty and/or optimize for maximum embedding similaity by shifting in z?
    
# ideas
- [ ] short-strip self supervision?


# later optimization ml experiments
- ghost norm, seqnorm and other normalization techniques
