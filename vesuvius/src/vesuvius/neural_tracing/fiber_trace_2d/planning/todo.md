# v0.1 initial fiber tracing

- profiling and speed!
    - lets add bechmark mode - --benchmark should run the training skipping testing and only over the first 100 batches then report samples/s (thats "images" in the cnn - e.g. individual patches)
    - then add --profile which should measure timings by stage: coord gen, coord aug, loading (zarr read/sampling), image augs, fw, bw+step
    - should output summary after running the 100 batches on the time each stage takes avg pe sample
    - also output per sample the values (print table-like so its easy to read)
- cache strip coord
    - add a separate cache (with it own config key) that caches the strip patch coords around cps
    - the cache should key by cps 3d coordiante + size-scale-step-/zarr-remote-path
    - if the patch size is changed that should still be counted as a hit if the cache entry has a >= patch size
    - if a larger patch is generated and stored in the cache that replace the smaller patch size
    - the cache should be used by training, the visualizations and prefetch - basically all strip reading should go through the same functions and they should by default use the cache as configured in the config
- median of TTA for tracing
    - TTA for the tracing shows the truth mostly somewhere within the the middle
    - so lets use that to use TTA within one trace
    - if not yet existing extend the coord aug process so we can move coords fw and bw thorugh the augs
    - then we use that to implement a TTA trace - it should run in the references (unagumented) space but at each step warp the point into all TTA spaces, sample the dir there, warp it back (note the fw/bw warp needs to support warping both coords and orientations!) then in the reference space we use the median dir. note we need try both ambigurous dirs from each TTA sample (and reference) and simply discard all those that go in the other direction relative to the last step (e.g. that are more than +-90 deg off from the last step).
    - lets add a separate med-tta flag that triggers this mode - it should then be visualized as a third columne vs reference only and the flock of traces
