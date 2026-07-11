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
- [ ] group norm: with a single sample per patch we seem to get a more global direction per patch! either remove the norm or do more batch-norm style normalization!


- Not quite before. After your question I re-scanned and found one remaining legacy dense helper in augmentation.py plus a test-only pseudo-inverse helper.

  Now removed:

  - _nearest_output_pixels_for_source_points
  - _transformed_source_point_coords_affine
  - _transformed_centerline_coords_affine
  - _reference_direction_to_source_grid_direction with np.linalg.pinv
  - stale “invert CP” wording

  So for geometric augmentation / TTA map inversion: yes, the search/bruteforce/after-the-fact inversion code is gone. The code now uses the explicitly built paired maps.

  Remaining argmin hits are unrelated to augmentation-map inversion:

  - pick nearest line vertex to CP for tangent estimation
  - choose nearest strip-z offset for display/training bookkeeping
  - choose an axis in strip geometry

  Re-ran:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py

  Result: 115 passed in 5.69s.



# later optimization ml experiments
- ghost norm, seqnorm and other normalization techniques
