# Task: rolling live OME-Zarr input cache for shared 3D inference

Add an explicitly enabled live-fetch mode to the shared Fiber/Lasagna 3D tiled
inference path. Live mode must fetch selected-input OME-Zarr chunks ahead of
the canonical inference traversal and conservatively remove cached chunks
behind that traversal so full-volume inference can run without materializing
the complete selected scale.

Requirements:

- The implementation is shared by Fiber 3D and regular Lasagna predict3d.
- Live mode is opt-in and applies only to full, non-cropped inference of one
  numeric OME-Zarr scale with valid `_download` source metadata.
- The selected scale is the only mutable/deletable cache content. Other scales,
  OME metadata, `.dl_cache`, and unrelated files must never be deleted.
- The live disk-cache target defaults to 10 TiB. It is conservative: exceeding
  the target is acceptable whenever no complete safe Z plane can be removed.
- Fetch lookahead defaults to 10,000 canonical inference tiles and remains
  bounded/lazy; no full Cartesian tile-job list may be created.
- Eviction is only by complete input Zarr Z-chunk planes. When the cache exceeds
  its target, remove the oldest eligible plane strictly behind the current
  relevant input Z band, recheck the projected size, and repeat until under the
  target or no eligible plane remains. Never evict ahead of inference, from the
  live/read-ahead band, or by Y/X/LRU selection.
- Existing bulk prefetch, cropped auto-download, output resume, TensorStore
  reads, multi-GPU scheduling, and numerical output must retain their behavior
  when live mode is disabled.
- The manager must be able to launch live-fetch inference without first doing a
  complete scale prefetch.
