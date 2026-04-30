# dense_batch_min_cut

Small C++/OpenCV experiments for dense batched min-cut preprocessing.

## Build

Requires OpenCV with `core`, `imgcodecs`, and `imgproc`.

```bash
cmake -S lasagna/dense_batch_min_cut -B lasagna/dense_batch_min_cut/build
cmake --build lasagna/dense_batch_min_cut/build
```

## Run

```bash
cd path/to/workdir
/path/to/villa2/lasagna/dense_batch_min_cut/build/dense_batch_preprocess \
  -i path/to/image.tif
```

Outputs:

- `<stem>_binary.tif`: 8-bit binary mask from a fixed threshold of 127.
- `<stem>_dt.tif`: normalized 16-bit distance transform visualization.
- `<stem>_skeleton_dt_ordered.tif`: 8-bit distance-ordered topology-preserving
  thinning of the binary foreground.
- `<stem>_ridges_voronoi_labels.tif`: global approximate medial ridges from
  OpenCV distance-transform labels.
- `<stem>_ridges_voronoi_same_cc.tif`: global label ridges, but neighbor checks
  are restricted to the same foreground connected component.
- `<stem>_ridges_cc_voronoi.tif`: per-foreground-component labeled distance
  transform ridges.
- `<stem>_ridges_cc_voronoi_angular.tif`: per-component ridges filtered to keep
  only pixels whose nearest boundary sites are separated by at least 120 degrees.
- `<stem>_ridges_cc_voronoi_angular_2core.tif`: thinned angular per-component
  ridges with iterative leaf pruning, intended to keep cyclic/core structures.

The threshold is intentionally fixed for repeatable comparisons. Skeletonization
operates on the binary foreground; the distance transform is used for
visualization and to order candidate removals in the distance-ordered variant.
The CLI prints timings in milliseconds for each experimental path.

## Candidate Optimizations

Ideas to test later:

- Boundary-only initialization for distance-ordered thinning, so interior pixels
  enter the queue only after becoming exposed.
- Bucket queue over quantized distance values instead of `std::priority_queue`.
- OpenCV labeled distance-transform ridges, using nearest-background label
  changes as an approximate medial axis.
- Chamfer/integer distance transforms (`DIST_MASK_3` or `DIST_MASK_5`) if metric
  approximation is acceptable.
- A single `mask -> removable` lookup table for endpoint and topology rules.
- Two-phase approximation: fast DT/label ridge candidates, then thinning or
  pruning only around candidate regions.
- Tile processing with halos for parallelism, with explicit boundary
  reconciliation.
- Connected-component split and parallel skeletonization per component.
- ITK or another proven implementation as a performance/correctness baseline.
