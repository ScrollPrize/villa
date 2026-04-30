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

- `<stem>_binary.tif`: 8-bit binary mask from a fixed inverted threshold of 127;
  dark input pixels become the foreground island/components.
- `<stem>_dt.tif`: normalized 16-bit distance transform visualization.
- `<stem>_component_voronoi_labels.tif`: 16-bit visualization of nearest
  foreground connected-component ids.
- `<stem>_component_voronoi_boundaries.tif`: boundaries where neighboring pixels
  belong to different nearest foreground components.
- `<stem>_component_voronoi_cell_loops.tif`: contour loops of each component's
  raster Voronoi cell.
- `<stem>_component_voronoi_cell_loops_connected.tif`: cell loops plus shortest
  single-pixel connector ridges between disconnected loop groups, constrained to
  the light distance domain so connectors do not cross dark islands.
- `<stem>_component_voronoi_rings.tif`: per-component Voronoi cells with the
  source component carved out, rendered as candidate rings.
- `<stem>_binary_contour_loops.tif`: hole contours from the binary foreground
  contour hierarchy.

The threshold and polarity are intentionally fixed for repeatable comparisons.
The component Voronoi path treats each dark foreground connected component as
one fat site, and computes cells/rings in the surrounding light region. The CLI
prints timings in milliseconds for the component Voronoi and contour-loop paths.

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
