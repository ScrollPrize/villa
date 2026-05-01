# dense_batch_min_cut

Small C++/OpenCV experiments for dense batched min-cut preprocessing.

## Build

Requires OpenCV with `core`, `imgcodecs`, `imgproc`, and `ximgproc`, plus
libtiff for named multipage TIFF layers.

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

Optional dense source-flow estimation:

```bash
cd path/to/workdir
/path/to/villa2/lasagna/dense_batch_min_cut/build/dense_batch_preprocess \
  -i path/to/image.tif --source x,y
```

Outputs:

- `<stem>_binary.tif`: 8-bit binary mask from a fixed inverted threshold of 127;
  dark input pixels become the foreground island/components.
- `<stem>_dt.tif`: normalized 16-bit distance transform through the light domain
  to the nearest dark foreground island.
- `<stem>_component_voronoi_labels.tif`: 16-bit visualization of nearest
  foreground connected-component ids.
- `<stem>_component_voronoi_boundaries.tif`: boundaries where neighboring pixels
  belong to different nearest foreground components.
- `<stem>_component_voronoi_boundary_skeleton.tif`: one-pixel thinning of the
  dense Voronoi boundary mask.
- `<stem>_component_voronoi_boundary_skeleton_pruned.tif`: boundary skeleton
  with short dead-end spurs removed when their maximum raw DT value is below the
  fixed pruning threshold.
- `<stem>_source_pixel_voronoi_ridges.tif`: the older dense ridge detector:
  pixels where neighboring OpenCV labeled-DT source-pixel ids differ.
- `<stem>_source_pixel_voronoi_ridge_skeleton.tif`: reserved diagnostic output.
  It is currently written as a blank image because the graph connector uses the
  dense source-pixel ridges directly.
- `<stem>_component_voronoi_boundary_skeleton_hybrid.tif`: clean pruned
  component-boundary skeleton plus selected source-pixel ridge connector pieces.
- `<stem>_component_voronoi_cell_loops.tif`: reserved diagnostic output for
  raster Voronoi-cell contour loops. It is currently written blank in the fast
  path because graph extraction does not use it.
- `<stem>_component_voronoi_cell_loops_connected.tif`: clean pruned
  component-boundary skeleton plus selected source-pixel ridge paths. The path
  search runs on the dense source-pixel ridge mask, not the thinned diagnostic,
  and maximizes the minimum DT value along the path, with shorter paths used as
  the tie-breaker. Short attachment segments are drawn to close 1-pixel gaps
  between selected paths and the clean skeleton.
- `<stem>_component_voronoi_rings.tif`: reserved diagnostic output for
  per-component Voronoi cells with the source component carved out. It is
  currently written blank in the fast path because graph extraction does not use
  it.
- `<stem>_binary_contour_loops.tif`: hole contours from the binary foreground
  contour hierarchy.
- `<stem>_graph_random_edges.tif`: graph visualization extracted from
  `<stem>_component_voronoi_cell_loops_connected.tif`; graph nodes are small
  circles at connected junction clusters, and edges are deterministic
  pseudo-random colors.
- `<stem>_graph_capacity.tif`: graph edges rendered in grayscale by edge
  capacity, where capacity is the minimum raw distance-transform value along
  the complete traced edge.
- `<stem>_dense_flow.tif`: optional 32-bit float dense source-flow map, written
  only when `--source x,y` is provided.
- `<stem>_dense_flow_u16.tif`: optional normalized 16-bit visualization of the
  dense source-flow map.
- `<stem>_graph_edge_flow.tif`: optional graph-edge visualization of propagated
  source flow.
- `<stem>_graph_source_edges.tif`: optional diagnostic showing which graph edges
  were treated as directly adjacent to the source region and therefore seeded
  with internal infinite capacity.
- `<stem>_layers.tif`: named multipage TIFF for easier inspection in GIMP.
  The pages are `binary_threshold`, `dt`, `loops`, `loops_connected`,
  `graph_random_edges`, and `graph_capacity`. When `--source x,y` is provided,
  the pages `dense_flow`, `graph_edge_flow`, and `graph_source_edges` are
  appended.

The threshold and polarity are intentionally fixed for repeatable comparisons.
The component Voronoi path treats each dark foreground connected component as
one fat site. The CLI prints a fixed-width timing table with elapsed time, CPU
time, and estimated CPU/elapsed utilization for the main stages and component
Voronoi substages.

`--source x,y` must point into the light/white distance domain, not into a dark
foreground island. The graph edge(s) bordering the source region are seeded with
internal infinite capacity. The small extracted graph is then evaluated with
exact per-node max-flow, edge flow is assigned from the maximum of its endpoint
flows, and dense pixels take the minimum of their raw DT value and the flow at
the nearest graph-edge pixel.

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
