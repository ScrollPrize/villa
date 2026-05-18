# dense_batch_min_cut

Small C++/OpenCV experiments for dense batched min-cut preprocessing.

## Build

Requires OpenCV with `core`, `imgcodecs`, `imgproc`, and `ximgproc`, plus
libtiff for named multipage TIFF layers.

Python callers normally do not need to run this manually. `dense_batch_flow.py`
builds the `dense_batch_flow` shared library on first use if it is missing,
using the same CMake build directory shown below. Set
`LASAGNA_DENSE_BATCH_FLOW_AUTOBUILD=0` to disable that behavior and require an
explicit manual build. The library-only build is:

```bash
cmake -S lasagna/dense_batch_min_cut -B lasagna/dense_batch_min_cut/build
cmake --build lasagna/dense_batch_min_cut/build --target dense_batch_flow
```

To build the CLI used below as well, omit `--target dense_batch_flow`.

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
- `<stem>_source_rim_ridges.tif`: ridge candidates where neighboring
  source-pixel labels map to sufficiently distant points along the source rim.
  The rim-label pass expands source islands by one pixel and pads the image
  with a one-pixel source rim while preserving the unmodified distance
  transform for graph capacities.
- `<stem>_source_rim_distance.tif`: max along-rim distance used by the ridge
  candidate test.
- `<stem>_source_rim_arc.tif`: source-rim arc-position visualization used to
  debug rim lookup and graph construction.
- `<stem>_source_rim_arc_skeleton.tif`: thinned skeleton of the arc-position
  visualization.
- `<stem>_source_rim_skeleton.tif`: thinned source-rim ridge skeleton used for
  graph extraction after frame cleanup.
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
- `<stem>_graph_edges_random.tif`: edge-only version of the deterministic
  pseudo-random graph edge visualization.
- `<stem>_graph_components_random.tif`: graph visualization where every
  connected graph component gets one deterministic pseudo-random color on a
  black background. Components pruned from the computational graph are still
  overlaid in this debug image with separate colors.
- `<stem>_graph_nodes.tif`: node-only graph visualization.
- `<stem>_graph_capacity.tif`: graph edges rendered in grayscale by edge
  capacity, where capacity is the minimum raw distance-transform value along
  the complete traced edge.
- `<stem>_graph_capacity_normalized.tif`: graph edges rendered in normalized
  grayscale capacity, where the maximum graph edge capacity is 255 in the image
  output and 1.0 in the layered TIFF page.
- `<stem>_graph_components.txt`: graph connectivity report with component node
  and edge counts, plus counts for self-loop, one-endpoint, and zero-endpoint
  edges. It also reports skeleton coverage: total skeleton pixels, node pixels,
  unique edge pixels, edge path pixels, missing pixels, and whether any
  extracted graph components still touch in the raster.
- `<stem>_dense_flow.tif`: optional 32-bit float dense source-flow map, written
  only when `--source x,y` is provided.
- `<stem>_dense_flow_u16.tif`: optional normalized 16-bit visualization of the
  dense source-flow map.
- `<stem>_graph_edge_flow.tif`: optional graph-edge visualization of propagated
  source flow.
- `<stem>_tree_dense_flow_no_backtrack.tif`: optional dense source-flow debug
  image before the route/backtracking pass. Compare this with
  `<stem>_tree_dense_flow.tif` to inspect the backtracking effect.
- `<stem>_nearest_graph_flow.tif`: optional dense source-flow debug image that
  assigns every white-domain pixel the flow value of its nearest graph pixel.
- `<stem>_tree_dense_flow_greedy_ascent.tif`: optional dense source-flow debug
  image that starts from the no-backtrack flow field, greedily walks to the
  highest neighboring larger flow value, and stops at the first local maximum.
- `<stem>_graph_source_edges.tif`: optional diagnostic showing the graph edges
  selected by source-edge ascent and used as flow sources.
- `<stem>_flow_gate_weight.tif`: optional dense gate-weight image computed as
  local greedy-ascent flow divided by the dilated local maximum of that same
  greedy-ascent flow. The dilation radius is `--backtrack-distance`.
- `<stem>_layers.tif`: named multipage TIFF for easier inspection in GIMP.
  The pages include `binary_threshold`, `dt`, `source_rim_distance`,
  `source_rim_arc`, `source_rim_arc_skeleton`, `loops_connected`,
  `graph_random_edges`, `graph_edges_random`, `graph_components_random`,
  `graph_nodes`, `graph_capacity`, and `graph_capacity_normalized`. When
  `--source x,y` is provided, the pages `tree_dense_flow_no_backtrack`,
  `nearest_graph_flow`, `tree_dense_flow_greedy_ascent`, `tree_dense_flow`,
  `graph_edge_flow`, `flow_gate_weight`, and related flow debug layers are
  appended.

The threshold and polarity are intentionally fixed for repeatable comparisons.
The component Voronoi path treats each dark foreground connected component as
one fat site. The CLI prints a fixed-width timing table with elapsed time, CPU
time, and estimated CPU/elapsed utilization for the main stages and component
Voronoi substages.
The rim-label pass expands source islands by one pixel, keeps only the expanded
white component containing `--source`, pads the image before computing rim
labels, then crops the debug layers back to the input size. This removes
ambiguous one-pixel rim pockets and post-expansion disconnected pockets from the
rim-distance model while leaving the original distance transform in place for
graph capacities. Multiple source-rim contours in one white connected component
are allowed; disconnected source islands are treated as infinitely far apart
along the rim, so their Voronoi boundary is kept. Every nearest-rim source pixel
selected by the labeled distance transform must still map to one rim contour.
Missing rim assignments mean the border-connected rim model is broken; the CLI
writes the usual debug outputs and exits non-zero with the component summary.
The extracted graph is expected to be one connected component; disconnected
graphs write the usual debug outputs and then exit non-zero with the component
summary.
As a regression check for graph extraction, raster-adjacent skeleton pixels
must not belong to different extracted graph components. The extractor repairs
those local misses by promoting the touched edge pixels to graph nodes and
adding a local connector edge; any remaining adjacent-component contact is an
error before the broader connectivity check. Detached non-largest graph
components are pruned after that repair; these are isolated skeleton islands,
not raster-adjacent graph extraction misses.

`--source x,y` must point into the light/white distance domain, not into a dark
foreground island. The source is snapped onto the nearest graph edge, that edge
is split at the snapped pixel, and the split node is seeded with the local graph
capacity. Graph edges touching that split node are marked in
`<stem>_graph_source_edges.tif`. The small extracted graph is then evaluated
with exact per-node max-flow, edge flow is assigned from the maximum of its
endpoint flows, and dense pixels take the minimum of their raw DT value and the
flow at the nearest graph-edge pixel.

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
