# Native winding graph

`spiral_graph` is the non-rendering replacement for the graph-construction and
consistency-checking part of `plot_winding_graph.py`. Graph algebra, TIFXYZ
topology, attachment, JSON/fiber parsing, packed-track I/O, crossing transport,
rollback, conflict witnesses, and persistence are C++. Python is used only to
reconstruct a Spiral checkpoint and run batched Torch inference.

The graph stores two exact integer transports on every relative equation:
the reported transport derived from model theta/annotations, and an
independent geometric transport derived from raw polar angle around the
configured umbilicus. A nonzero relative cycle is retained as holonomy on the
integer lift of wrapped theta: the cycle returns to another winding-sheet copy
of the same patch. It is inconsistent only when its reported and geometric
holonomies differ. Local patch or source theta inconsistencies and
contradictory absolute anchors remain hard errors, and every add remains
transactional.

The continuous coordinate represented by the inputs is
`adjusted_turn = sheet - patch_theta_potential + theta / (2*pi)`. The theta
provider already supplies the fractional term, so the graph keeps the sheet
and seam transport as exact integers rather than accumulating floating-point
angles.

## Build and test

The normal project build includes the extension:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Installing the project exposes the `spiral_graph` Python package. The existing
`pyproject.toml`/`uv sync` path also builds it through scikit-build-core.

## Python use

```python
from spiral_graph import InputRole, SpiralThetaProvider, WindingGraph

theta = SpiralThetaProvider(
    "/data/run/checkpoint_fitted.ckpt",
    umbilicus="/data/scroll/umbilicus.json",  # optional when auto-discoverable
    device="cuda",
)

graph = WindingGraph.create("/data/cache/winding-graph", theta)
result = graph.add_patches(["/data/scroll/patches"])
assert result.committed

# Cached patches can be disabled as contact targets without discarding their
# cached geometry/theta. Validity must be chosen before a dependent source is
# committed; it is stored in the cache on the next save.
graph.set_patch_valid("suspect-patch-uuid", False)

for path, role in [
    ("/data/abs_winding.json", InputRole.ABSOLUTE),
    ("/data/relative_winding.json", InputRole.RELATIVE),
    ("/data/same_winding.json", InputRole.SAME_WINDING),
]:
    result = graph.add_point_collections([path], role)
    if not result.committed:
        break

if result.committed:
    result = graph.add_fibers("/data/fibers")

if not result.committed:
    conflict = result.conflict
    print(conflict.kind, conflict.residual)
    for edge in conflict.cycle:
        # node_name applies to graph/anchor conflicts. PATCH_THETA and
        # SOURCE_THETA witnesses use source-local topology node numbers.
        print(
            edge.from_node,
            edge.to_node,
            edge.delta,
            edge.provenance.source,
            edge.provenance.item,
            "closing" if edge.closing else "tree",
        )

# Saving is explicit so a bulk construction does not rewrite a growing cache
# after every append.
graph.save()
```

Open a saved graph without loading Torch when only queries and diagnostics are
needed:

```python
graph = WindingGraph.open("/data/cache/winding-graph")
print(graph.stats().patch_count, graph.stats().constraint_count)

# The representative is selected by a deterministic spanning-tree gauge.
# period == 0 means it is unique. Otherwise all values
# representative + k*period are reachable on different winding sheets.
lifted = graph.lifted_relative_winding("patch-a", "patch-b")
print(lifted.representative, lifted.period)

# Retained cycles remain available as full provenance witnesses.
for index in range(min(graph.stats().holonomy_count, 10)):
    cycle = graph.holonomy(index)
    print(
        cycle.reported_holonomy,
        cycle.geometric_holonomy,
        cycle.inconsistency,
        cycle.closing_constraint.provenance.item,
    )

# This compact form avoids reconstructing every witness path and is suitable
# for checking hundreds of thousands of cycles.
bad = [audit for audit in graph.holonomy_audits() if audit.inconsistency]
print(f"{len(bad)} geometrically inconsistent cycles")
```

`lifted_relative_winding()` is the only relative query. It returns the compact
equivalence class of the infinite lifted graph rather than pretending that one
global integer exists for every physical patch.

Holdout point collections can be attached and converted to root-frame
constraints without mutating or saving the graph. This is useful for measuring
consistency against independent geometry:

```python
graph = WindingGraph.open("/data/cache/winding-graph", theta)
for constraint in graph.inspect_point_collections(
    ["/data/holdout-relative.json"], InputRole.RELATIVE
):
    source = graph.node_name(constraint.from_node)
    target = graph.node_name(constraint.to_node)
    relation = graph.lifted_relative_winding(source, target)
    if relation is None:
        print(source, target, "disconnected")
        continue
    gauge_residual = relation.representative - constraint.delta
    residual = (
        gauge_residual if relation.period == 0
        else gauge_residual % relation.period
    )
    print(source, target, constraint.delta, relation.period, residual)
```

`benchmarks/evaluate_winding_graph_consistency.py` builds these holdouts from
named `wNN_*` TIFXYZ meshes and from sampled rows/columns of large continuous
reference meshes. It opens an existing cache and never adds patches, replays
registered sources, or calls `save()`.

Pass a theta provider to `open` only when more geometric inputs will be added.
Cache open validates the version and fingerprints every registered patch and
source rather than silently using changed inputs. `SpiralThetaProvider` also
records a key derived from the checkpoint and umbilicus files; reopening for
further geometric additions rejects a different checkpoint provider.
Opening a patch-only version-1 cache with `SpiralThetaProvider` computes and
caches the missing polar patch phase without rerunning checkpoint inference.
Version-1 caches that already contain constraints must be rebuilt from their
sources because no independent geometric transport was stored for those
constraints.

## Component layout export

A saved cache is exported from its largest approved-fiber component. The
checkpoint is required only to orient positive `u` and identify one coherent
theta seam. Integer winding is the component-relative lift obtained by walking
oriented H fibers and approved fiber links from a deterministic H root; V
fibers transport that gauge without creating turns along `z`. Raw polar theta
around the umbilicus supplies the fractional winding field.

```sh
python -m spiral_graph.export_component \
    --cache /data/cache/winding-graph \
    --checkpoint /data/run/checkpoint_fitted.ckpt \
    --output /tmp/component-layout \
    --spacing 20
```

H fibers run along rows/`u`; V fibers run along columns/`v`, and increasing
`v` follows physical `z`. Same-axis links form logical tracks and H/V links
form crossing knots. A robust Ceres solve keeps `u` and `v` in physical voxel
units. Patches are then registered by deterministic rigid/reflected RANSAC and
Ceres refinement, first against fibers and then against already placed patch
overlaps. A second global Ceres pose-graph solve closes patch-overlap cycles;
direct fiber contacts are its absolute anchors. Only overlap edges that agree
with the fiber-carried unwrapped seam turn enter that solve. Graph constraints,
graph-reported turns, geometric theta, and patch component membership do not
participate.

The outputs are `overview.png`, `layout.json`, `patch_index.tif`, int32
`winding.tif`, float32 `fractional_winding.tif`, and `surface.tifxyz/`. Raster
row zero is maximum `v` (normally the highest physical `z`). Agreeing overlaps
are arithmetically averaged; samples with an XYZ disagreement over two voxels
or an integer-winding disagreement are invalidated and recorded. Output quads
must be continuously supported by an input patch. Pose-graph edges are
re-gated after refinement, and patches disconnected from all fiber anchors are
quarantined before a clean re-solve. Raster overlap is evaluated in descending
registration quality: a patch with at least 16 overlap samples is quarantined
when incompatible samples outnumber agreeing samples, while a smaller local
defect is masked without discarding the patch's good area. All quarantines and
remaining local conflicts are recorded in `layout.json`. The destination must
not already exist, and the default maximum is 100 million raster samples.

## Packed tracks

Tracks require the packed `.vctracks` directory and its uncompressed
`.crossings.npz` sidecar. Build the reusable point index separately; its
external merge sort is bounded by `memory_budget_bytes` and the result is
memory-mapped by later graph builds.

```python
tracks = "/data/tracks/scroll.dbm.vctracks"
index = "/fast-cache/scroll.dbm.winding-index"

info = graph.prepare_track_index(
    tracks,
    index,
    cell_size=32,
    memory_budget_bytes=512 << 20,
)
print(info.points, info.cells, info.already_present)

result = graph.add_tracks(
    tracks,
    "/data/tracks/scroll.dbm.crossings.npz",
    index,
)
```

If sidecar paths are omitted, `add_tracks` uses the adjacent
`<stem>.crossings.npz` and `<stem>.winding-index` names. Track coordinates and
crossings are memory-mapped. Theta is inferred in bounded batches, reported
and geometric local potentials use two temporary disk-backed int32 arrays,
and the billion-point
contact candidate set uses one bit per point. A late patch replays registered
sources against the new surface index but filters already-present assertions,
so graph constraints do not multiply on every replay.

## Theta crossings

Patch transport uses the same compact valid-quad topology as
`PatchSamplingAtlas` and the same rule as `ThetaCrossingMap` in `fit_spiral.py`:

```text
step = (theta_delta > pi) - (theta_delta < -pi)
```

It computes root-relative int32 potentials on the atlas tree, applies the final
quad-centre-to-fractional-point crossing, and validates all non-tree neighbors.
This replaces the old plot script's per-query Dijkstra strip construction.

## Performance benchmark

The reproducible graph-core comparison uses 80,000 patches and 547,115
consistent constraints:

```sh
python benchmarks/benchmark_winding_graph_python.py \
    --nodes 80000 --edges 547115 --repeats 5
build/bin/benchmark_winding_graph_core 80000 547115 5
```

On the development aarch64 Linux host, the Release native core averaged
0.137304 s over five runs (0.116511 s minimum, 0.133355 s median), versus
0.354142 s for the equivalent pure-Python weighted-union baseline: 2.58x
faster. This measures graph mutation only. The one-time track-index build and
checkpoint inference/contact scan are deliberately separate stages.

The full 1,014,441,480-point production track scan was not timed because the
implementation session's execution sandbox did not expose the host's CUDA
devices. Its metadata and crossing mappings were opened successfully
(19,746,134 tracks and 100,799,388 directed crossing records), but the
end-to-end five-minute target must still be measured in a process with access
to the host GPU and production storage.

For that scale, budget scratch storage as a separate resource: the two local
track-potential arrays require eight bytes per point, the final point-id index requires
eight, and external-sort runs use twenty bytes per point while the index is
being built (plus the occupied-cell table). The index records the coordinate
file size and modification time and refuses reuse after the track source
changes.

The fast patch loader currently requires directly mappable, uncompressed,
single-plane float32 strip TIFFs and rejects `mask.tif`; compressed or masked
TIFXYZ is not silently loaded into a large resident fallback. Add the complete
patch set in one call when possible. Late patches are exact and supported, but
registered sources must be replayed to discover new contacts, so repeatedly
adding individual patches after a billion-point track source is intentionally
more expensive than a bulk patch append.
