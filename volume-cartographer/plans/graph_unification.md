# Rationale

- we use a lot of graph-structure within vc3d
- want/need reuse where possible
- remove desync opportunities due to multiple sources of truth

# Design questions

- APIs
- persistent storage
- in-memory structures
- distrbuted sync/conflict resolution
- import/export formats

# Design proposal

- global list (md docs) of the object types and their fields
    - json serializations for export and import of all of them
    - shared structures/fields (tags, approval, sync info, author, approval author, last_write author + timestamps)
- local graph-db for transactional processing - e.g. changes to objects possibly need corrections in connected objects (e.g. change fiber segment changes link anchor position)
- should contain all metadata of all objects (e.g. patches, fibers, links, …)
- actual data can be external to db (tifxyzs, fibers, …)
- per object
    - uuid
    - generation number
    - (?) data checksum (on a canonical data representation)?
    - (?) metadata checksum (again on a canonical serialization)
- some objects are db-only (links?)
- could split objects into metadata & storage: e.g. fibers would only be the point lists, tags etc would all live in the db.
- sync:
    - sync conflict detection needs:
        - last synced stated
        - two sides
        - if only one changed: no conlifct
        - if both change vs last sync: conflict
    - a sync process would output all changes since last sync state + the last synced state
    - PROBLEM: in distrinbuted sync “last sync” state will also desync?
- APIs need central db process and downstream on top of it, avoid any parallel access
- scope of graph structures
    - in db: metadata of
        - links
        - tifxyz/quadmesh
        - fibers
        - point/point groups (whats the object here?)
    - data (out of db)
        - tifxyz data
        - fiber data / line data
        - point group data? (probably yes because this can be large amounts auto-generated)
- out-of-scope
    - fiberlets, anchors and their crossing constraints
    - segment overlap graph (do we still use this in current workflows?)

# Notes

- persistant and globall consistent structures and designs are a poor fit to LLM design process - need to be careful

# Current state

### Fiber Chains And Branch Links

Summary: A fiber is a linear control-point/line-point chain; reciprocal branch links connect control points across fibers.

- API: Vc3dFiberJson, LineAnnotationController::StoredFiber, FiberBranchRef, and per-span segment_to_next.
- Persistent: One vc3d_fiber JSON per fiber containing geometry, span metadata, and branch records.
- In memory: Ordered point vectors plus per-fiber branch vectors; VC3D derives connected components with union-find.
- Sync/download/distribution: Branch edges are duplicated in both endpoint files and synchronized by VC3D multi-file saves. Spiral receives uploaded copies; there is no general
distributed merge.

### Segment Overlap Graph

Summary: Segments record other segments whose surfaces overlap.

- API: QuadSurface::overlappingIds() and read/write_overlapping_json().
- Persistent: Each segment owns an overlapping.json adjacency list.
- In memory: One std::set<string> per loaded QuadSurface.
- Sync/download/distribution: Tools update both endpoint files separately; symmetry is conventional, not structurally enforced. Remote segment caching copies the optional file.

### Point-Collection Graph

Summary: Point collections support point-to-point links, collection-to-collection winding links, and ordered annotation chains.

- API: PointCollections, ColPoint::links, Collection::windings_linked; Spiral point_collection.py.
- Persistent: Versioned vc_pointcollections JSON.
- In memory: VC3D keeps both collection-local and global point maps; Spiral derives point-to-patch links and ordered strips.
- Sync/download/distribution: Files are local or uploaded into a Spiral session. Spiral’s loader reconstructs its required topology and does not preserve every VC3D graph field.

### Fiberlet Graph

Summary: Anchors are nodes and fiberlets are directed geometric routes between anchors, with path and transition costs.

- API: FiberletStorage*, FiberletChunkDataset, FiberletGraph, and crop/replay graph APIs.
- Persistent: Authoritative sparse Zarr chunks for anchors, prefixes, routes, and optionally traces.
- In memory: Decoded chunk payloads or materialized contiguous node/edge/adjacency arrays.
- Sync/download/distribution: Chunk generation, caching, and write-back are coordinated within one process. There is currently no general remote graph synchronization protocol.

### Fiber Constraint And Classification Graphs

Summary: Traced fibers are split into pieces; proximity/winding measurements form pairwise edges used by pruning, BP, consensus, and H/V classification.

- API: FiberTraceConstraintReport, FiberTraceBeliefPropagation, FiberTraceConsensus, and BinaryBeliefPropagation.
- Persistent: Source traces may be stored in Zarr; constraints and solved labels are generally derived, with OBJ/JSON diagnostics rather than an authoritative graph.
- In memory: Piece vectors, pairwise constraints/factors, adjacency, and solver messages.
- Sync/download/distribution: Entirely process-local batch computation; no incremental synchronization.

### Spiral Track-Crossing Graph

Summary: Surface tracks are nodes; exact shared voxels between opposite track families create crossing edges.

- API: TrackGraph over rustworkx.PyGraph, backed by packed CSR arrays.
- Persistent: Authoritative track DBM, optional .vctracks/ packed mirror, and fingerprinted .crossings.npz CSR sidecar.
- In memory: Memory-mapped track arrays, CSR crossing metadata, and payload-free rustworkx topology.
- Sync/download/distribution: Owned by the Spiral service dataset; stale sidecars are rejected by fingerprints. VC3D does not download this graph.

### Spiral Theta/Winding Topology

Summary: Patch quads, attached point chains, fibers, and tracks contribute edges used to unwrap transformed theta consistently.

- API: ThetaCrossingMap.register_nodes(), register_edges(), unwrap trees, and compact potential sources.
- Persistent: No independent graph file; topology is rebuilt from session inputs. Model parameters live in Spiral checkpoints.
- In memory: CPU canonical edge topology plus GPU int8 crossings, node theta, and winding potentials.
- Sync/download/distribution: Service-local derived state; checkpoints and preview artifacts do not distribute the graph itself.

### Spiral Patch-Overlap Connections

Summary: Offline tooling detects overlapping patch pairs and finds cross-patch paths through stitched quad-grid graphs.

- API: connect_overlapping_patches.py, PatchGraph, SurfacePatchIndex, and SciPy CSR/Dijkstra.
- Persistent: overlap_pairs.jsonl plus a separate vc_pointcollections connection file.
- In memory: Cached per-patch CSR graphs and temporary two-patch graphs with bridge edges.
- Sync/download/distribution: Outputs are local; the point-collection file may be uploaded to Spiral, while the pair list remains a separate source that can drift.

# Details

### Dependencies

- Ladybugdb? usable similar to squlite but for graphs
- alternative sqlite + qraph extensions?

### primitives in the db

- fibers & lines (cps and intermediate points)
- patches (tifxyz quadmeshes)
- links
- anchors (separate or as part of links?)

### shared fields

- uuid
- authors
    - create
    - last_change
- generation number+author+hostname
- format version field
- tags

### Structure/Design

- database manages metadata and transactions (so controls also open-for-write access)
- database must be a central process and implement cooperative locking of the db fs data (probably supported by the chosen db)
- primitive index - need a global object index for all primitives known - including segmetns, fibers, etc.

### IO / transaction atomicity

- multi-object edits are committed as one transaction in the db
    - for those objects that have data outside the db: write new version into a temp file
    - perform the transaction (which shoulde reference the uuids + generation numbers, not filesnames)
    - atomically swap old to new versions on disk
    - this means around the transaction both old and new file is available

### serialization & export/import

- graph export
    - dbg/small graphs: json files containing the (potentially filtered) graph but not the data (just referenced by uuid)
        - filter by connectivity, tags etc, and also by which fields to include
    - large (options)
        - parquet
        - .lbdb (ladybug db - one-off for export)
        - GraphAr

### APIs

- need apis to filter, iterate etc - most functionality should operate on a database handle/id and just deliver shallow primitives (e.g. metadata copies but links to data)
- the more of our code can work with the central db interface the better, but we can add compatibility conversion at first
    - compat conversion for read-only is uncritical for write should be carefully considered or upgraded to the full db api

### reuse graph access/filtering

- all the filtering / search / iteration apis should be available also for export
    - e.g. search for some graph part, result can be used with a db handle to work on that data (possibly write) or to export to some of the supported formats

### Various things to fix / solver

- for vc3d: need a “edit” api so primitives can be opened writeable or readable
    - only few/one object should be editable at a time
    - this is only relevant for where transactional commit is not possible (e.g. mmaped / pointer access to tifxyz)

### Open Questions

- how to represent point (groups)
    - lines?
    - links with many anchors
- spatial indexes - we already use them in various places and probably should use custom ones because especially fibers and long segments need to be handled intelligently (can’t just index by bbox as that is often a singificant part of the scroll)

# Roadmap

- add a central db and primitive data indexing
- plan and document the primitives
- implement first core apis (database opening, shared metadata, links, transactional edits with external data)
- convert vc3d internal structs to db based access
    - fibers
    - davids new atlas
    - …
- convert “external” users
    - spiral
    - …?
- profit