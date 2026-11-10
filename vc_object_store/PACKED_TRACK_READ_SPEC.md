# Packed track read for Spiral — spec

Status: proposal, Oct 31 2026. Nothing here is implemented yet.

## Why

Spiral (branch `spiral-object-store`, commits `3bb5801e6` and `1fd805f5e`) now
reads tracks and their crossings from the store. It does this through the
generic read view: `revision_read_view` (or `hub_read_view_at` at a hub
position) returns every selected object and every relationship among them as
nested Python dicts, and `object_store_inputs.read_snapshot` keeps them for
the whole fit session.

That doesn't scale to tracks. Measured on a 43,902-track, 99,421-crossing box
of `2um_ds2_ps256_surf_v2` (`read_snapshot` with tracks only):

| | Measured | Per item | Whole scroll (19.7M tracks, ~50M crossings) |
|---|---|---|---|
| Process RSS after `read_snapshot` | 2.27 GB | | |
| Python heap: crossing relationships | 1.15 GB | ~12 KB / crossing | ~610 GB |
| Python heap: track object records | 94 MB | ~2.2 KB / track | ~44 GB |
| Crossing table the fit actually uses | 6 MB | 63 B / crossing | ~3 GB |

The whole-scroll column is a straight-line extrapolation and is only
order-of-magnitude. Crossing density varies across the scroll.

Crossing edges are heavy because each anchor lists the crossing point
followed by every point inside the 12-voxel tangent radius
(`src/linking/TrackTrack.cpp:135-160`). That's ~25 point references per edge,
about 1.8 KB of JSON, and about 12 KB once decoded into Python. Those points
are the dependencies the store uses for staleness, so they stay on the write
side. This spec only changes what is *read*.

## Goal

A read that hands Spiral just what it consumes from tracks and crossings, as
NumPy arrays packed in C++. It must keep the store's semantics: the same read
snapshot, selection, applicability and approval filtering, and the same
answers at hub positions.

Targets:

- Memory: about 100 B per crossing and about 100 B per track plus 12 B per
  point, both in Python and at the C++ peak.
- No Python object per track or per crossing.
- Throughput is the colleague's call. The current per-track Python path costs
  about 65 µs per track for the record view plus about 25 µs per geometry
  read, against about 1.5 µs per track for a raw SQL chunk scan.

Non-goals:

- Changing what the track–track detector writes, anchor dependency lists, or
  staleness tracking.
- Ingest and link speed (`ingest` ~7 ms/track, `link_tracks_to_tracks` commit
  ~0.7 ms/edge). These are related, but separate work.
- Fibers, patches and point collections. They keep the dict view.

## What Spiral consumes today

From `spiral-fitting/object_store_inputs.py` (`read_snapshot`,
`filter_relationships`, `materialize_tracks`, `_track_crossing_csr`,
`StoredObject.provenance`, `StoredRelationship.provenance`).

Per selected track:

- `uuid` and `object_revision` (replica-local, used only to resolve);
- `geometry_revision`, used for identity and to read `points`;
- frame: `scale_factor` and `to_fit` in the fit volume (from
  `resolve_geometry_coordinates` / `geometry_scale_to_frame`). An object with
  no valid frame in the fit volume is an error;
- `meta.coordinates`, `track.family`, `track.dataset_ref` and
  `track.external_key`. These four make up the adapter metadata identity:
  family gates crossings, and dataset plus numeric key order the rows;
- `points`: int32 XYZ, shape `(n, 3)`, in stored voxels;
- tracks with no geometry are skipped with a note under the default selector,
  and are an error under an explicit one.

Per track–track `crossing` edge with both endpoints selected:

- edge `uuid`, `revision` (replica-local), `generation`, `approval_state`,
  and `applicable` (the store's own check, `endpointApplies`);
- each endpoint's `a_geometry_revision` / `b_geometry_revision`;
- `a_anchor.points[0].index[0]` and `b_anchor.points[0].index[0]`: the
  crossing point on each track;
- `meta.track_crossing.endpoints[<uuid>].arclength_voxels` for both tracks;
- `meta.track_crossing.clearance_voxels`.

Spiral does not read the anchor position or direction, the dependency points
after `[0]`, `dataset_ref`, the detector parameters or `source_id`.

Filtering, applied in this order with these exclusion-count labels:

1. `endpoint outside selection`
2. `inapplicable`
3. `unused kind <kind>`: edge kinds Spiral doesn't consume. Track–surface
   `same-surface` edges land here, deliberately.
4. `rejected`
5. `<state> (approved-only)`, when `approved_links_only`.

## Proposed interface

One new option on both read views, so tracks and crossings come from the same
read transaction as the rest of the inputs:

```python
view = store.revision_read_view(selection, packed_tracks=request)
view = store.hub_read_view_at(position, object_uuids, packed_tracks=request)
```

`selection` / `object_uuids` name the non-track objects as today.
`request` is a dict:

| Key | Meaning |
|---|---|
| `uuids` | `ndarray (N, 16) uint8` of track UUIDs, or `None` for selection by tags / all active |
| `tags` | list of tags (any-of, as `resolve_selector` does), or `None` |
| `volume_uuid` | the fit volume; frames resolve into it |
| `approved_only` | bool, Spiral's `approved_links_only` |
| `require_geometry` | bool. True means a selected track without geometry is an error; False means it is skipped and counted. Matches explicit vs default selectors. |

Selecting inside the call avoids passing 20M UUIDs through Python. Selection
semantics must match `object_store_inputs.resolve_selector`: default is every
active, non-retired track; at a hub position, those present at that position.
If replicating that in C++ is unattractive, accepting only a `uuids` array is
fine, since 16 bytes per track is cheap.

`view['packed_tracks']` is a dict of arrays.

### Tracks (N rows)

Rows are sorted by `(dataset_ref, numeric external_key, external_key, uuid)`,
the order `object_store_inputs._track_order` uses. Non-numeric keys sort after
numeric ones within a dataset.

| Key | dtype / shape | Notes |
|---|---|---|
| `uuid` | `uint8 (N,16)` | |
| `object_revision` | `uint8 (N,16)` | replica-local |
| `geometry_revision` | `uint8 (N,16)` | |
| `family` | `int8 (N,)` | 0 horizontal, 1 vertical, -1 anything else (`TRACK_FAMILY_CODES`) |
| `datasets` | `list[str]` | distinct `dataset_ref`s |
| `dataset` | `int32 (N,)` | index into `datasets` |
| `external_key` | `bytes` + `int64 (N+1,)` offsets | concatenated UTF-8 keys |
| `coordinates` | `list[str]` | distinct `meta.coordinates` JSON values (canonical form) |
| `coordinates_index` | `int32 (N,)` | index into `coordinates` |
| `scale_factor`, `to_fit` | `float64 (N,)` | frame in `volume_uuid` |
| `points` | `int32 (P,3)` | stored XYZ, concatenated |
| `offsets` | `int64 (N+1,)` | into `points` |

The `datasets` / `coordinates` tables keep the per-track metadata identity
exact without a string per track. Tracks almost always share one dataset and
one frame.

### Crossings (M rows, one per edge, not directed)

These are only the `crossing` edges between two selected tracks that pass the
filtering above.

| Key | dtype / shape | Notes |
|---|---|---|
| `uuid`, `revision` | `uint8 (M,16)` | revision is replica-local |
| `generation` | `int64 (M,)` | |
| `approval_state` | `int8 (M,)` | 0 pending, 1 approved (rejected never returned) |
| `a`, `b` | `int32 (M,)` | rows into the track arrays, in stored endpoint order |
| `a_geometry_revision`, `b_geometry_revision` | `uint8 (M,16)` | pins the edge was computed on |
| `a_local`, `b_local` | `int32 (M,)` | `points[0].index[0]` of each anchor |
| `a_arclength`, `b_arclength` | `float64 (M,)` | stored (dataset) voxels |
| `clearance` | `float64 (M,)` | stored voxels |

Rows are in any deterministic order. Spiral re-sorts them into its directed
CSR.

### Counts and problems

- `exclusions`: `{label: count}` with the labels listed above, for every edge
  that touches a selected track and is not returned. That includes
  track–surface edges, counted under `unused kind same-surface`.
- `skipped_without_geometry`: count, when `require_geometry` is false.
- `problems`: list of strings, capped at, say, 100 plus a total count. It
  covers frame errors, `points` that aren't `(n, 3)` int32, and a crossing
  with no point anchor, arclength or clearance, or an anchor index outside
  its track. Spiral turns any problem into `ObjectStoreInputError`, as it
  does today.

### The generic view

- Relationships returned as dicts in `view['relationships']` must not include
  any edge with a track endpoint. Otherwise millions of track–surface edges
  would still come back as dicts.
- Tracks are not in `view['objects']`. `view['selection']` may omit them.

## Implementation notes (store side)

- **Stream, don't collect.** `ObjectStore::relationshipsAt` builds a
  `std::vector<SelectedRelationship>` with full `EdgeRecord`s and anchors, plus
  a `std::set<std::string>` of every member version. At 50M edges that is
  itself tens of GB. The packed path should visit each member edge once,
  evaluate `endpointApplies` with its full anchor, append the few values
  above, and drop the record.
- Since `meta` is JSON, extracting `track_crossing.clearance_voxels` and the
  two `endpoints.<uuid>.arclength_voxels` with `json_extract` in SQL, or one
  parse per edge, is enough. There's no need to build `nlohmann::json` for the
  whole record.
- Track points: one contiguous output buffer. Read geometry arrays in chunks
  and release the GIL. `TrackDetail.pointCount` can size the buffer up front.
- Release the GIL for the whole packed read and build NumPy arrays from owned
  buffers (nanobind `ndarray` with a capsule owner) to avoid copies.

## Spiral-side changes (follow-up, not store work)

- **`read_snapshot`:** request `packed_tracks`, build no `StoredObject` for
  tracks, and keep the arrays on the snapshot.
- **`materialize_tracks`:** use the arrays directly. Arclength and tortuosity
  are already vectorized. Apply the z-window filter to `to_fit * z` per track.
  Build the directed CSR from the crossing arrays, with clearance scaled by the
  `a` track's `to_fit` as now. This replaces the per-edge Python loop in
  `_track_crossing_csr`.
- **Manifest:**
  - Today every track and crossing is a per-entry provenance dict in
    `manifest['objects']` / `manifest['relationships']`. At scale that alone
    is tens of GB of JSON.
  - Replace those entries with one section:
    `manifest['tracks'] = {'count', 'crossings', 'identity_sha256'}`.
  - The digest is computed in Python over canonical bytes of the shared
    identities and the consumed values:
    - for tracks: uuid, geometry revision, family, dataset, external key and
      coordinates;
    - for crossings: uuid, generation, approval state, endpoint UUIDs and
      geometry revisions, local indices, arclengths and clearance.
  - Replica-local revisions stay out, as `REPLICA_LOCAL_FIELDS` does now.
  - `track_identity()` (used to refuse in-place input updates) and
    `resume_problems` compare that section.
- **Link refresh:** `link_tracks_to_tracks(track_uuids=...)` also takes a
  20M-entry list. It could take the same selector, but that belongs to the
  link-speed work.

## Open question: how VC3D compares fitted tracks

VC3D (`apps/VC3D/SpiralStoreInputs.cpp`, `FitTrack`) currently reads a pin
per track from the manifest and compares each with the store's head, the
same way it compares fibers. With a digest-only manifest it can't do that. It
also shouldn't re-hash Spiral's digest in C++: compare shared identities and
values, never re-hash across languages. Two options:

1. **The Spiral service answers it.** It has the code and the packed read, so
   it re-reads and compares on request, and VC3D shows one "tracks: fitted /
   changed" row.
2. **The store answers "has anything changed since".** Given the fit's read
   position (journal sequence) and the track selection, the store reports
   whether any selected track's head, or any crossing among them, changed
   after it. That's a pure identity comparison and is cheap in SQL.

Option 2 also suits a 20M-track fit in VC3D, where per-track rows aren't
useful anyway.

## Verification

- **Equality:** on the fixture lattice (`spiral-fitting/tests/test_object_store_tracks.py`)
  and on the 44k-track box, the packed path's `PackedTrackCollection` and
  directed crossing CSR must be bit-identical to the current dict path. Use
  the A/B harness
  `/tmp/claude-1000/-mnt-bcache-projects-vesuvius-scrolls-villa/03ee524e-badc-492a-8dec-576a8529a9bf/scratchpad/ab/ab_tracks.py`
  (store `.../scratchpad/ab/tracks.vcobj`). It compares prepared track
  bundles and seeded samples. The scratchpad is temporary, so copy it if you
  need it later.
- **Filtering:** same exclusion counts as `filter_relationships` for rejected,
  pending under approved-only, inapplicable after a track geometry edit, and
  track–surface edges.
- **Hub positions:** same result through `hub_read_view_at` at a position as
  through `revision_read_view` on the matching selection.
- **Memory:** RSS before and after `read_snapshot` on the 44k box, which is
  2.27 GB today. The target is under ~50 MB above baseline, plus the points.
