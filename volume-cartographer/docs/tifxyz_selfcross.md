# tifxyz Self-Intersection Census

`vc_tifxyz_selfcross` reports where a tifxyz surface passes through itself.
It is report-only: it writes a JSON report and, optionally, a point
collection of crossing sites, and never modifies the surface.

## Why transverse contacts specifically

Proximity between wraps is normal in a crushed scroll and can be arbitrarily
small while the trace is correct, so no distance threshold separates good
traces from bad ones. A transverse self-intersection is different in kind: an
embedded surface cannot pass through itself, so a triangle of the trace
crossing another triangle of the same trace is a defect with no innocent
reading, and no threshold to argue about.

Contacts are classified rather than collapsed into yes/no:

- `transverse` — the two triangle interiors pass through each other. This is
  the defect signal.
- `coplanar` — (near) coplanar triangles with overlapping projections. Two
  sheets pressed flat against each other look like this, so it is reported
  but never counted as a crossing.
- `grazing` — a vertex or edge lies within tolerance of the other triangle's
  plane, so the sign data the interval test depends on is not trustworthy at
  the input's own float32 resolution. Reported separately rather than
  guessed at.

Adjacent quads (Chebyshev distance `--exclude` in grid indices, default 1 =
shared-vertex neighbours) are excluded: their triangles sharing space is what
a surface does. Every quad is triangulated both ways and the two censuses are
reported separately, because a twisted quad can cross under one diagonal and
not the other; a surface is only called clean when both find nothing.

## Usage

```sh
vc_tifxyz_selfcross <surface.tifxyz> -o report.json
vc_tifxyz_selfcross <surface.tifxyz> -o report.json --collection sites.json
vc_tifxyz_selfcross <surface.tifxyz> -o report.json --fail-on-crossing
```

The report carries per-diagonal counts and every transverse contact with its
quad indices, penetration depth (voxels), crossing angle, and a 3D site. The
`--collection` file is written through `PointCollections::saveToJSON`, so it
loads in VC3D like any other point collection and the sites can be inspected
in place.

Exit codes: `0` census ran (whatever it found), `1` error, `3` transverse
contacts found and `--fail-on-crossing` was given — usable as a gate in
scripts, e.g. after an export or before a merge.

## What the numbers mean

Counts are triangle-pair contacts summed per triangulation, not distinct
crossing events: one crossing region typically produces several contacts.
They rank severity; they do not count places.

The census runs on the surface as this codebase loads it (`z <= 0` cells
invalid, `mask.tif` applied). The same method applied to a raw tifxyz that
treats `z <= 0` cells as valid can differ slightly in contact counts
(measured 1.49% of rows over one 185-trace corpus, with no clean/not-clean
verdict changing).

A clean result means exactly: zero non-adjacent transverse contacts under
both triangulations, at the stated exclusion and edge-length filter. It is
not a statement about coplanar or grazing contacts, and not a general
statement of surface quality.

`--maxedge` (default 60 voxels) drops quads with any edge longer than the
limit before testing. This is not cosmetic: a grid can contain
discontinuities where two adjacent valid cells sit far apart in 3D, and a
triangle built across such a gap spans many wraps and crosses everything in
its path purely because the mesh has a hole there.

## Determinism

Two runs on the same surface produce identical reports regardless of thread
count. A triangle pair can share several broad-phase cells; each pair is
owned by exactly one cell (the one containing the minimum corner of the two
bounding boxes' overlap) and tested only there, and results are sorted into
canonical order before writing.
