# tifxyz Grid-Topology Census

`vc_tifxyz_topology` reports the ways a tifxyz surface stops being one
connected sheet: islands, holes, tears and folds. It is report-only — it
writes a JSON report and, optionally, a point collection of defect sites, and
never modifies the surface. Several surfaces can be given at once, which is
the point: the census is one linear pass over each grid, so a whole scroll's
traces can be ranked and only the flagged ones need a person.

## Why these four

A tifxyz surface is a 2D grid of 3D points that is supposed to describe one
connected sheet of papyrus. Nothing in the format enforces that, nothing that
writes one checks it, and the tools that read one disagree about what to do
when it is not true:

| class | what it is | what happens today |
|---|---|---|
| `islands` | valid quads outside the largest edge-connected component | `vc_flatten` labels edge-connected triangles, keeps the largest component and drops the rest, prints one line to stdout and exits 0 (`core/src/ABFFlattening.cpp`) |
| `holes` | a run of invalid cells enclosed by valid ones, as opposed to the invalid padding the sheet is written into | `vc_tifxyz2obj --inpaint` fills them with invented geometry, warning that "holes can break flattening"; nothing says how many there are or where before you decide |
| `tears` | grid-adjacent vertices far apart in 3D | `vc_tifxyz_selfcross` drops the quads that span them (`--maxedge`) and reports only how many; `vc_tifxyz2zarr_sparse` has no such guard and rasterizes them at full length |
| `folds` | quads whose two triangles face away from each other under the triangulation villa itself builds | nothing looks; `vc_tifxyz2obj`, `vc_tifxyz2zarr_sparse` and `ABFFlattening` all emit the crossed pair |

So the same surface can be clean to one tool and broken to the next, and
nothing says so. The census says so once, in the surface's own terms.

Self-intersection is deliberately not here. `vc_tifxyz_selfcross` censuses it,
under the same validity rule, and the two reports are complementary: a
triangle-pair test cannot see an island, a hole or a fold, and this tool does
not test triangle pairs.

## Usage

```sh
vc_tifxyz_topology <surface.tifxyz> -o report.json
vc_tifxyz_topology <a.tifxyz> <b.tifxyz> ... -o report.json --collection sites.json
vc_tifxyz_topology <surface.tifxyz> -o report.json --fail-on islands,tears
```

The report carries a `summary` (how many surfaces were censused, how many
carry each class, and a `flagged` worklist naming the ones that do) and a
`surfaces` array in input order, each entry holding the complete counts plus
up to `--max-sites` located sites per class. Every site has a grid index and
a 3D position, so `--collection` can be opened in VC3D and each defect
inspected in place. Sites from several surfaces are merged into one
collection, so that flag is only meaningful for surfaces traced on the same
volume.

Exit codes: `0` census ran (whatever it found), `1` an error or a surface
that could not be read, `3` `--fail-on` named a class that was found. One
unreadable surface does not discard the other censuses: it is recorded in its
entry, the run continues, and the exit code still says something failed.

## What the numbers mean

Counts are over the surface as this codebase loads it: `z <= 0` cells are
invalid and `mask.tif` applies, the same rule `vc_tifxyz_selfcross` censuses
under.

`islands` counts quads, not vertices, because quads are the unit `vc_flatten`
works in. The largest component is never counted as an island — it is the
surface. `isolated_vertices` is reported separately: a valid vertex that is a
corner of no valid quad carries no surface, but it is still a point of the
surface to everything that takes a bounding box, which is how one stray cell
inflates a render canvas.

`holes` are enclosed runs only. The invalid cells around a sheet, and a bay
cut in from its boundary, reach the grid border and are the sheet's own
shape, not damage to it — the same border test `core/src/InpaintSurface.cpp`
applies before it fills, and the same 8-connectivity, so the two agree about
which runs are holes. (8 for the invalid cells against 4 for the valid quads
is also the pairing that keeps the two answers consistent: a diagonal chain
of invalid cells does separate the sheet, so it must not also count as two
separate holes.)

`tears` are measured against the surface's own median grid step, per
direction, so no absolute voxel threshold has to be guessed for a scroll or a
resolution — a 45 µm trace and a 1.1 µm trace are judged by the same rule.
`--tear-factor` (default 4) is that multiple. Adjacent torn edges are joined
into one site, including diagonally, so a seam that steps sideways is one
place to look at rather than hundreds of numbers.

`folds` are quads whose two triangles face more than 120° apart across the
**p01–p10** diagonal — the one `vc_tifxyz2obj` (`(p10,p00,p01)` +
`(p10,p01,p11)`), `vc_tifxyz2zarr_sparse` and `ABFFlattening` all build, with
the normals formed exactly as `vc_tifxyz2obj` forms them. A quad reported
here is one those tools will carry as a crossed pair. Papyrus does not turn
through a right angle between one grid cell and the next — at a 20-voxel step
on a 7.91 µm scan that is 158 µm — so a quad that does is the trace being
locally incoherent.

Folds across the other diagonal are counted as `quads_on_unused_diagonal` and
not located: nothing in this repository triangulates that way, so they change
no consumer's geometry, but `vc_tifxyz_selfcross` censuses both diagonals and
a crossing it reports under the unused one starts there.

The angle threshold is not decoration:
coordinates are float32 running to ~1e4 voxels, where one ULP is already
~1e-3, and real grids hold quads as thin as 0.7 × 20 voxels. On one of those
the two cross products carry more error than signal, and their dot product
comes out at ~1e-5 of the product of their magnitudes with an essentially
random sign — a plain sign test reports 67% of the published corpus as
folded, almost all of it slivers. A real fold is not marginal, so requiring a
large disagreement costs nothing and removes the noise. Zero-area quads are
counted as `degenerate_quads` and asked no fold question at all.

Adjacent folded quads are clustered the same way tears are, and the report
gives both `quads` and `sites` with the site list ordered by size, so an
isolated quad and a region that turns back on itself are distinguishable.

A clean result means exactly: one component, no enclosed invalid runs, no
grid edge above the stated multiple of the median step, and no folded quad.
It is not a statement about self-intersection, about whether the surface
follows the papyrus, or about anything the geometry alone cannot show.

## What this does not do

It does not repair anything, and that is deliberate.

Three of the four classes have no safe automatic fix. Filling a hole invents
papyrus that was not traced, which is why `vc_tifxyz2obj --inpaint` is opt-in
and says so. Closing a tear either moves real vertices or
deletes them, and which one is right depends on whether the trace jumped or
the sheet is genuinely torn there — evidence this tool does not have.
Unfolding a quad is a local edit to a parametrization whose global effect on
the flattened image is not local.

The fourth, dropping islands, is the one subtractive edit that invents
nothing, and `vc_flatten` already performs it silently. Making it explicit is
a small change to a consumer, not a new mode here; what it needs first is
evidence of how much area is being dropped in practice, which is what this
census produces.

## Fixtures

`core/test/test_tifxyz_topology.cpp` covers the kernel through the same header the tool
includes, and `core/test/test_tifxyz_topology_cli.cpp` drives the binary. Two of the
fixtures are quads copied verbatim from published traces, because both of them were
mistakes this census made before the corpus corrected it:

| fixture | where it comes from | what it pins |
|---|---|---|
| a 0.7 × 20 voxel sliver | `PHercParis4/…/20260603222816-on-20230205180739-7.91um.tifxyz`, cell (96, 764) | its two triangle normals are 90.0005° apart — float32 noise, not a fold. A plain sign test called 67% of the corpus folded, almost all of it this |
| a 148°/80° quad | `PHerc0172/…/20251106170358-on-20241024131838-7.91um.tifxyz`, cell (286, 566) | it folds across p00–p11 but not across the diagonal villa builds, so it is counted and not located |

The CLI test is registered behind `if(TARGET vc_tifxyz_topology)`: the test shards
configure with `VC_BUILD_APPS=OFF`, and the `TARGET_FILE` generator expression would be a
configure error there.

## Determinism

Two runs on the same surfaces produce byte-identical reports. The census is
single-threaded and hash-free, and every emitted list is sorted by size with
an explicit tie-break on the grid index, so "the largest component" never
depends on iteration order. Timing goes to stderr, never into the report.
