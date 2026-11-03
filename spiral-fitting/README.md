# Spiral fitting

Code and helpers to fit a canonical Archimedean spiral to deformed scrolls.
`spiral_service.py` hosts one persistent interactive fit session over HTTP for
the VC3D Spiral workspace; `fit_spiral.py` is the underlying fitter.

## Running the object-store workflow

Use VC3D and the Spiral service from the same checkout: the explicit-inputs
protocol is API **45**. Update the existing Spiral environment with `uv sync`
from this directory when deploying this checkout (native prerequisites must
already be installed). This builds the Python bindings, not the standalone
`vc_object_ingest` and `vc_object_hub` tools; use those from a matching native
object-store build.

| Input | Command-line fit | Service fit |
| --- | --- | --- |
| Patches, PCLs, fibers, tracks and their relationships | `--object-store` | Service's bound store |
| Outer shell | Store surface tagged `spiral-outer-shell` | Same |
| Scroll facts and umbilicus | `--inputs` manifest's `scroll` and `umbilicus` | VC3D project |
| Lasagna fields | Manifest's `lasagna.manifest` and the cache; missing packs may be built | Project-selected dataset; packs must already be cached |
| Fiber directions | Manifest's `fiber_directions` | Project-selected fiber-inference dataset's cached artifact |
| Winding inference | Manifest's `winding_inference` | Service's `--winding-inference` binding |

1. Ingest geometry once as described under **One-time dataset ingest**. For
   tracks, first run `python convert_track_store.py /data/scroll/tracks/input.dbm`,
   then ingest that DBM with its packed mirror using `vc_object_ingest --tracks`.
   Existing stores need only the track ingest, not another import of their
   patches/PCLs/fibers. Frames declare stored coordinates per reference-volume
   voxel; use the real scale for each source. Fits do not read DBMs,
   `.vctracks` directories or `.crossings.npz` sidecars after ingest.
2. For CLI use, write an input manifest naming the fit volume, its facts
   (preferably with `base_shape_zyx`) and every input file; see **Input
   manifest**. Lasagna scale is derived where possible; otherwise an explicit
   `scroll.lasagna_scale` is required. See **Input manifest** for the
   `source_to_base == 1` restriction and cross-checks.
3. For service use, select the project volume, import its umbilicus, and
   select the Lasagna dataset in VC3D. Pack its fields into the service's
   cache first; see **Lasagna inputs must be packed first**. A local copy of
   a remote dataset must be packed with `--as` naming the same URL VC3D uses.
   Prepare the fiber-direction cache too if enabling that loss, and bind
   the winding-inference artifact with `--winding-inference` if enabling
   winding-model supervision.
4. Run either command below from `spiral-fitting/`. The example enables
   tracks explicitly: `input_use_tracks` defaults to false. Choose a real
   fit window and input paths for your scroll.

```sh
FIT_SPIRAL_OUT_DIR=/data/spiral-output \
FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin":9000,"z_end":9100,"input_use_tracks":true}' \
  .venv/bin/python fit_spiral.py \
  --inputs /data/spiral-inputs.json --object-store /data/scroll.vcobj \
  --object-store-author "$USER" --cache /data/spiral-cache

.venv/bin/python spiral_service.py \
  --port 8765 --object-store /data/scroll.vcobj \
  --winding-inference /data/winding_inference \
  --output /data/spiral-output --cache /data/spiral-cache \
  --gpus 0 --session-name scroll-fit
```

For the service, enable tracks and choose the z window in VC3D before
Initialize Fit. Localhost mode can launch the service itself. A remote
service needs a replica of the same hub as VC3D; see **Object store,
positions and input updates** for launch and SSH migration requirements.

Startup normally writes refreshed relationships, including track crossings.
`object_store_refresh_links: false` consumes existing relationships instead;
there is no fallback crossing scan in the fitter. Track selection uses
`object_store_track_uuids` / `object_store_track_tags`, like the other kinds.
Remove the retired `track_scale_factor` config setting; each stored track
now declares its frame. Track databases are never input paths.

Save drafts into the store, then **Update Fit Inputs** to adopt eligible
geometry changes without resetting optimization. Changes to tracks, their
crossings, the outer shell, the volume or prepared-input settings require
**Rebuild with Current Inputs**. With tracks enabled and
`track_exclusion_radius > 0` (default 16), input replacement also requires a
rebuild because the exclusion masks depend on the trusted geometry. Saved
checkpoints remain subject to input-manifest validation; a pre-migration
track fit is not automatically equivalent to a new store-backed fit.

## CUDA startup check

VC3D fit sessions and command-line fits allocate one CUDA element and synchronize
before loading fit inputs. This creates the CUDA context before dataset reads
increase filesystem cache pressure, and reports startup failures independently
of atlas size. CPU-only `FitContext.load_host_inputs()` callers remain unchanged.
The GPU atlas progress messages appear when geometry is actually materialized.

On Linux NVIDIA GB10 systems, startup OOM diagnostics include host memory figures
and NVIDIA's manual cache-flush workaround. The application never flushes system
caches itself. Early initialization does not guarantee that the full fit will
fit in shared CPU/GPU memory.

Validate with the existing environment, from the repository root:

```sh
AGENTS_AGENT_MODE=1 PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_cuda_startup.py \
  spiral-fitting/tests/test_spiral_headless.py \
  spiral-fitting/tests/test_spiral_progress.py
AGENTS_AGENT_MODE=1 PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python -c \
  'from types import SimpleNamespace; from fit_spiral import FitContext; FitContext.check_cuda_ready(SimpleNamespace(progress=None))'
```

See [NVIDIA's DGX Spark memory guidance](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).

## Running without CUDA

Fits run on CUDA whenever it is available. Without it they run on Apple's MPS
backend, and without that on the CPU; `FIT_SPIRAL_DEVICE=cuda|mps|cpu` (or `cuda:1`)
picks one explicitly (`devices.py`). A distributed (torchrun) fit needs CUDA or
the CPU: torch.distributed has no MPS backend. The Triton kernels are CUDA only, so MPS and the
CPU use the PyTorch paths, and the CUDA startup check is skipped.

MPS has no float64. Device tensors that are float64 on CUDA take their dtype
from `devices.float_hi()`, which is float32 on MPS;
`FIT_SPIRAL_MAX_PRECISION_FLOAT=32` applies the same cap on every device, for
comparing a CPU run with an MPS one. `get_ink_metrics.py --device` picks the
nnU-Net device; without it the folds take CUDA, else MPS, else the CPU, whatever
the fitting variables say.

## Input manifest

`fit_spiral.py --inputs MANIFEST` names every input a command-line fit reads
by path; nothing is discovered in a dataset directory and no filename is
inferred. Relative paths resolve against the manifest's own directory, so a
fit runs the same from any working directory; files may have any names and
live anywhere. Geometry (patches, PCLs, fibers, tracks and the outer shell)
is read from `--object-store` only and has no manifest entry. A service
session takes the scroll's facts and inputs from the VC3D project and its own
launch bindings instead; see
[The VC3D project's inputs](#the-vc3d-projects-inputs). This is not covered
by scrollprize.org/tutorial_spiral.

`runners/run_single.py --resume` records the object store's persistent UUID
in its run state and checks it even when the fit stage is already complete.
Moving the store or using a replica of the same store preserves this identity;
switching to a different store is refused. Older runner state files without
the UUID cannot establish which store was used and are refused too; use a new
output directory. This check identifies the store, not its current contents;
the fitter's checkpoint input-manifest checks still apply when fitting resumes.

```json
{
  "schema_version": 1,
  "volume_uuid": "reference-volume-id",
  "scroll": {
    "name": "PHerc0826",
    "voxel_size_um": 9.362,
    "base_shape_zyx": [12000, 8000, 8000],
    "spiral_outward_sense": "CW"
  },
  "umbilicus": {"path": "umbilicus.json", "coordinate_scale": 1},
  "lasagna": {
    "manifest": "lasagna/predictions.lasagna.json",
    "original_url": "https://example.org/predictions.lasagna.json"
  },
  "fiber_directions": "fiber_directions.npz",
  "winding_inference": "winding_inference",
  "scroll_zarr": "scroll.ome.zarr/0"
}
```

Required:

- `schema_version` — must equal `1`.
- `volume_uuid` — the reference volume the inputs are in: the volume the
  selected store objects declare (`vc_object_ingest --volume-uuid`). A fit
  whose objects are in another volume is refused, and checkpoints record it.
- `scroll` — the scroll's physical facts (below).
- `umbilicus` — `path` to its control-point JSON, and optionally
  `coordinate_scale` (its coordinates per fit-volume voxel, default 1).

Optional, but required when the configuration's losses read them (checked,
all at once, before the fit starts):

- `lasagna.manifest` — the local `*.lasagna.json` manifest (see "Lasagna
  inputs must be packed first"). `lasagna.original_url` says the local
  manifest is a copy of the remote dataset at that URL: the fit reads it as
  that dataset, sharing its packs with a service fitting the URL. Nothing is
  downloaded.
- `fiber_directions` — the samples file (see "Fiber direction samples").
- `winding_inference` — the crossing artifact directory (see "Neural
  winding-inference losses").
- `scroll_zarr` — the CT volume, a path or a URL; used for visualization and
  the Lasagna shape check only.

Unknown keys are refused, with guidance for a former `spiral-scroll.json`'s
(`spiral-scroll.json` is not read: move its facts into `scroll` and name each
input; its `paths` overrides are retired) and for geometry keys.

### Scroll facts

- `name`, `voxel_size_um` — required.
- `z_direction_is_top_to_bottom`, `left_handed_coordinates` — the fitted
  volume's two properties from the open-data catalog (`metadata.json`), copied
  verbatim as `true`/`false`. Together they fix the spiral's sense under the
  catalog convention that every scroll shows the same spiral seen from its
  top: `"CW"` when they are equal, `"ACW"` when they differ. The z direction
  also orients exported surfaces (below).
- `spiral_outward_sense` — `"CW"` or `"ACW"` (case-insensitive). Derived from
  the two catalog properties when both are present, and then only needed as a
  cross-check (a mismatch is an error). Without them it is required and is
  read off the CT data by a person in VC3D, or from an already-fitted spiral.
  The fit setting `spiral_outward_sense` (`FIT_SPIRAL_CONFIG_OVERRIDES`) may
  give it instead, and must agree with the manifest where both do.

Exported surfaces (the per-winding `meshes/`, the combined preview and
`flatten_spiral_checkpoint.py`'s source surface) read like the scroll: column 0
is the outermost wrap, U running outside to inside as orient-segment
normalises it, and row 0 is the top of the scroll when
`z_direction_is_top_to_bottom` is known (rows stay in z order when it is
not). Each `meta.json` and preview manifest records the z direction the
layout was made with under `grid_orientation` (a grid without it is an older
one in sampling order: innermost first, rows by z); `winding_column_ranges`
are in the written grid, outermost winding first. `left_handed_coordinates` does not change the grid, only the
direction of its cross-product normal, which renders correct with
`flip-normals = not left_handed_coordinates`.

Optional `base_shape_zyx`: the fit volume's shape in its own voxels, a ZYX
list of three positive integers (the catalog `metadata.json` shape of a
full-resolution volume). Exports record it, and the fit derives the Lasagna
scale from it.

The Lasagna arrays' downsampling relative to the fit volume (`lasagna_scale`)
is derived for command-line fits when a fit reads the
Lasagna dataset (see "Lasagna inputs must be packed first") and both shapes
are known: `scroll.base_shape_zyx` and the Lasagna manifest's `base_shape_zyx`. The two shapes
must be power-of-two levels of one grid (ceil or floor halving; at one level
they may differ by one voxel, the voxel-count and inclusive conventions), and
the scale is then the manifest's base voxels per array voxel (2^`scaledown`)
over its base voxels per fit-volume voxel, which must be a whole number. The
manifest's `base_shape_zyx` is level 0 of the finest real scan, and
`scaledown` counts from it even when the Lasagna was computed from a
downsampled level: PHercParis4's, inferred on level 2 of its 2.4 µm volume,
states that volume's full shape and normals at `scaledown` 4 (16
full-resolution voxels). A manifest with `source_to_base` other than 1
(levels counted from a pre-downsampled volume) has no agreed convention
between VC3D and the Lasagna writer yet, so the CLI requires an explicit
scale for it. VC3D's service request still computes
`source_to_base * 2^scaledown` before converting to fit voxels; the CLI
restriction is not currently enforced there. **The derivation assumes the two grids cover
the same aligned volume extent** (one origin, one extent): shapes cannot show
a cropped or offset Lasagna dataset, or another volume of the same shape.

An explicit `scroll.lasagna_scale` (a positive integer) is then a cross-check, and a
fit whose value disagrees with the derived one is refused. Where the scale
cannot be derived (either shape missing, shapes that are not levels of one
grid, a fractional result) `lasagna_scale` is required: there is no default.
Read it from the arrays' own `.zattrs` multiscales metadata, not from another
scroll's example. Whatever its source, a fit that opens the scroll zarr
refuses Lasagna packs whose shape is not that volume's downsampling by the
scale.

Optional `num_windings`: the scroll's winding count, where someone has
estimated it (an integer of at least 2). It replaces the generic default of
both `shell_outer_winding_idx` and `model_gap_expander_num_windings`, which is
Scroll 1's count (130). The exporter writes windings up to
`shell_outer_winding_idx`, so on a scroll with fewer windings the generic
value exports windings that lie outside the papyrus. It is a default rather
than a fact the scroll owns: `FIT_SPIRAL_CONFIG_OVERRIDES` still overrides it,
and `model_gap_expander_capacity_windings` is raised to `num_windings + 3`
only when the count would not otherwise fit. A service reads no manifest: set
the winding count in the fit configuration (the Spiral panel, or the
service's `--config`).

Migrating a dataset directory: write a manifest beside it naming its
`umbilicus.json`, `lasagna_inputs/*.lasagna.json`, `fiber_directions.npz` and
`winding_inference/` explicitly, move `spiral-scroll.json`'s facts into
`scroll` (its `umbilicus.coordinate_scale` into `umbilicus`), and set
`volume_uuid` to the `--volume-uuid` its store objects were ingested with.

## Object-store inputs

`fit_spiral.py` reads patches, fibers, point collections (PCLs) and tracks,
and the relationships between them, from a `vc_object_store` database
(`object_store_inputs.py`). It never reads patch directories, fiber JSONs,
PCL documents or track databases, and never links points to patches or finds
track crossings itself: the store's detectors do. The other inputs (umbilicus,
lasagna normal/gradient stores, fiber directions, winding inference) are
named by the input manifest (`--inputs`; see "Input manifest").

### One-time dataset ingest

Ingest the dataset's files into a store once, with the store's
`vc_object_ingest` (built with `vc_object_store`; see its `docs/README.md`).
Every object needs a coordinate frame (`--volume-uuid`, `--scale-factor`: stored
coordinates = reference-volume voxels × the scale factor), and VC3D-era files
need identities derived from one UUID namespace per dataset
(`--derive-uuids`; generate it once, record it with the dataset, and reuse
it, so the same files always map to the same objects). For the conventional
Spiral layout, where patches and PCLs are in fit-volume voxels and version-3
fibers without `coordinate_base_shape_zyx` are at 4× that scale, that is two
runs, one per frame:

```sh
NS=<dataset uuid namespace>   # e.g. python3 -c 'import uuid; print(uuid.uuid4())', once
ingest="vc_object_ingest --db dataset.vcobj --author $USER --derive-uuids $NS --volume-uuid VOLUME"
patches=(); for p in dataset/verified_patches/*/; do patches+=(--surface "$p"); done
$ingest --scale-factor 1 --renumber-point-ids "${patches[@]}" \
    --point-collections dataset/abs_winding.json \
    --point-collections dataset/relative_windings.json \
    --point-collections dataset/same_windings.json
$ingest --scale-factor 4 --drop-dangling-branches --fiber dataset/fibers
$ingest --scale-factor 1 --tag spiral-outer-shell --surface dataset/outer_shell
python convert_track_store.py dataset/tracks/2um_ds2_ps256_surf_v2.dbm
$ingest --scale-factor S --tracks dataset/tracks/2um_ds2_ps256_surf_v2.dbm
```

`VOLUME` names the reference volume (its catalog `metadata.json` id, or any
stable name such as `s1-reference`); the fit infers it from the objects, and
the input manifest's `volume_uuid` must name the same one. Fibers
declaring `coordinate_base_shape_zyx` need the scale that domain implies
instead of 4. Tracks are read from the DBM's packed `.vctracks` mirror (see
"Packing large track databases"); `S` is their stored voxels per fit-volume
voxel (1 when they were extracted from the fit volume itself). The two repair
flags handle the 2026-10 Spiral dataset's defects
(branches to fiber files that do not exist; `same_windings.json` collections
that each number their points from 0) and are recorded on the objects.

What the ingest covers: tifxyz patches (`--surface`, one directory each,
stored as kind `surface`), the outer shell (a tifxyz surface tagged
`spiral-outer-shell`), the PCL role documents, and the fiber directory with
its same-winding branches
(which become `fiber_branch` relationships; adjacent branches are kept but
never used by the fit), and tracks (`--tracks`, one object per track, keyed by
its packed store and source ID). A collection is absolute-winding when its
`meta.source.winding_is_absolute` is true, which the ingest sets for a
document named `abs_winding.json`; otherwise its points' `wind_a` annotations
make it relative, and none make it same-winding. `patch-overlap-pcls.json` is
not ingested: the store's overlap detector replaces it. Ingest is not repeated
to update a store; edit objects through the store's API instead.

### Running a fit

The command-line fit reads its inputs from the store named by `--object-store`
(required; the config keys `object_store_path` and `object_store_author` are
set from the command line, and `FIT_SPIRAL_CONFIG_OVERRIDES` cannot name them):

```sh
FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 9000, "z_end": 9100}' \
  python fit_spiral.py --inputs spiral-inputs.json --object-store dataset.vcobj \
  --object-store-author me
```

`--object-store-author` is optional: it defaults to `$USER`, then `$USERNAME`,
then `spiral-fit` if neither is set to a nonempty value. Pass it explicitly to
override that identity. The service keeps its `spiral-service@<host>` default.

At startup rank 0 resolves the selection (`object_store_{patch,pcl,fiber,track}_{uuids,tags}`;
unset selects every active object, an empty list none; `input_use_*` toggles
off select none), validates every selected object's frame, then refreshes the
PCL-patch, fiber-patch, patch-overlap and track-crossing links as
`--object-store-author`,
writing them back to the store. Detectors whose object kinds are not all
selected are skipped. `input_use_patch_overlaps: false` skips the
patch-overlap detector and adds no overlap pairs to the fit (a rebuild
setting). Every detector failure is reported together and stops
the fit before training (detectors that already committed are named; a rerun
reconciles them). `object_store_refresh_links: false` reads the store's
current relationships read-only.

After a successful refresh the store records a link marker (not journaled;
it adds no position and no hub sync carries it): the position, the fit
volume, the link parameters, normal source and detector versions, and the
selection. The next refresh skips the detectors when the marker's inputs
match, the selection is the same or smaller, and nothing since its position
touched a selected object, a detector relationship (point-surface,
same-surface, crossing) or a volume registration. A service input update
links only objects added since (pairs with a new object, partners found by
stored bounds); those links are approximate (an existing PCL keeps its old
surface link alongside one to a new surface), and the next startup refreshes
in full. `object_store_full_link_refresh: true` always runs every detector.
Each detector's mode and duration are logged, and its progress shows in the
service's `input_preparation.progress` (stderr for a CLI fit).

Pending and approved relationships are used
and rejected ones never; `object_store_approved_links_only` uses approved ones
only (refresh still runs, so new links stay pending and unused).

The outer shell is the one active surface tagged `spiral-outer-shell` (two are
an error). It is never a patch: the default and tag selections leave it out,
and naming it in `object_store_patch_uuids` is refused, so no detector links
it. It is read with the selection, in the same fit volume, and its revision
is part of the input manifest and digest; a fit with shell losses or
winding-model supervision refuses a store without one, and with
`input_use_outer_shell` off it is not read. An input update that changes the
shell is refused (the fit consumed it at build): rebuild instead.

Fiber-patch refresh uses the fit's lasagna normals when `input_use_normals` is
on and the normal stores exist (whatever the dense-normal loss weight), else
`object_store_normal_ct_*` structure-tensor normals, else the fiber centerline
alone (with a warning). The log names the source.

After refresh every rank reads the same frozen snapshot and materializes it:
patch grids, PCL points by stable ID with their `wind_a`, fibers trimmed and
decimated with every patch contact and branch endpoint kept, branch links,
patch overlaps as same-surface point pairs (`pcl_patch_overlap_*`, sampling
group `patch_overlaps`), and the tracks lying entirely inside the z window
with the crossing CSR of their `crossing` relationships, from which the fit
builds its track crossing graph. Track-surface relationships are not used. Attachments on patch quads the losses do not sample
are dropped and reported. The ranks' input digests must agree. The log prints
the selection, consumed relationships and constraint counts; `python
object_store_inputs.py dataset.vcobj --z-begin Z0 --z-end Z1` prints the same
report without training.

On a hosted replica (a clone of a store hub; see the store's "Hosted hub and
replicas") the inputs are a **hub position**: startup pulls the hub head,
refreshes links over the selection, publishes them with a conditional push
that succeeds only while the hub is still at that head, and pins the
resulting position, which later hub commits never change. A commit landing
meanwhile, even an unrelated one, makes it pull and relink (bounded; a
conflict keeps the hub's side, since this replica writes only recomputable
links). `object_store_input_position` reads a position exactly, without
linking. The manifest records the position (outside the digest: equal inputs
at two positions are equal) and the link parameters and normal source, so a
later change of either shows the fit's links as stale.

The inputs are frozen for the fit: settings that would relink or re-derive
them (`config.OBJECT_STORE_PCL_KEYS`, the PCL role toggles, selection and
frame settings) need a new fit. Each run writes `object_store_inputs.json`
(store, selection, the revision and scale of every object and the revision of
every consumed relationship, the fit volume, external-input frames,
materialization settings, and the refresh mode with its fiber-patch normal
source) and checkpoints carry the same manifest and digest; a resume whose
inputs differ is refused unless `object_store_allow_changed_inputs`.

The digest identifies inputs the same way on every replica of a store (see
the store's "Hosted hub and replicas"): an object by its geometry revision
and a hash of the metadata the adapter reads (`metadata_sha256`: its frame,
erosion, PCL role and sampling group, fiber classification), a relationship
by its record, approval, generation, endpoint geometry revisions and a hash
of its anchors and metadata (`content_sha256`). Object revision and
relationship version UUIDs differ between replicas; the manifest keeps them
(`object_revision`, `revision`) to resolve the inputs locally, outside the
digest. A checkpoint written before these identities is compared on geometry
revisions and relationship generations, with a warning.

### Frames and units

The fit's unit is the reference-volume voxel (ZYX), whatever the stored
scale: each object's stored XYZ is divided by its own scale factor once, at
materialization, and patch grid indices are unchanged. Loss thresholds, ROI
bounds, spacings and radial offsets are all in reference-volume voxels. An
object without a declared frame, or frames naming different volumes, is an
error; nothing is assumed to be at scale 1. The umbilicus keeps its
input-manifest coordinate scale and lasagna stores their
`lasagna_scale`; the outer shell, fiber directions and winding inference are
in fit-volume voxels. Tracks declare their frames like every other object;
their crossings are found on the stored integer voxels and their arclength
positions and clearances converted to fit voxels.

`fit_volume_level` (default 0) fits at a pyramid level of the reference
volume instead: a fit voxel is 2^level reference voxels, so the z range and
every setting above are in that level's voxels. Object frames, the scroll's
facts (shape, voxel size, umbilicus and Lasagna scales), the store's overlap
pairs and the fiber directions a service extracts are converted to it; link
refresh still works in the reference volume's own voxels, and exported
surfaces and previews are written back in them. The winding-inference
artifact and a `scroll_zarr` array are read as they are, so they must already
be at that level. A checkpoint resumes only at the level it was written at.

### Tests and smoke checks

From the repository root, with `vc_object_store`'s Python module on the path
(its build directory's `python/`):

```sh
PYTHONPATH=spiral-fitting:$VC_OBJECT_STORE_BUILD/python spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_object_store_inputs.py \
  spiral-fitting/tests/test_object_store_materialize.py \
  spiral-fitting/tests/test_object_store_fibers.py \
  spiral-fitting/tests/test_object_store_overlaps.py \
  spiral-fitting/tests/test_object_store_tracks.py \
  spiral-fitting/tests/test_object_store_startup.py \
  spiral-fitting/tests/test_object_store_positions.py \
  spiral-fitting/tests/test_object_store_lazy_relink.py
```

These build small stores (`tests/object_store_fixture.py`; shared fixtures in
`tests/conftest.py`) and cover selection, frames and scales, refresh and its
idempotency, aggregated failures, read-only and approved-only modes, two-rank
startup (a rank-local failure, manifest mismatch), resume refusal, positions
linked and pinned on a hub replica or a store shared in place (including
edits racing the link refresh), lazy and incremental relinking, tracks and
their crossings (equal to the ones Spiral's own exact-crossing builder
finds), and the runtime constraints against
`tests/data/object_store_reference.json`, which the removed file loader
produced from the same fixture. An opt-in startup test refreshes and
materializes a real store (give it a scratch copy: it writes links):

```sh
OBJECT_STORE_SMOKE_STORE=/scratch/dataset.vcobj OBJECT_STORE_SMOKE_Z=9000:9100 \
  PYTHONPATH=spiral-fitting:$VC_OBJECT_STORE_BUILD/python spiral-fitting/.venv/bin/python \
  -m pytest -s -q spiral-fitting/tests/test_object_store_startup.py -k real_data
```

## Lasagna inputs must be packed first

A fit's Lasagna dataset is a `*.lasagna.json` manifest: the command-line
fit's input manifest names it (`lasagna.manifest`), a service's project
selects it. The fit reads its `nx`, `ny` and `grad_mag` channels, which must be single-channel
arrays sharing one array name (the manifest's `zarr` paths name it, e.g.
`las_008_nx.ome.zarr/4`), only through resident pools (`lasagna_data.py`:
there is no other loading path). Pools are derived data in the fit's cache
(`--cache`, default `~/.cache/vc3d/spiral`), keyed by the dataset's identity:

```
<cache>/lasagna/<identity>/normals_g<array>/   nx and ny
<cache>/lasagna/<identity>/grad_mag_g<array>/  grad_mag
```

A remote dataset (an `s3://`, `http(s)://` or `sftp://` manifest URL) is
identified by its URL: `url-` and a digest of the manifest URL, with the scheme
and host lower-cased and any fragment or trailing slash dropped. Data at a URL
is taken never to change; re-publishing different arrays at the same URL
leaves stale packs. Another spelling of the same bucket (an `https://` S3
endpoint for an `s3://` URL, a mirror) is another dataset.

A local dataset's identity is the `run_uuid` of the manifest's provenance
record (its `"provenance"` file, e.g. `inference.json`, as published datasets
carry), else `sha256-` and the SHA-256 of the manifest file; a manifest whose
named provenance record is missing is refused rather than identified
differently. A local copy of a remote dataset is a different dataset unless
you name the URL it was copied from: `pack_resident_pools.py --as <URL>` packs
it, and an input manifest's `lasagna.original_url` fits it, under that URL's
identity, so a service fitting the URL finds its packs. Either way a host
needs only the packs, not the arrays; the packer writes the dataset's source
document (`lasagna-<identity>.json`, recording its location and, where the
manifest states it, its frame: `base_shape_zyx`, `source_to_base` and the
channels' `scaledown`) beside them. The command-line fit derives `lasagna_scale` from that
frame (see "Input manifest").

The command-line fit builds a missing pool from the arrays (reporting chunk
progress; in DDP only rank 0 builds). By default a service requires prepared
packs in its cache. VC3D automatically passes `--build-missing-lasagna-packs`
when launching its owned local service. An independently launched service can
opt in with the same flag:

```sh
python spiral_service.py --object-store /path/to/store --output /path/to/output \
    --cache ~/.cache/vc3d/spiral --build-missing-lasagna-packs
```

The project supplies the dataset identity, array paths and scale. Missing packs
are built only from accessible local arrays, with shared packing locks and
progress reporting, before link refresh or replacement of the resident fit.
Checkpoint resumes prepare missing packs too, retaining their recorded input
position. Existing packs are reused without needing the source arrays. A packing
failure leaves the resident session intact. The same flag extracts missing
fiber-direction samples (see "Fiber direction samples"). It cannot be enabled
by a fit request.

A remote Lasagna dataset (an open-data one, or any whose arrays are on S3)
reaches a service on VC3D's host as its arrays' directories in VC3D's remote
cache. Before a build, VC3D fetches every object of the arrays whose packs the
service's cache lacks into those directories (logging its progress), so the
service packs them like local arrays; once packed, the cached arrays are no
longer read.

For a remote service, or one without this flag, pack where the arrays are and
copy the packs over:

```sh
python pack_resident_pools.py /path/to/dataset.lasagna.json \
    --as https://example/dataset.lasagna.json \
    --cache ~/.cache/vc3d/spiral --what normals,grad_mag
rsync -a ~/.cache/vc3d/spiral/lasagna/<identity>/ service-host:<cache>/lasagna/<identity>/
```

The packer only iterates over chunk files that exist, so a store that holds only
the z chunk rows of your fit window packs correctly; rows outside it read back
as no-data. Pass `--ct <scroll zarr> --ct-group <group>` to drop bricks outside
the CT mask.

## Sweep runner output

`runners/run_sweep.py` prefixes each active fit's live `PROGRESS` and
every-200-step loss lines with its configuration name. Optimization progress
includes the average iteration rate for the current stage (`it/s`). Complete
combined stdout/stderr for every attempt remains available under
`<output>/.sweep/logs/<config>.log`.



## Flattening a fitted checkpoint

`flatten_spiral_checkpoint.py` is a standalone, one-shot exporter. It
reconstructs the combined surface from a fitted checkpoint, launches a private
Lasagna service, flattens with `flatten_fast_nofilter.json`, writes the final
TIFXYZ directory, and tears the service down even if the job fails or is
interrupted:

```sh
python flatten_spiral_checkpoint.py \
    /path/to/checkpoint_fitted.ckpt \
    /path/to/output.tifxyz
```

The checkpoint format does not embed the fixed umbilicus curve. The script
looks for `umbilicus.json` in the checkpoint's ancestors, in
`$SPIRAL_DATASET`, and in the standard local s1 dataset location. For other
layouts, pass `--umbilicus /path/to/umbilicus.json`. Use `--lasagna-dir` if
the Lasagna repository is not in its standard sibling or `~/villa` location.
An existing output path is never overwritten.

## Hover descriptions (descriptions.json)

The text VC3D shows when hovering a Spiral tab control or a setting in the
Spiral Configuration window lives in `descriptions.json`, not in code:

- `config` is keyed by configuration key. The service attaches each entry to
  the `/configuration` catalog as the field's `description`, re-reading the
  file per request, so an edit shows after VC3D reconnects. A new
  configuration key needs an entry here; `tests/test_config.py` fails on a
  missing or stale key.
- `panel` is keyed by the control ids registered with `describe(...)` in
  VC3D's `SpiralPanel.cpp`. VC3D reads this section from the copy beside the
  `spiral_service.py` it would launch (the same lookup, including
  `SPIRAL_SERVICE_PATH`) and reloads it as soon as the file changes. A panel id
  without an entry falls back to the `config` entry of the same name.

Plain text only; a newline starts a new line in the tooltip.

## Flow stages

`model_num_flow_stages` sets how many stationary velocity fields the flow
diffeomorphism composes. The stages are the slabs of the flow lattices'
leading axis (`flow_field.flows.{0,1}` are `[stages, 3, ...]`), integrated in
order by one fused kernel launch per direction; the inverse runs the slabs
backwards in reverse order. One stage is the original single-field model.

## Flow-gradient conditioning

These optional settings change how the flow lattices are optimized, without
adding loss terms or changing the model parameterization. The smoothing,
lazy-moment and shared-second-moment switches default to off; gradient
clipping defaults to disabled. The `optimizer_flow_*` settings apply at run
boundaries and are read every step.

The step order is: DDP gradient averaging, NaN/Inf detection and replacement
with zero, optional clipping, optional smoothing, then the optimizer update.
Sanitizing before smoothing prevents a single invalid entry from
contaminating its neighborhood.

- `optimizer_flow_grad_smoothing` Gaussian-smooths each flow lattice's
  gradient. `optimizer_flow_grad_smoothing_sigma_voxels` sets the standard
  deviation in scroll-voxel units of the flow coordinate frame. Cartesian
  smoothing is isotropic. Cylindrical smoothing runs along z and periodically
  around each ring, independently for the local z/radial/tangential components.
  `optimizer_flow_grad_smoothing_across_sigma_voxels` optionally smooths across
  rings at matching angles; its default of 0 disables this pass. These are
  lattice directions approximating along/across-sheet directions, not measured
  sheet tangents or winding boundaries. Kernels are truncated at three sigma
  and renormalized at nonperiodic borders to preserve constant gradients.
- `optimizer_flow_grad_smoothing_low_res_sigma_voxels` overrides the coarse
  lattice's along-sheet width; 0 uses the same width as the fine lattice. With
  the default fine spacing of 16 voxels and sixfold coarse spacing, a width of
  32 voxels is 2 fine cells but only about 0.33 coarse cells. Very small widths
  become identity kernels. The fitter reports effective widths at startup
  when smoothing is enabled. The across-ring width has no separate coarse
  override and is ignored for Cartesian fields.
- `optimizer_flow_lazy_moments` updates moments and applies the gradient step
  only to entries whose **processed gradient is nonzero**. Smoothing can make
  an entry active even without a direct sample there. Other entries retain
  their moments; configured AdamW weight decay still applies everywhere.
  This preserves gradient-scale history through unsampled steps and suppresses
  momentum-only movement there, but retains stale history too. It does not
  prevent large relative steps on a previously untouched smoothing tail.
- `optimizer_flow_shared_second_moment` uses one denominator across all cells
  and vector components of each lattice's leading slab, independently for
  each flow stage and for the coarse/fine lattices. Per-entry Adam scaling can
  make even a smooth gradient produce nearly sign-sized steps, particularly
  on first touch. A shared denominator preserves relative magnitudes and
  direction of the first moment before lazy masking and weight decay; it
  does not guarantee a smooth final displacement. The shared statistic is the
  mean of positive stored second moments, capped before averaging at
  `optimizer_flow_shared_second_moment_clip_quantile` (default 0.99). A value
  of 1 disables the cap. This limits outliers' effect on the common scale at
  the cost of local adaptivity. Full per-entry moments remain stored.
- `optimizer_flow_grad_clip_median_multiple` clips individual gradient
  components to this multiple of the median nonzero absolute component value,
  separately for each lattice slab. Zero disables it. Clipping before blur
  limits how far an extreme gradient can affect neighbors and optimizer
  moments, but can also suppress legitimate corrections and change vector
  direction. The median and second-moment cap are estimated from fixed-stride
  samples; the final averages and clipped fractions use the whole slab.
  Sampling is deterministic for identical inputs but may miss sparse support.
- `model_flow_field_low_res_lr_scale` multiplies the scheduled base learning
  rate for the coarse lattice independently of the fine lattice's existing
  ramp. It defaults to 1; 0 freezes coarse parameter values, including weight
  decay, although moments may still update. The fitter reads it every step,
  but the configuration catalog currently classifies it as a model-rebuild
  setting, like the other `model_` learning-rate controls.

`flow_grad_smoothing.py` provides Triton CUDA kernels and PyTorch reference/
fallback paths. `lazy_moment_adamw.LazyMomentAdamW` retains AdamW's state format
so its flags can be switched between runs without converting moment buffers.
For custom optimizer updates, diagnostics report the denominator, nonzero
update fraction, and per-component RMS over nonzero updates in voxel units.
These describe flow-parameter increments, excluding weight decay, not final
sheet displacement after integration. After both moment flags are disabled,
stored optimizer diagnostics can still describe the last custom step rather
than the current fused AdamW step. Clipping diagnostics report the bound
and fraction clipped. No full-scroll accuracy or throughput comparison is
established by the implementation tests.

### Rationale and references

Gaussian update smoothing has precedent in
[Vercauteren et al., *Diffeomorphic Demons*](https://www-sop.inria.fr/asclepios/Publications/Tom.Vercauteren/DiffeoDemons-NeuroImage08-Vercauteren.pdf).
Smooth velocity metrics are central to
[Beg et al., *Computing Large Deformation Metric Mappings*](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Beg05.pdf).
These motivate the preconditioning here; this discrete blur followed by Adam,
clipping and masking is not an implementation of classical Sobolev gradient
descent or LDDMM, and inherits no guarantee of the same fitted solution or
fold-free numerical integration. Direction-dependent smoothing has a related
motivation in [Pace et al.'s sliding-organ registration](https://pmc.ncbi.nlm.nih.gov/articles/PMC4112204/),
although this fitter uses cylindrical coordinates rather than detected sliding
interfaces.

Lazy moments follow the masked-update idea of
[PyTorch SparseAdam](https://docs.pytorch.org/docs/main/generated/torch.optim.SparseAdam.html),
using an explicit nonzero mask on dense gradients, a global per-parameter step
counter, AdamW's epsilon placement, and optional decoupled weight decay.
[Adam-mini](https://arxiv.org/abs/2406.16793) provides precedent for sharing
adaptive scales within parameter blocks; our grouping and winsorization are
custom choices, and retaining full moments does not provide its state-memory
saving. [Koloskova et al., *Revisiting Gradient Clipping*](https://proceedings.mlr.press/v202/koloskova23a/koloskova23a.pdf)
analyze clipping's stabilization and stochastic bias; their results do not
validate this particular sampled-median rule or its combination with Adam.

## Spiral service host setup

VC3D connects to a Spiral service in one of three modes, all speaking the same
authenticated HTTP protocol:

- **Localhost** — VC3D launches and owns the service on loopback. Nothing to
  set up beyond the Python environment; the optional output, cache and
  winding-inference paths are chosen in the connection panel and VC3D
  launches the service bound to the project's store with those values and
  `--build-missing-lasagna-packs`: local Lasagna arrays, and remote ones VC3D
  first caches in full, are packed automatically on first use. Changing a
  binding restarts the owned service.
  Automatic service startup waits until you first open the Spiral workspace.
  If no project is open then, it waits for the project's store. Disconnect
  cancels that wait; revisiting the workspace does not reconnect it.
- **Remote (SSH)** — the supported internet flow. SSH access to the host is
  the only client-side prerequisite: VC3D opens and manages its own SSH
  tunnel, reads the service's auto-generated API key over SSH, and attaches to
  a persistent loopback service you start on the host. VC3D never starts the
  service on a remote host.
- **Remote (LAN)** — direct HTTP on a trusted network, authenticated with the
  service's auto-generated API key. No reverse proxies, VPNs, or manual
  tunnels are ever required.

In every mode the service — not the client — owns the host paths: it is
launched with `--object-store` (patches, point collections, fibers, tracks and
their links; see [Object-store inputs](#object-store-inputs)), `--output` (all
generated state), the optional `--cache` (derived host caches; default
`$XDG_CACHE_HOME/vc3d/spiral`, i.e. `~/.cache/vc3d/spiral`) and the optional
`--winding-inference` (below), resolves them once at startup, and advertises
them through `/bindings`. There is no dataset: the scroll's facts, the
umbilicus, the Lasagna dataset and the fiber-inference dataset come from the
client's project with each session request (below); the outer shell is the
store's surface tagged `spiral-outer-shell`. Clients change run parameters
but cannot repoint the session at other host paths or another object store;
a request naming any input path other than a checkpoint is refused.

`--winding-inference PATH` binds the winding-inference artifact (the crossing
directory holding its `manifest.json`; see "Neural winding-inference
losses") that winding-model supervision reads directly from that host path.
It is a supervision input, not a cache and not a VC3D project asset: VC3D
never downloads or interprets it, and no client can override it. `/bindings`
reports whether it is bound and usable (`bindings.winding_inference`:
`bound`, `path`, `available`, `error`; `capabilities.winding_inference`), so
VC3D can say whether winding-model supervision can run. A fit enabling it
without a usable binding fails preparation, leaving any resident fit as it
was. Changing the binding means restarting the service and preparing a new
fit. A named path that is not a directory refuses the launch.

### The VC3D project's inputs

A service session fits the VC3D project that asks for it. The project file
is VC3D's and need not be readable on the service's host, so every initialize
and rebuild carries a `project` block (`fit_session.SpiralProject`):

- the project's name;
- its selected Lasagna dataset as VC3D's manifest reader resolves it: its
  identity (see "Lasagna inputs must be packed first"), the OME-Zarr stores
  of the `nx`, `ny` and `grad_mag` channels (one-channel arrays sharing one
  array name, e.g. `s_nx.ome.zarr/4`; local paths or remote locations), or
  why the fit cannot read it (an unreadable dataset, multichannel arrays).
  The fit and its startup link refresh read the identity's pools from the
  service's cache, never the stores;
- for each store volume its volumes' annotations are kept in (a catalog
  volume once, whatever its working level): the facts the project knows
  exactly (`voxel_size_um`; a catalog volume's `z_direction_is_top_to_bottom`,
  `left_handed_coordinates` and full-resolution `base_shape_zyx`), the
  Lasagna arrays' downsampling relative to its voxels (`lasagna_scale`, from
  the manifest's `source_to_base`, scaledown and base shape and the volume's
  shape) and its umbilicus from the project's umbilicus store (`uuid`,
  `revision`, `points_zyx` in the volume's own voxels).

The fit volume is the one the selection's objects name (a checkpoint resume:
the one its inputs were read in). The build fits that project volume:

- **Scroll.** The volume's facts and the project's name. Where the volume's
  orientation is unknown (a volume outside the catalog) the fit setting
  `spiral_outward_sense` (the Spiral panel's *Outward sense*) gives the
  sense; where it is known, a setting that disagrees is refused.
- **Lasagna.** The project's selected dataset, local or remote, by its
  identity: the service writes it as `project-inputs/lasagna-<identity>.json`
  (the file the fit reads) and loads its pools from its cache at the volume's
  `lasagna_scale` (see "Lasagna inputs must be packed first"). A fit whose
  losses need normals or gradient magnitudes the project cannot give, or
  whose pools the cache lacks, is refused before anything is linked. The
  startup link refresh reads the normals pack too (`normal_pool`); without
  one in the cache it refreshes fiber links centerline-only, with a warning.
  The command-line fit builds its normals pack before refreshing.
- **Umbilicus.** The project's, written under the output directory
  (`project-inputs/umbilicus-<revision>.json`) as the file the fit reads, at
  coordinate scale 1.

The service reads no scroll specification or input manifest (those are the
command-line fit's); winding counts come from the fit configuration. A fit
volume that is not a project volume, or has no umbilicus, fails the
preparation and leaves any resident fit as it was. The session's
`session_request` records `project` (name, volume, its facts, the Lasagna
dataset, the umbilicus's identity), the resolved `scroll` and `paths` (the
inputs read by path, the fit volume as `volume_uuid`, and their
`provenance`: the project and the service's bindings), which checkpoints
record too. A
rebuild after the project's volume changed (another umbilicus revision, a
renamed project, another Lasagna dataset) rereads everything.

### Object store, positions and input updates

`--object-store STORE.vcobj` binds the service to one store for its lifetime;
`--object-store-author NAME` names the author of the links it writes (default
`spiral-service@<host>`). `object_store_path`, `object_store_author` and
`object_store_input_position` are service-owned configuration: `/configuration`
lists them under `service_owned`, a request that leaves them out or repeats
the bound values is accepted, and a request naming another value is refused.
VC3D removes them from saved profiles before sending a request.

**A service on a plain store** (no `--object-store-hub`) shares the file in
place with the clients on its machine (VC3D's project store): binding it
makes the store record positions of its own (its journal head after each
write, by any connection; the store's schema version becomes 18, so binaries
from before the index refuse it). There is no hub, no replica and no sync:
VC3D's saves are in the file the service reads. The service links over the
selection and pins the new head when every write since the head it linked
at is its own; another writer in between (a save, an approval) makes it
relink at the new head, up to the same bound. Links it wrote for a head it
then abandons stay as pending detector output, which the next refresh
replaces. Keep the store on a local disk (SQLite's WAL does not work over
network filesystems); a service on another machine needs a hub.

**A hub-bound service** (`--object-store-hub HUB`, a path or an
`sftp://[user@]host/path` address) works on its own replica of that hub:
STORE is cloned there when absent and otherwise checked to replicate HUB; a
STORE that already is a replica binds its own hub without the flag. Startup
pulls once, so an unreachable hub fails the launch; for an `sftp://` hub the
service's ssh runs in batch mode, so its account needs a key (or agent) with
write access to the hub file and its directory and a `known_hosts` entry.
SSH hubs now require one-time migration with all clients stopped:

```sh
vc_object_hub enable-ssh-locking sftp://data-host/share/scroll.hub.vcobj
```

The server needs Linux, SSH command execution, `flock`, `timeout`, and an
external OpenSSH `sftp-server`. Every client must be updated and use the SSH
address after migration, including a service on the hub host: do not mix SSH
process locking with local-path or mounted access. Previously mounted
replicas must push before migration and be recloned through SSH afterward.
The resulting `.lock/process-v1` is permanent and must not be removed.
See the store's [hub migration instructions](../vc_object_store/docs/README.md#hosted-hub-and-replicas).
For operations longer than the default 30-minute SSH session deadline, set
`VC_HUB_SESSION_SECONDS` on the client before launch.

The service's replica writes only detector links, pulls and pushes
only during a build or an update, and is the only detector writer for its
selection (a command-line refresh against the same hub would conflict with it).

```sh
python spiral_service.py --output OUT --object-store ~/spiral/replica.vcobj \
    --object-store-hub sftp://data-host/share/scroll.hub.vcobj
```

Each build (Initialize Fit, a full Rebuild Fit) links and pins a position (the
hub's, or the shared store's own), including at least the request's
`inputs_position` (the position a client's saved, or pushed, edits reached),
before tearing anything down: a failure there leaves
the resident fit untouched and is reported in `input_preparation_error`. A
checkpoint resumes at the position it recorded. A model-only rebuild keeps the
position. `POST /session/update-inputs {"inputs_position", "expected_input_generation"}`
updates a resident fit in place: the service links and pins a position, then
every rank reads it, prepares its inputs without touching resident state,
releases the old inputs' device state and installs the new ones at an
iteration boundary. The model, optimiser, scheduler and iteration are kept;
status reports `input_update` while one runs and `input_generation`
afterwards. A refusal (while linking or preparing, on any rank) leaves the fit
as it was; a failure after the release fails the session
(`input_preparation_error.recovery_required`): rebuild it or load a
checkpoint.

Tracks and their crossings are consumed at build. If either changes, an
input update is refused: use **Rebuild with Current Inputs**. The same
applies to outer-shell changes and to input updates while tracks use
trusted-geometry exclusion (`track_exclusion_radius > 0`).

Between builds and updates the inputs are frozen: runs, pauses, checkpoint
loads and model-only rebuilds keep them, and `POST /session/rebuild {"inputs":
"reload"}` (VC3D: *Rebuild Fit* → *Rebuild with Current Inputs*) rereads the
store. Without `"inputs"` a rebuild that changes only model configuration
keeps the inputs and one that changes anything else rereads them; the
response's `inputs` says which happened.

Status carries `input_digest` (the replica-independent digest of the fit's
selection), `input_summary` (counts per kind and relationship, the volume,
whether links were refreshed) and `input_position` (hub, sequence, the
requested position, the head it was linked at, and where it came from). Every
build and update publishes the selection, with each object's and
relationship's revision pins, as an `input_manifest` artifact
(`object_store_inputs.json`). Previews, the preview index and autosave
metadata record the digest and position of the inputs they were made from. A
checkpoint whose recorded inputs differ from the session's is refused with the
differences, the `{"inputs": "reload"}` remedy and the
`object_store_allow_changed_inputs` override.

Only one client changes a session at a time: mutating requests carry the
`X-Spiral-Session-Token` of the client holding the session lease
(`/session/lease/claim`, `/session/lease/release`), and one service holds the
lease file under its output directory. Uploads accept resume checkpoints only.

### Creating the Spiral Python environment

The service host needs the Spiral environment (a CUDA-capable PyTorch plus the
dependencies in `pyproject.toml`, Python ≥ 3.14). With [uv](https://docs.astral.sh/uv/):

```sh
cd spiral-fitting
uv sync            # creates .venv from pyproject.toml
```

This also builds Spiral's native helpers as `vc_spiral.spiral_sampling`,
`vc_spiral.track_crossings` and `vc_spiral.track_store`. OpenMP is used when the toolchain provides it; the
same modules build with serial kernels when it does not.

or with conda/pip, install `torch` for your CUDA version and then
`pip install -e . --group cuda` from `spiral-fitting/` (pip 25.1 or newer).

On Linux x86-64, `uv sync` installs `brook-cu12`, CuPy and cuCIM for the GPU
backend of `extract_surface_tracks.py` instead of Kimimaro, its CPU backend;
on a machine without an NVIDIA GPU of compute capability 8.0 or newer, add
Kimimaro with `uv sync --extra cpu` (with pip: `pip install -e '.[cpu]'`), and
pass `--extra cpu` on later syncs too, since a plain `uv sync` removes it again.
Other platforms get Kimimaro by default.

On AMD GPUs (ROCm, Linux x86-64) the CUDA-only packages are not needed and
`uv sync` would pull the CUDA 12 build of PyTorch, so create the environment
with a ROCm PyTorch build and without the `cuda` dependency group instead. The
Kimimaro CPU backend of `extract_surface_tracks.py` is used (`--backend cpu`):

```sh
cd spiral-fitting
uv venv --python 3.14
uv pip install --index-url https://download.pytorch.org/whl/rocm7.2 "torch>=2.11,<2.13"
uv pip install -e '.[cpu]'
.venv/bin/python -c 'import torch; print(torch.version.hip, torch.cuda.is_available())'
```

Use `.venv/bin/python` (or `uv run --no-sync`) afterwards: a plain `uv sync`
or `uv run` would reinstall the CUDA build of torch and the `cuda` group.

### Resident sparse field pools

Normals and gradient magnitude samples are served by fully
resident device brick pools. Each pack's occupied bricks are packed once
into the fit's cache by `pack_resident_pools.py` (see "Lasagna inputs must be
packed first" for the layout):

```sh
python pack_resident_pools.py /path/to/dataset.lasagna.json --cache <cache> \
    --ct /path/to/<scroll>_ds2.zarr --ct-group 2 --verify 2000
```

`--ct` zeroes every voxel whose CT voxel reads 0 (the mask region around the
scroll) so those bricks drop out of the pool and sample as no-data. The
fitter loads the pools restricted to the configured z-ROI in one
sequential read per channel (for the full s1 ROI: ~10 GiB normals); after
that every gather is pure device indexing with no I/O and no
eviction. When a required pool is missing, the command-line fitter builds it
before GPU loading and reports chunk progress. In DDP runs only rank 0 builds
it. Manual prepacking with `--ct` remains useful because the CT mask can
substantially reduce the resident pool size.
Set `FIT_SPIRAL_RESIDENT_BOUNDS_CHECK=1` to enable per-gather bounds
assertions when debugging new sampling code.

### Internet flow (SSH attach)

Start a persistent loopback service on the GPU host with its store. Give
each independently operated service a stable session name, port, and GPU.
Nothing is exposed on the network; VC3D tunnels to it over SSH:

```sh
tmux new -s spiral-alice 'python spiral_service.py --port 8765 \
    --object-store /data/scrolls/s1.vcobj \
    --output /data/spiral-output/s1 --gpus 0 --session-name alice'

tmux new -s spiral-bob 'python spiral_service.py --port 8766 \
    --object-store /data/scrolls/s1.vcobj \
    --output /data/spiral-output/s1 --gpus 1 --session-name bob'
```

The service uses only physical CUDA device `0` by default. Select a different
device or enable distributed fitting across several GPUs with a comma-separated
host-side list:

```sh
python spiral_service.py --port 8765 \
    --object-store /data/scrolls/s1.vcobj \
    --output /data/spiral-output/s1 --gpus 0,1,2,3
```

Multi-GPU sessions run one fitter rank per listed device and split the configured
per-step sample counts across those ranks by default. The device list is fixed for
the lifetime of the service; restart it to change the selection.

A named service writes autosaves, previews, artifacts, uploaded checkpoints
and Lasagna output beneath `<output>/<session-name>/`, held
under an exclusive lease: two live services cannot own the same
output/session-name pair. Launches without `--session-name` use `<output>/`
directly. Bound inputs (the winding-inference artifact) and the shared user
cache stay untouched.

Every completed Spiral preview is flattened by the host's Lasagna service
before it becomes downloadable in VC3D. The published grid uses a fixed
20-voxel output step: each dimension is
`ceil(((source_points - 1) * source_step) / 20) + 1`. Winding membership,
loss-map overlays, and run differences are transferred through Lasagna's
output-to-source correspondence so they remain aligned when the output grid
dimensions differ from the Spiral grid. If flattening or artifact mapping
fails, the service reports the publication error and VC3D keeps displaying the
previous successfully published preview.

Every checkpoint and every raw preview export carry a content digest of the
model state that placed the surface (`model_state_sha256`: live parameters,
frozen constraint-bake epochs and run window). The service records each
published preview under that digest in
`<output>/.spiral-published/preview-index.json`. When the fitter reports that
its resident model equals a checkpoint - after a save, an in-session load or a
startup resume - the service re-shows the flattened surface it already
published for that digest instead of waiting for a new export and flatten, and
pins that surface against the fixed-count preview retention for as long as the
checkpoint file exists. A checkpoint saved without a published preview, or one
whose preview has already been pruned, gets nothing back; request a preview as
before.

On first start the service generates a strong API key at
`~/.config/vc3d/spiral_api_key` (mode `0600`) and prints it to the console.
For an SSH profile you never copy it: VC3D reads that file over SSH.

In VC3D's Spiral workspace, add a *Remote (SSH)* profile with the
`[user@]host` destination (your `~/.ssh/config` aliases, agents, and jump
hosts work unchanged) and the service port (`8765` above), then Connect.
Non-interactive SSH authentication (keys or an agent) is required. If SSH does
not trust the host key yet, run `ssh <destination>` once in a terminal to
accept it — VC3D deliberately never auto-trusts host keys.

The fit survives viewer disconnects, laptop sleep, and network drops;
disconnecting or closing VC3D never terminates a service it did not launch.
The workspace reports the active loading, optimization, checkpoint, and
preview stage with elapsed time. Stages with a real work total also show a
counter and ETA; opaque native or CUDA operations deliberately use an
indeterminate bar instead of a guessed overall percentage. The same stage
updates are printed by standalone `fit_spiral.py`, with periodic elapsed-time
heartbeats when output is captured to a log.
While connected, the circular-arrow button beside the connection controls
restarts the remote service and reconnects automatically. The service replaces
its own process in place, so a containing `tmux` session remains alive and an
attached terminal is not disconnected.

### Trusted-LAN flow (direct HTTP)

```sh
python spiral_service.py --bind 0.0.0.0 --port 8765 \
    --object-store /data/scrolls/s1.vcobj \
    --output /data/spiral-output/s1
```

Copy the API key printed at startup into the *Remote (LAN)* profile's API key
field (or export `SPIRAL_API_KEY` before starting VC3D). A non-loopback bind
always requires an API key (auto-generated when absent).

**Plaintext-HTTP risk note:** direct HTTP is not encrypted — on-path observers
can read the API key and the transferred data, so use it only on networks the
operator trusts. Over the internet, use an SSH profile instead. HTTPS
endpoints behind an existing TLS proxy also work; VC3D uses normal system CA
validation and never ignores certificate errors.

### API key file

- Location: `~/.config/vc3d/spiral_api_key` (respects `XDG_CONFIG_HOME`), or
  pass `--api-key-file PATH`.
- The key is created on first start (mode `0600`) and reused on later starts.
- To rotate it, delete the file and restart the service; reconnect clients
  with the new key. The key is never written to HTTP logs, responses, or the
  ready line — the console print at startup is the intended way to obtain it.
- `--nonce` is only for processes launched and owned by VC3D.

### Bindings, output, and cache

A service has no dataset. `--dataset` is refused with migration guidance:
geometry comes from `--object-store`, the scroll's facts, umbilicus, Lasagna
and fiber-inference datasets from the VC3D project, and winding inference
from `--winding-inference`. `GET /bindings` advertises the host bindings
(`bindings.output_directory`, `cache_directory`, `winding_inference`), the
capabilities, the bound store and its selection, and the checkpoints under
the output directory that a load may name. Reading a large store's selection
can take minutes, so `/bindings` never waits for it: until the background read
for the store as it stands finishes, `inputs` is `{"pending": true}`; ask again
for the summary.

`--output` is required. Every piece
of generated state — run directories, autosaves, previews, published
artifacts, upload staging, and uploaded checkpoints — lives
under it (under `<output>/<session-name>` for a named service). Make sure its
filesystem has room for checkpoints and previews.

`--cache` holds derived host caches (content-addressed, shareable between
scrolls): prepared Lasagna packs and fiber directions, selected by the
project's datasets' identities. It defaults to `$XDG_CACHE_HOME/vc3d/spiral`
(`~/.cache/vc3d/spiral`). The winding-inference artifact is never read
through it. The headless `fit_spiral.py` CLI accepts the same `--cache` with the same default
(`FIT_SPIRAL_CACHE_DIR` still overrides it for the CLI).

The service never writes its bound inputs; its only store writes are the
link refreshes, as `--object-store-author`.

### Connecting from VC3D

Open the Spiral workspace and pick the profile in the *Spiral Service*
section. For the local profile, optionally set output, cache and
winding-inference paths there — VC3D launches its owned service bound to the
project's store and those values (an empty output defaults to a directory per
store under VC3D's data directory). A remote service's bindings are set on
its host. Connection must succeed (an authenticated `/health` handshake and
an API-version check) before session controls enable. The output and cache
rows populate read-only from the service's `/bindings`, and the panel says
whether winding-model supervision can run there; run parameters (z range,
iterations, advanced config) stay editable and persist per profile.
Generated previews, geometry, and checkpoints transfer through the artifact
API into a local cache — no shared filesystem is needed.

The fit's patches, point collections and fibers are displayed from an
object store attached to the VC3D project, never from service files: the
store the project routes surfaces, fibers and point collections to (*File >
Stores > Show Surfaces From*, and so on). Spiral needs all three routed to
one store; when the routes differ or are unset it has none. A local profile
launches its service with `--object-store` set to that store; for a remote
profile, attach a replica of the store the service was started with and
route the three kinds to it. The panel's *Object store* row compares the
store UUIDs: when they differ, or there is no such store, the fit's inputs
are not displayed. When they match, VC3D downloads
the session's input manifest and shows each fitted surface, collection and
fiber as the attached store holds it now: the store is edited live, and a
commit that touches a fitted input or link updates just that input's overlay
and the notes, so they agree with the rest of VC3D. The resident fit does not follow. *Fit
inputs* lists the digest, the counts per kind and the links, and notes what
differs from what the fit read: inputs no longer in the store, inputs edited
or with other metadata, fitted links that changed or would no longer be
consumed, links among the fitted inputs that were not fitted, and other store
changes a rebuild's selection may include; *Rebuild Fit* reads them.
*Show fitted fibers* in Display draws the fit's fibers on the plane views,
within the point view tolerance of the slice.

A local profile on a plain store launches its service on that same file:
saved edits are already where the service reads them, and *Update Fit
Inputs* names the store's head. A direct profile to a loopback address is
treated the same way when the service advertises the store's path; any other
profile cannot name positions of a plain store it does not share, so its
updates need a hub. With a hub-bound service the profile works through a
replica of the service's hub: a local profile whose store is a replica
launches its service with its own replica of the same hub (under the output
root). For a remote profile, attach a replica of the service's hub; *Clone the
Service's Hub* (shown when Spiral's store is not one) prefills the hub
address, through the SSH destination for an SSH profile, or the profile's
saved hub address, and routes the three kinds to the replica. The
*Object store* row then compares hub identities and positions as well as
store UUIDs.

Edits are saved into the store (or replica), not into the fit. *Save Drafts
to Store* (Shift+E) writes the ready brush patches (new surfaces tagged
`spiral-brush`, in the preview's frame of the fit volume, or new revisions of
ones saved earlier) and point-collection edits (fitted collections keep their
objects and point ids) as checked revisions; a replica's background sync
pushes them. *Update Fit Inputs* updates an idle fit in place at a position
including those edits (initializing one when there is none, rebuilding a
failed one). With a hub it first holds the replica's pushes, pushes
everything saved and waits (hub conflicts stop it and are named), and
afterwards pulls to the position the service fitted. It reports the hub commits that position includes beyond the
request and those after it, which show as stale. *Save and Update Fit
Inputs* does both. The run panel's position summary shows the four states:
unsaved drawings, saved edits not yet pushed (or in conflict), a position
being linked, and the position the fit reads.

The VC3D project is the only source of the scroll's name and voxel
resolution and of the Lasagna dataset (its arrays and scale). None of them
are panel settings: the panel reports them
read-only, and the service rejects a session request that carries
`scroll_name`, `voxel_size_um`, `lasagna_group` or `lasagna_scale`.

Optional supervision sources have rebuild-scoped boolean switches in Advanced
config. Set an `input_use_*` key to `false` to skip validation, loading,
sampling, and losses for that source without changing its tuned weights or
sample counts. Available switches cover verified patches, tracks,
fibers, each PCL role (`absolute`, `relative`, `same_winding`, and
`drawn_control_points`), normals, gradient magnitude, winding inference, and
the outer shell. For example:

```json
{
  "input_use_tracks": false,
  "input_use_fibers": false,
  "input_use_pcl_drawn_control_points": false
}
```

Every role switch requires a whole-fit rebuild: the switches select the
fit's frozen object-store inputs, and a disabled role's store content is
simply not selected. Disabling a prerequisite also disables its dependent
supervision: winding inference needs the outer shell.

Patches, point collections and fibers cannot be added to or edited in a
resident fit from VC3D: *Add* and *Commit* stay disabled and say why. Edit
them in the store and rebuild with current inputs. The drawing tools below
still work and keep their drawings in VC3D, listed as local drafts in the
input list (Remove discards one), for store-backed editing to submit; leaving
the workspace with drawings asks before discarding them.

In the flattened Spiral preview, tap `Ctrl` to toggle patch painting: left-drag
paints, right-drag erases, and `Escape` exits. Ctrl+wheel changes brush size
without toggling the mode; Shift+right-drag draws control-point lines. Starting
a stroke on a session-drawn selection extends that patch in its original color.
Drawn selections remain brush-editable across preview updates when their
original surface can be projected onto the new one. Finalization keeps the
largest edge-connected component of complete quads; selections with no
complete quad remain editable and report an error.

Brush regression checks use the existing build and Python environment:

```bash
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build --target VC3D test_spiral_brush_patch test_spiral_point_placement_mode test_spiral_point_collection_edit -j 4
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen ctest --test-dir volume-cartographer/build -R '^spiral_(brush_patch|point_placement_mode|point_collection_edit)$' --output-on-failure
```

Set `SPIRAL_PATCH_REAL_INPUT` to a real tifxyz patch directory to include the
read-only selection check. `SPIRAL_PATCH_EVIDENCE_DIR`
optionally receives before/after selection-mask images. These checks cover CPU
geometry and the editing workflow; they do not run GPU fitting.

In the Spiral workspace, `Q` and `E` place same-winding and relative-winding
points. Relative annotations count 0, 1, 2, ... in placement order; `F` reverses
the chain and mirrors annotations. Shift+E marks local drafts ready.

Protocol and acceptance checks for the store-backed service (from the
repository root; `vc_object_store`'s Python module on `PYTHONPATH`):

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=spiral-fitting:<store-build>/python spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_service_object_store.py \
  spiral-fitting/tests/test_runtime_object_store.py \
  spiral-fitting/tests/test_service_hub_positions.py \
  spiral-fitting/tests/test_service_mutations.py \
  spiral-fitting/tests/test_workspace_cleanup.py \
  spiral-fitting/tests/test_spiral_service_v2.py
AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build --target test_spiral_store_inputs test_spiral_service_protocol -j 4
AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen SPIRAL_TEST_PYTHON="$PWD/spiral-fitting/.venv/bin/python" \
  PYTHONPATH=<store-build>/python ctest --test-dir volume-cartographer/build -R '^spiral_' --output-on-failure
```

`spiral_service_protocol` starts `spiral_service.py` on a real store (its
runtime faked by `tests/protocol_client_service.py`) and drives
`SpiralServiceManager` through the lease, store binding, input manifest and
reload rebuild; it is skipped without `SPIRAL_TEST_PYTHON`.

Directional DT timing is an independent control on every interactive Run.
When **Restrict DT losses to final** is unchecked, the Run adds no DT gate.
When checked, the adjacent percentage is the eligible suffix of the originally
requested Run: the first eligible iteration is
`run_start + floor(iterations * (1 - percentage / 100))`. Thus 25% of a
10,000-iteration Run suppresses DT for 7,500 iterations and permits it for the
final 2,500; for small Runs the eligible step count is rounded up. Zero percent
suppresses DT for the whole Run and 100% adds no suppression. Stopping early
does not recalculate the original window. The schedule is transient Run state,
not advanced configuration or checkpoint state, and is cleared before the
Run's autosave.

**Checkpoints** are one panel section, and loading one is one button. It lists
what the service advertises (checkpoints under its output directory, such as
the autosave) plus any **client-local `.ckpt`**
you browse for; a local file is uploaded to the service's
`<output>/uploaded-checkpoints/` directory on the way (the panel shows
progress and the transfer restarts if interrupted). Checkpoints are identified
by SHA-256, so choosing content the service already retains reuses it without
transferring the file again, and the service validates new archives and keeps
the newest few unique uploads.

Before the first fit is initialized, *Load* initializes it directly from the
selected checkpoint; it does not first construct a throwaway model. The
configuration profile becomes **Checkpoint** and displays the resolved
configuration carried by that checkpoint. *Initialize Fit* is the separate
from-scratch action.

With an existing fit, *Load* replaces the resident model's weights, optimiser
and RNG state in place. When the checkpoint does not match the live model the service refuses
it and says what a rebuild would have to replace: rebuilding the **model only**
keeps the frozen inputs, while a **whole-fit** rebuild re-reads the project's
inputs and the object store. A checkpoint fitted to other store inputs is refused with
the differences and the reload rebuild as the remedy. The panel reports the reasons and asks; a checkpoint no
rebuild can accept — one whose inputs are in another volume, or whose stored
configuration holds a value the schema cannot interpret — is reported and
nothing is offered. A checkpoint-backed session takes its durable configuration
from the checkpoint, so the local advanced-config profile does not override it.

A checkpoint's stored configuration is loaded tolerantly
(`checkpoint_migrations.tolerate_config`): keys the schema no longer has are
dropped and keys the checkpoint predates take their current defaults. Each
such edit is reported as a note or session warning. Only a stored value the
schema cannot validate (a retired enum member, an out-of-range number) refuses
the checkpoint. Tolerance covers configuration alone: a checkpoint whose
parameters do not fit the live model — a multi-stage flow saved in the
pre-slab `extra_flow_fields.*` layout, or an exponential-gap fit from before
late August 2026 — is still refused on tensor geometry.

The Iterations value on *Run* is a count added to the checkpoint's durable
iteration. The progress bar is local to that run and therefore starts at zero;
the session status line reports the global current and target iterations.

The section also holds *Save on Service* and *Download…*, and reports the
checkpoint the resident fit was actually built from. That report is read-only:
it is not a field, and a rebuild carries it forward by itself.

### Shutdown and logs

Stop the service with `Ctrl-C` or `SIGTERM` (`tmux kill-session -t spiral`);
it tears the fit session down at a safe boundary. Logs go to the service's
stdout/stderr on the host — for a `tmux` session, `tmux attach -t spiral`; for
an unowned service VC3D's Python-output dialog only reminds you of this. A
service started on an explicit port can be restarted immediately (the socket
uses `SO_REUSEADDR`). VC3D's remote restart control does not run
`tmux kill-session`; it gracefully closes the fit and re-executes the service
with the same interpreter, arguments, and process ID. Note that a large artifact
download during a running fit competes with the fitter for the Python
interpreter and can slow iterations somewhat.

### Optional systemd user unit

```ini
# ~/.config/systemd/user/spiral-service.service
[Unit]
Description=VC3D Spiral fitting service

[Service]
WorkingDirectory=%h/villa/spiral-fitting
ExecStart=%h/villa/spiral-fitting/.venv/bin/python \
    %h/villa/spiral-fitting/spiral_service.py \
    --port 8765 \
    --object-store /data/scrolls/s1.vcobj \
    --output /data/spiral-output/s1 --gpus 0
Restart=on-failure

[Install]
WantedBy=default.target
```

```sh
systemctl --user daemon-reload
systemctl --user enable --now spiral-service
journalctl --user -u spiral-service -f     # logs (includes the API key print)
```

Direct command-line use remains fully supported; the unit is a convenience.

## Extracting surface tracks

`extract_surface_tracks.py` turns a surface-prediction volume into the tracks
DBM that the next sections pack, index and rasterize. The predictions are
binarized (`> 0`) and cut into thin slabs: horizontal ribbons 4 voxels deep
every 16 voxels of z, then vertical zx and zy slabs 4 voxels thick every 64
voxels across the occupied yx range. In every slab the connected components
are labelled, max-pooled by 4, filtered by size and skeletonized, and each
skeleton chain between branch or end points with at least 10 vertices becomes
a track.

```sh
python extract_surface_tracks.py \
    --predictions /path/to/<surface-predictions>.zarr/0 \
    --out /path/to/<dataset>/tracks/<name>.dbm \
    --z-min 10900 --z-max 11300
```

Every option defaults to the configuration at the top of the script, which
also holds the slab geometry, the size thresholds and `path_mode`. Its two
paths are placeholders: the script stops with an error until they are set or
passed as `--predictions` and `--out`. `--predictions` takes a local path or a
URL such as `s3://...`, opened with `open_zarr` from `../vesuvius/src`; the GPU
backend reads a local zarr v2 array in the layout of the surface predictions
(uint8, blosc, `/` chunk keys) straight from its chunk files instead. The z
range is half-open (`[z-min, z-max)`).

Two backends write the same keys in the same format:

- `gpu`: Brook (`brook-cu12`) and cuCIM on NVIDIA GPUs, one worker process
  per GPU. It needs Linux x86-64, GPUs of compute capability 8.0 or newer
  (Ampere or later) and an NVIDIA driver R545 or newer (R570 or newer where the
  driver JIT-compiles Brook's PTX). The CUDA 12 libraries come as pip wheels
  (`brook-cu12` bundles its runtime; CuPy and cuCIM use NVIDIA's CUDA wheels
  installed with them), so no system CUDA toolkit is needed. `--gpus 0,1`
  selects GPUs by their `nvidia-smi` index; the default is every visible GPU.
  Without `--gpus`, a set `CUDA_VISIBLE_DEVICES` is read the same way
  (`nvidia-smi` indices, PCI bus order); UUID and MIG entries are not accepted,
  so pass `--gpus` instead. With `--gpus`, `--backend auto` does not fall back
  to Kimimaro when the GPU backend cannot run.
- `cpu`: Kimimaro, one slab at a time. `uv sync` installs it on every platform
  except Linux x86-64, where it is the optional `cpu` extra
  (`uv sync --extra cpu`) for machines without a compatible GPU.

`--backend auto` (the default) picks the GPU backend when its check passes
(platform, packages, and the NVIDIA driver version and the compute capability
of the selected GPUs, as reported by `nvidia-smi`) and `path_mode` is
`'interjoint'`, and Kimimaro otherwise. The
script prints the backend it uses and why. `--backend gpu` and `--backend cpu`
stop with the reason when that backend cannot run. The GPU backend implements
only the `'interjoint'` path mode.

Brook implements Kimimaro's skeletonization on the GPU. Its skeletons are close
to Kimimaro's but not always identical, so the two backends' DBMs agree closely
rather than byte for byte (see the measurements below). Kimimaro collects the
skeletons of its 8 worker processes as they finish, so the order of the tracks
within a key can also differ between two CPU runs.

Every key (`h:{z}`, `vy:{y}`, `vx:{x}`) holds a pickled list of `(N, 3)` int32
ZYX arrays in full-resolution voxels, and an empty slab holds an empty list.
Keys already in the DBM are skipped, so an interrupted run resumes where it
stopped, with either backend; delete the DBM to recompute it. When
`write_native_packed_store` is set (the default), the script then writes the
packed store described in the next section. That step imports `tracks.py`,
which needs PyTorch; `--no-packed-store` skips it, and `convert_track_store.py`
can write the store later.

The GPU backend decodes the z range once into a bit-packed block in POSIX
shared memory (`/dev/shm`) of about (z-max − z-min) × Y × X / 8 bytes, where Y
and X are the full extent of the volume, plus at most a quarter of that again
(at the default slab spacing) for the byte columns of the vx slabs. For 400
slices of a 7888 × 8096 volume the block takes 3.2 GB. The script stops with an
error when `/dev/shm` has too little free space.

Measured on Scroll 1 surface predictions with the default slab geometry and
thresholds. Times run from the start of the extraction to the closed DBM,
reading and decoding included, interpreter start-up and the packed store not:

| Input (Y × X) | z range | Hardware | Backend | Runs | Time |
| :--- | :--- | :--- | :--- | ---: | ---: |
| public predictions, 7888 × 8096 | 10900–11300 | Intel Core i9-14900KF | Kimimaro, 8 workers | 1 | 336.9 s |
| same | 10900–11300 | NVIDIA RTX 4090, driver 615 | GPU | 3 | 4.1, 4.2, 4.2 s |
| predictions, 8174 × 8174 | 10900–11300 | 4 × NVIDIA H100 80GB, driver 570 | GPU | 3 | 4.4, 4.5, 4.6 s |
| same | 10752–14848 | 4 × NVIDIA H100 80GB | GPU | 3 | 15.7, 15.9, 19.6 s |
| same | 10752–14848 | 1 × NVIDIA H100 80GB | GPU | 1 | 40.0 s |

The 19.6 s run was the first in a new environment (numba compiles and caches
its kernels once). On z 10752–14848 (446 keys) the GPU backend wrote 1,006,831
tracks with 41,530,237 points, and a Kimimaro run on the same range wrote
1,006,740 tracks with 41,543,412 points (+0.01 % tracks, −0.03 % points). The
GPU DBMs were byte-identical across runs and between 1 and 4 GPUs, and the tests
compare them with the per-slab loop run with `brook.skeletonize`. These are
single inputs; speed and agreement depend on the data and the GPUs.

Run the tests from the repository root:

```sh
PYTHONPATH=spiral-fitting spiral-fitting/.venv/bin/python -m pytest -q \
  spiral-fitting/tests/test_extract_surface_tracks.py
```

They compare `fast_tracks.py` with the networkx walk of the CPU backend on
fixed and random graphs, check the backend selection and the command line
without a GPU or Kimimaro, and run the CPU backend on a small synthetic volume
when Kimimaro is installed. With Brook, CuPy, cuCIM and a GPU of compute
capability 8.0 or newer, they also run the GPU backend on one GPU on two small
synthetic zarr stores, one read by the direct chunk reader and one through the
fallback reader, and compare the DBMs with the per-slab loop run with
`brook.skeletonize`.

## Packing large track databases

Legacy track DBMs store a pickled list of NumPy arrays in every key. For large
datasets this spends minutes decoding millions of Python objects each time a
fit starts. Convert a DBM once to the adjacent packed format:

```sh
python convert_track_store.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm
```

This writes `2um_ds2_ps256_surf_v2.dbm.vctracks/` atomically. The directory
contains contiguous coordinates, ragged offsets, source IDs, family codes,
Z bounds, arclengths, and tortuosities. It is what `vc_object_ingest --tracks`
imports into an object store (fits read tracks only from there), while the DBM
stays the authoritative source. A source-file fingerprint prevents a stale
store from being used after the DBM changes; rerun with `--force` to replace it.
The crossing builder below also stages directly from a current packed store,
bypassing DBM and pickle decoding.

## Caching exact track crossings

A fit takes its track crossings from the object store (its link refresh runs
the store's track-crossing detector, which uses the same exact-crossing
kernel). `grow_track_graph.py` instead reads them from a CSR sidecar beside
the DBM; build it once:

```sh
python build_track_crossings.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm \
    --z-min 4000 --z-max 17000 \
    --temp-dir /fast/disk/tmp
```

The optional Z range is half-open (`[z-min, z-max)`) and retains only tracks
entirely contained in that range. Omit both options to index the whole DBM.
The standalone builder uses a hybrid memory/disk index: it streams DBM tracks
into temporary coordinate and packed-voxel files, keeps the coordinates
memory-mapped, then loads and radix-sorts the packed keys in RAM. The native
`vc_spiral.track_crossings` kernel uses all requested workers for sorting,
exact-voxel discovery, arclength calculation, and pair consolidation. The
extension is built with the other Spiral native modules by `uv sync` from this
directory. A slower Python fallback remains available.

The builder needs roughly 20 bytes of temporary disk space per selected point.
The native radix sort temporarily holds about 32 RAM bytes per point; after the
sort, those arrays are released before the 8-byte-per-point arclength vector and
compact 16-byte crossing events are consolidated. This avoids retaining either
the selected track database or Python dictionaries of crossing pairs in RAM.
Temporary files are removed after the sidecar is written. Without `--temp-dir`,
the temporary workspace is created beside the tracks DBM rather than under the
system temporary directory.

The script writes
`/data/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz` atomically. The
sidecar includes a fingerprint of every DBM backing file, so a stale one is
rebuilt rather than reused. Re-run the builder after changing the DBM
(`--force` replaces a current cache); building another range replaces it.

## Converting track DBMs to OME-Zarr

`tracks_to_ome_zarr.py` rasterizes the ZYX polylines produced by
`extract_surface_tracks.py` into a compressed `uint8` OME-Zarr. Value 0 is
background; values 1–255 are assigned with proximity-aware reuse and display
as categorical colors with VC3D's Glasbey colormap. Rasterization uses worker
processes, while independent Zarr chunks are compressed and written by a
thread pool using Zstandard level 3.

Use a paired OME-Zarr to copy the exact volume shape and physical geometry:

```sh
python tracks_to_ome_zarr.py \
    /data/tracks/2um_ds2_ps256_surf_v2.dbm \
    --out /data/tracks/2um_ds2_ps256_tracks.ome.zarr \
    --like /data/volumes/2um.ome.zarr \
    --like-group 0
```

Alternatively pass `--shape Z,Y,X`. If neither `--shape` nor `--like` is
given, the script first scans the DBM and uses the maximum track coordinate
plus one. The explicit forms avoid that extra pass for large databases.
`--resume` continues an interrupted conversion. Multiple positional DBMs are
combined into one output, so separate scrolls should be converted in separate
commands.

## Neural winding-inference losses

Set `dense_spacing_mode` to `winding_model` and provide the compact exported
crossing directory: the input manifest's `winding_inference` for a
command-line fit, the service's `--winding-inference` binding for a service
fit (read directly from that host path; never through the cache). Two vocabularies
deliberately coexist: `winding_model` names the fitting mode and its tunables
(`sample_count_winding_model_*`, `winding_model_relative_pair_delta`,
`winding_model_huber_delta`, the `dense_spacing_winding_model_*` losses),
while `winding_inference` names the exported artifact and everything tied to
its on-disk identity (the input path, the `winding_inference_crossings`
artifact type, and the checkpoint fingerprint field). The store is
checksum-verified and copied to each fitting GPU at startup; rays whose
crossings cannot intersect the configured z-range are excluded from sampling,
and optimisation then does no inference-store filesystem I/O. The default
24,000 samples per step are split evenly between long relative-winding pairs
(`sample_count_winding_model_relative_pairs`, index separation drawn from
`winding_model_relative_pair_delta`) and adjacent-passage density pairs
(`sample_count_winding_model_density_pairs`). The independent Lasagna normal
and native minimum-spacing losses remain available alongside it.

The compact store is created by the Vesuvius winding-model
`export_spiral_supervision.py` tool; see its `NATIVE_PHASE_CACHE.md` for the
exact export command and format.

For a headless fit, name the artifact in the input manifest (`--inputs`) and
select inference mode (plus any independently disabled losses) through
`FIT_SPIRAL_CONFIG_OVERRIDES`. The declarative input catalog decides which
named inputs the configuration requires; a required one the manifest does
not name, or that is missing, is reported before the fit starts.

To follow a headless fit while it runs, set `FIT_SPIRAL_METRICS_HISTORY` to a
file path: every 200 iterations the fit appends one JSON line,
`{"iteration": N, "metrics": {...}}`, whose metrics hold `total_loss`, each
`<term>_loss` and the other logged values (what it also prints, and what
`runners/run_single.py` records for sweeps). For example:

```bash
FIT_SPIRAL_METRICS_HISTORY=/tmp/fit_metrics.jsonl \
  ./.venv/bin/python fit_spiral.py --inputs spiral-inputs.json --object-store dataset.vcobj > fit.log 2>&1 &
tail -f /tmp/fit_metrics.jsonl
```

## Fiber direction samples

The optional fiber-direction loss consumes one packed artifact extracted from a
Lasagna fiber prediction: a remote manifest URL, or a local manifest whose
arrays are Zarr v2 directories (VC3D's local copy of an open-data manifest,
beside its `lasagna-remote.json` marker, reads its remote arrays). Extraction
reads only chunks intersecting the requested z ROI and keeps the
highest-presence voxel in each fixed prediction-space cell:

```bash
./.venv/bin/python fiber_direction_samples.py \
  https://example/fibers.lasagna.json \
  /path/to/fiber_directions.npz \
  --z-roi 10000,11000 --output-scale 4 \
  --presence-threshold 160 --cell-size 2
```

The z ROI is half-open and expressed in the output/fitter coordinate system;
`--output-scale 4` means one fitter `/2` coordinate is four base `/0` voxels. The
extractor always covers the fiber volume's complete XY extent.

A fit refuses a z window outside the range an artifact was extracted over
(its recorded `z_roi`).

A service reads no `fiber_directions.npz` path: it reads the samples of the
project's selected fiber-inference dataset, by that dataset's identity (see
"Lasagna inputs must be packed first") and the fit volume's scale, from its
cache: `<cache>/fiber-directions/<identity>/scale<output scale>.npz`. A service
launched with `--build-missing-lasagna-packs` (as VC3D's owned local service
is) extracts missing ones there itself during input preparation, over the
prediction's whole height, logging its progress. For any other service,
extract them over the whole height (no `--z-roi`), or place an existing
artifact there, and copy the file to the service's cache:

```bash
./.venv/bin/python fiber_direction_samples.py \
  https://example/fibers.lasagna.json --cache ~/.cache/vc3d/spiral --output-scale 4
./.venv/bin/python fiber_direction_samples.py \
  --adopt /path/to/fiber_directions.npz --cache ~/.cache/vc3d/spiral
```

A build whose loss needs them, on a service that does not extract them,
refuses a cache without them before linking.

Both `input_use_fiber_directions` and `loss_weight_fiber_directions` default
to off/zero; set the toggle true and the weight above zero to enable the
loss. The command-line fitter then loads the artifact its input manifest
names as `fiber_directions` (a service, the cached one) and samples
`sample_count_fiber_direction_points` observations per step.
Positions and directions constrain only local fitted-sheet orientation; they
do not attach a sample to a particular winding.

## Fiber classification

Each fiber strip is tagged at load time (`spiral_helpers.classify_fiber_hv`):
VC3D's manual H/V tag wins, then its automatic tag when the recorded certainty
reaches `pcl_vertical_fiber_min_auto_certainty`, then a geometric fallback that
calls a strip vertical when its z extent is at least
`pcl_vertical_fiber_min_z_fraction` of its path length. The fit log reports the
vertical / horizontal / untagged split.

## Vertical-fiber radial offset

Papyrus carries its horizontal fibers on the front face of the sheet (the face
toward the umbilicus and the lower-winding neighbour) and its vertical fibers
on the back face, so the two fiber classes sit a few voxels apart along the
sheet normal and at most one of them can lie on the face the fit targets. The
strip losses read every fiber as lying *on* the fitted winding, which leaves a
permanent residual on vertical strips and on every vertical-to-horizontal link
junction. With `pcl_vertical_fiber_radial_offset_enabled` set, vertical strips
(as classified by `spiral_helpers.classify_fiber_hv`, see above) carry a
per-point target offset of `pcl_vertical_fiber_radial_offset_voxels` (default
4, positive = outward) along the sheet normal of the fitted spiral (the
scan-space gradient of the fitted winding, not the straight line to the
umbilicus): the radius loss, the whole-strip DT target, the DT snap, and the
satisfaction metric all expect those points that far outside the winding, on
its back face, instead of on it. Horizontal strips and regular point
collections are never offset.

The offset is a physical scroll-space distance, not a spiral-space constant.
In spiral space the fitted sheet's normal is the radial direction, so the
offset acts on the shifted radius, but the scroll-to-spiral map is not an
isometry (the gap expander rescales the radial coordinate per gap and the flow
stretches locally), so each point's offset is multiplied by the transform's
local stretch along the sheet normal, `|J^T n|`
(`sample_spiral.get_radial_normal_stretch`; the same `J^T` covector transport
the dense-normals loss uses). `J^T n` is the scan-space gradient of the fitted
winding, so a positive offset displaces the expected fiber position along it:
the increasing-winding direction of the fitted spiral at that point, away from
the umbilicus, which coincides with the line from the umbilicus only where the
sheet happens to be perpendicular to it. The loss evaluates that stretch for its sampled
points each step under `no_grad`, the DT target cache and the satisfaction
metric for the points they read. After a constraint bake the resident geometry
lives in the frozen stack's output frame, so each vertical strip and fiber
catalog point also carries `radial_offset_bake_scale`, the product of every
frozen epoch's normal stretch at the pre-bake point
(`fit_spiral.accumulate_radial_offset_bake_scale`); ingested inputs pick it up
when they are pushed through the stack, re-materialised strips inherit it from
the catalog, and the final scroll-space export drops it because the composed
transform's Jacobian then carries the whole stretch. The offset travels with
the strip bundle (`radial_offsets`, resident-frame voxels after the bake
scale) next to the winding annotations, so linked components mixing both fiber
classes read as one winding. Unlike the other `pcl_` settings, both keys are
Run-scoped: a Run that changes them refills every retained strip's offsets from
its stored vertical/horizontal tag and rebuilds the strip bundle and DT target
caches, so no fit rebuild is needed. VC3D's spiral panel exposes them as a
checkbox and a voxel distance next to the fibers path.
