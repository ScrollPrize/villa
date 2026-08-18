# `las_manager`

`las_manager` is the shared orchestration CLI for Fiber 3D and Lasagna
inference. It provides global configuration, open-data and snapshot discovery,
prefetch, durable Fiber and Lasagna inference runs in tmux, portable inference
provenance, and atomic Atlas staging/ingestion for both portable bundle forms.

## Configuration

Initialize `${XDG_CONFIG_HOME:-~/.config}/las_manager/config.toml`:

```bash
las_manager config init
las_manager config show
```

`LAS_MANAGER_CONFIG` overrides the location for automation. Relative paths are
resolved relative to the config file. The initial file deliberately leaves
`snapshot_dirs`, `cache_dir`, `output_dir`, `venv`, `atlas_dir`, and
`upload_staging_s3` empty. Commands validate only the values they need. AWS
credentials and profiles are external and must not be written to this file.

Configure `snapshot_dirs` with any combination of a run collection, one
TensorBoard run directory, or a `snapshots/` directory:

```toml
snapshot_dirs = ["/ephemeral/me/fiber/runs"]
cache_dir = "/ephemeral/me/las_manager_cache"
output_dir = "/ephemeral/me/fiber_inferences"
venv = "/home/me/.venv_las"
params = ["--tile-size", "512", "--border", "32", "--overlap", "96", "--devices", "all"]
```

`params` is a TOML array of backend argv tokens applied to every Fiber and
Lasagna inference. Arguments supplied after `inference run ... --` follow these
defaults and override repeated options. An explicit `--device` or `--devices`
also removes the configured mutually exclusive device selector.

## Catalog and volume selectors

```bash
las_manager fetch
las_manager volume ls
las_manager volume ls --sample PHerc0332 --format uint8
```

Human output is an aligned table grouped by scroll. The scroll is printed once
and `├─`/`└─` branches identify its volumes. `PREFETCHED` lists numeric OME
groups that already contain local chunk data in the manager cache:

```text
SCROLL     VOLUME                                             SHAPE               VOXEL    FORMAT  PREFETCHED  ORIGINS
---------  -------------------------------------------------  ------------------  -------  ------  ----------  -------
PHerc0125  20250821151825-9.362um-1.2m-113keV-masked.zarr      20840x 8387x 8387  9.362um  uint8   1,2         s3
```

Metadata-only groups are not reported as prefetched. Use `volume ls --json`
for scripts and other machine consumers; its schema is independent of the
human table.
The catalog's depth/height/width shape is space-padded to widths 6/5/5 so each
component aligns vertically. The separate volume ID is omitted from the human
table because it is already the prefix of the long volume name.

The raw catalog and validation sidecar live under `<cache_dir>/catalog`.
`fetch` always revalidates. Volume commands refresh a missing or hour-old cache
using ETag/Last-Modified when available; a malformed refresh never replaces a
valid cache, and a failed refresh falls back with a warning. Each indexed
record retains its sample/volume IDs, full catalog entry, license, every OME
origin/access root, selected public S3 origin, catalog hash, validators, and
fetch timestamp for later provenance and Atlas ingestion.

Stable volume selectors are `sample_id/long_id`, a globally unique `long_id`,
or a globally unique volume ID. Exact matching wins; otherwise a unique prefix
is accepted and ambiguous matches are printed as an error.

## Snapshot index

```bash
las_manager snapshot ls
las_manager sn l --backend fiber3d
```

The first listing inspects checkpoints with `torch.load(..., mmap=True,
weights_only=True)` and computes their SHA-256. Extracted metadata is cached by
canonical path, byte size, and nanosecond mtime. Subsequent unchanged listings
do not reopen or rehash the checkpoint. Output includes a display-only ordinal,
stable `backend/run/checkpoint` selector, training step/test metric, patch
shape, precision policy, optional Atlas model ID, and hash prefix. Missing
legacy metadata is displayed as `-`; unsafe pickle fallback is never used.

Snapshot records are backend-neutral and preserve candidate Atlas
model fields (task, architecture, patch/output schema, training/checkpoint
identity, precision, code revision, and optional Atlas model ID). Fiber and
Lasagna checkpoints are detected from checkpoint structure rather than their
filenames and have distinct `fiber3d/...` and `lasagna/...` selectors. Use
`--backend` only to disambiguate a legacy shorthand that matches both.

## Prefetch and inference

Download exactly one OME-Zarr group into the configured cache:

```bash
las_manager volume prefetch PHerc0332/20260411134726-2.400um-0.2m-78keV-masked.zarr 1 --workers 512
```

The OME root is stored at
`<cache_dir>/volumes/<sample>/<long_id>/`; inference reads its numbered group.
Downloader `_download` metadata is retained, and the existing Lasagna
downloader performs all listing, resume, and transfer work.

Launch either backend with a stable snapshot and volume selector:

```bash
las_manager inference run fiber3d/my-run/best.pt PHerc0332/20260411134726-2.400um-0.2m-78keV-masked.zarr 1 --download-workers 512 -- --devices all
las_manager inference run lasagna/cos-run/model_best.pt PHerc0332/20260411134726-2.400um-0.2m-78keV-masked.zarr 1 --download-workers 512 -- --devices all
las_manager inference ls
las_manager run ls
las_manager tmux attach fiber-PHerc0332
```

The command reserves a concise run directory, launches tmux, prints the path,
and returns immediately. Inside tmux, prefetch completes before the GPU child
starts, so downloader activity cannot collide with inference workers. Follow it
with `las_manager run ls`, `las_manager tmux attach <run>`, or inspect
`<run>/run.log`. `--no-prefetch` skips this up-front phase, initializes only the
local source descriptor, and leaves the backend's crop-aware on-demand
downloading enabled during inference. `--download-workers` applies to either
mode. An explicit backend `--no-download` after `--` still wins. Arguments
after `--` are passed unchanged to the selected backend.

For a full volume that cannot remain entirely in the cache, replace the bulk
prefetch phase with the shared rolling selected-scale cache:

```bash
las_manager inference run fiber3d/my-run/best.pt \
  PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr 0 \
  --live-fetch --download-workers 512
```

This initializes only the remote source descriptor, then launches inference
immediately. Defaults are `--live-cache-gib 10240` and
`--live-fetch-ahead-tiles 10000`. Completed selected-scale chunks count toward
the target; only complete old Z-chunk planes behind the committed inference
frontier may be removed. The settings and final cache counters are copied from
portable inference provenance into manager run metadata. `--live-fetch` is
full-volume only and rejects backend/configured `--no-download`.
This includes output-format overrides such as `--ome-compressor none`; newly
created outputs otherwise use the shared Blosc/Zstd default. Resumed arrays
always retain their persisted compressor, and `inference.json` inventories the
actual compressor of every generated level.
The configured venv is used via its absolute `bin/python`; no interactive
activation or `PYTHONPATH` is needed after the documented monorepo install.

Current Fiber snapshots embed the authoritative training/inference config and
the manager does not extract a second runtime config. For a legacy checkpoint,
use `--legacy-config /path/to/config.json`. Direct Fiber invocation follows the
same convention: omit the positional config for a current checkpoint; provide
it only for a legacy checkpoint.

Lasagna runs invoke `preprocess_cos_omezarr predict3d` with the selected
checkpoint. Its direct CLI writes the same portable `inference.json` envelope
as Fiber, with `artifact_kind = "lasagna"`. The product section preserves the
manifest's source-to-base mapping, gradient encoding scale/factor, crops,
channel groups, Zarr paths, and output scaledowns.

Each launch atomically reserves a human label such as
`PHerc0332-20251211183505-las-sd1-362b6b59`. This is not the Atlas canonical
identity: complete volume, model, backend, source group, command, timestamp,
revision, and UUID identity remain in structured metadata. The layout is:

```text
<output_dir>/<run-name>/
  metadata.json
  command.json
  provenance_context.json
  run.log
  artifacts/
    <run-name>.lasagna.json
    inference.json
    ... generated OME-Zarrs ...
```

`metadata.json` carries the immutable UUID, complete source and checkpoint
identity, separate prefetch/inference/upload/Atlas lifecycle states, and private
host details. `command.json` records the original and exact resolved argv plus
the versioned prefetch request. The tmux wrapper tees inference output byte for
byte to both `run.log` and its pane (including carriage-return progress), while
prefetch output is logged before the child starts. It preserves the child exit code,
and atomically records `created`, `running`, `completed`, `failed`, or
`interrupted`. `inference ls` reconciles stale active records without deleting
artifacts. A zero child exit is accepted as completed only when
`artifacts/inference.json` is valid and itself reports `completed`; otherwise
the record is failed with `completion_error`. Only `artifacts/` is intended to become the portable upload bundle;
host paths, logs, and tmux data remain outside it.

Fiber inference itself writes `artifacts/inference.json`, including the source
OME group and observed scale relationship, effective output levels and crop,
tile settings, checkpoint/config hashes and metadata, runtime/repository
identity, and a structural inventory of every generated OME-Zarr level. The
inventory hashes only bounded metadata/root files; it never walks millions of
chunks. Paths in this document are bundle-relative. The `.lasagna.json`
manifest links it as `provenance: inference.json`, and manifest load/save keeps
that link plus unknown forward-compatible fields.

Outside tmux, `tmux attach` attaches normally. Inside tmux, it links the run
window immediately after the current window and selects it, avoiding nested
tmux sessions. New runs store a stable tmux `window_id` tagged with their run
UUID, so linking and window renumbering do not invalidate attachment. An
inference child can outlive a lost tmux wrapper; such a run remains visible in
the durable inference list and log but correctly reports that no terminal can
be reattached. Window tabs use the short `inf-<scroll>-<uuid4>` form rather
than the full run name.

## Abbreviations and completion

Command tokens accept only exact or unique-prefix matches. Entity selectors
follow the same rule but never use fuzzy matching. For Bash, install the
standard per-user lazy-loaded completion once from each venv that provides
`las_manager`:

```bash
las_manager completion install
```

This writes the canonical loader to
`${XDG_DATA_HOME:-~/.local/share}/bash-completion/completions/las_manager` and
keeps additive providers under the adjacent `las_manager` user-data tree. The
loader resolves the external `las_manager` selected by the current `PATH` and
dispatches only to that exact registered executable. Activating another
registered venv switches completion automatically; deleted venv providers are
ignored. Open a new shell after installation, or source the printed loader
path once in the current shell.

For temporary Bash setup or for Zsh, generate shell setup directly:

```bash
eval "$(las_manager completion bash)"
# or: eval "$(las_manager completion zsh)"
```

Both Bash and Zsh dynamically complete cached snapshots, cached catalog
volumes, durable inferences, and live tmux runs. Completion uses read-only
cache endpoints. It does not refresh the catalog, open an uncached checkpoint,
download data, reconcile records, or otherwise modify run state.

Completion also understands unique command abbreviations, command-specific
flags, backend values, catalog sample IDs and formats, and positional volume,
snapshot, inference, and run selectors. Scale completion is exact and
network-free: after any part of a volume has been prefetched, it reads the
local OME `.zattrs` dataset paths and downloaded numeric groups. Before local
OME metadata exists, no scale is proposed rather than guessing remote levels.

A final literal `help` requests help for the longest command prefix the manager
understands:

```bash
las_manager volume help
las_manager vol pre help
```

Arguments following `--` belong to the inference backend; a trailing `help`
there is forwarded unchanged.

## Atlas validation and staging upload

Configure a local Atlas checkout and private staging prefix:

```toml
atlas_dir = "/home/me/vesuvius-atlas"
upload_staging_s3 = "s3://private-staging/my-inferences"
rclone_params = ["--s3-provider", "AWS", "--s3-env-auth", "--transfers", "512", "--buffer-size", "2M", "--size-only", "--fast-list", "-P", "--stats-one-line"]
```

### End-to-end inference and publication workflow

Run these commands in order:

1. Start inference and wait for it to report `completed`:

   ```bash
   las_manager inference run <snapshot> <sample/volume> <scale>
   las_manager inference ls
   ```

   The manager creates
   `<output_dir>/<run-name>/`. Portable results are written below its
   `artifacts/` directory; logs and host-specific metadata stay beside that
   directory and are not uploaded.

2. Validate the completed bundle without writing anything:

   ```bash
   las_manager open-data validate <run-name>
   ```

3. Stage the bytes and register them in the configured local Atlas checkout:

   ```bash
   las_manager open-data upload <run-name>
   ```

   This copies `artifacts/` to
   `<upload_staging_s3>/<run-uuid>/` (for example,
   `s3://philodemos/hendrik/lasagna/inference/<run-uuid>/`) and writes or
   updates the corresponding files below `<atlas_dir>/data/models/lasagna/`
   and `<atlas_dir>/data/volumes/`. It does **not** publish to the public
   open-data bucket.

4. Review, commit, push, and merge the generated Atlas metadata through the
   normal Atlas pull-request workflow:

   ```bash
   cd <atlas_dir>
   git diff
   git add data/models data/volumes
   git commit -m "Register Lasagna inference"
   git push
   ```

5. After that metadata is merged, an Atlas operator plans and executes the
   existing copy-first publication workflow:

   ```bash
   cd <atlas_dir>/vesuvius-atlas-py
   uv run vesuvius-atlas data-sync plan --sample-id <sample>
   uv run vesuvius-atlas data-sync execute --sample-id <sample>
   uv run vesuvius-atlas data-sync status --poll
   ```

   Data sync copies the staged prediction into the canonical public open-data
   storage and records its `public-read` origin. The Atlas catalogue export
   then exposes that public origin online; the private staging origin is not
   included in the public metadata.

Validate or upload a completed Fiber or Lasagna run:

```bash
las_manager open-data validate <run>
las_manager open-data upload <run>
```

The manager resolves model identity from the inference checkpoint SHA-256 and
configured `snapshot_dirs`, freshly rehashing candidates before use. A carried
numeric Atlas ID is honored; otherwise the actual run's single UTC timestamp
defines the ID. Byte-identical checkpoint aliases are normalized, while hash,
run, checkpoint, timestamp, and Atlas collisions fail before staging. Missing
models are registered automatically.

The registered model follows the existing minimal Lasagna convention: numeric
data references, `architecture = "fiber3d/unet"`, `task = "lasagna"`,
`creation.process = "model_training"`, the checkpoint path relative to a
configured snapshot root, and `snapshot_sha256`. The checked-out Villa source
commit is recorded only in portable `inference.code_commit`, not in Atlas model
metadata; `repository.dirty` reports local modifications. Packaged deployments
outside a Git checkout may supply `VILLA_CODE_COMMIT` and otherwise record
`null`.

Upload writes `_INCOMPLETE` at the final run-UUID prefix, bulk-copies the fixed
file inventory with `rclone`, and removes the marker only after rclone succeeds.
Retry invokes rclone again so its configured comparison flags resume or skip
existing objects. Completed run UUIDs and their artifact bundles are immutable:
`rclone copy --size-only` does not detect changed same-size objects or remove
stale destination objects.

`rclone_params` is passed verbatim to `rclone copy`; edit the array to tune
transfer concurrency, buffering, progress output, or S3 authentication for the
host. The default is optimized for many small Zarr objects and requires
`rclone` on `PATH` plus AWS credentials in the environment.

Atlas ingests both Fiber and Lasagna output as the existing copy-first
`lasagna` entry with identity `(volume, model, input level)`. Portable
provenance remains in `inference.json` and is not duplicated into the Atlas
entry's `creation_info`. Atlas stores the private source as an access-root URL
plus a relative origin path and joins them for data-sync; public metadata export
keeps only the subsequently added `public-read` origin. Publication remains an explicit
`vesuvius-atlas data-sync` operation; the manager never writes the public
bucket and leaves `atlas_publication = not_started`.

### Refinement direction: reuse generic Atlas ingestion

Atlas already provides the `lasagna` data type, its canonical destination and
validation rules, copy-first `data-sync` publication, model ingestion, and the
generic `ingest data` path. The current Atlas-side `inference_bundle.py`
adapter adds convenience around the manager's portable `inference.json`: it
validates the bundle, derives the sample/volume/model/level identity, verifies
the licence and snapshot identity, creates a missing model, and registers the
existing `lasagna` data entry.

This adapter is not fundamentally required for publication. A preferred future
refinement is to keep interpretation of manager-specific `inference.json` in
`las_manager`, generate the minimal model and data-registration inputs there,
and invoke shared generic Atlas registration APIs. Once that path preserves the
same validation and collision guarantees, remove the dedicated Atlas
`inference_bundle.py` adapter and its tests. Atlas should retain only generally
useful Lasagna support, including Fiber's three-or-four-product bundle
validation, required `nx`/`ny` products, recursive architecture paths such as
`fiber3d/unet`, and any approved snapshot identity fields.
