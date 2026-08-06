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
```

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
SCROLL     VOLUME                                             ID              SHAPE            VOXEL    FORMAT  PREFETCHED  ORIGINS
---------  -------------------------------------------------  --------------  ---------------  -------  ------  ----------  -------
PHerc0125  └─ 20250821151825-9.362um-1.2m-113keV-masked.zarr  20250821151825  20840x8387x8387  9.362um  uint8   1,2         s3
```

Metadata-only groups are not reported as prefetched. Use `volume ls --json`
for scripts and other machine consumers; its schema is independent of the
human table.

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

Prefetch completes before the detached GPU job starts, so downloader activity
cannot collide with inference workers. `--no-prefetch` reuses an already
populated cache. Arguments after `--` are passed unchanged to the selected backend.
This includes output-format overrides such as `--ome-compressor none`; newly
created outputs otherwise use the shared Blosc/Zstd default. Resumed arrays
always retain their persisted compressor, and `inference.json` inventories the
actual compressor of every generated level.
The configured venv is used via its absolute `bin/python`; no interactive
activation is needed.

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

Each launch atomically reserves:

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
identity, separate inference/upload/Atlas lifecycle states, and private host
details. `command.json` records the original and exact resolved argv. The tmux
wrapper streams combined output to `run.log`, preserves the child exit code,
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
tmux sessions.

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
```

Validate or upload a completed Fiber or Lasagna run:

```bash
las_manager open-data validate <run> --model-id 20260806120000
las_manager open-data upload <run> --model-id 20260806120000
```

The model ID must come from trusted checkpoint provenance or `--model-id`; it
is never inferred from a filename. `--register-model` explicitly creates a
normal Atlas model from carried checkpoint metadata. Conflicting IDs and wrong
model tasks are rejected.

Upload hashes files while reading them, writes `_INCOMPLETE` at the final
run-UUID prefix, uploads the bundle, writes and verifies
`upload-manifest.json` last, and removes the marker as the commit. Retry reads
that manifest without recursively listing a remote Zarr and refuses a
same-UUID/different-content collision.

Atlas ingests both Fiber and Lasagna output as the existing copy-first
`lasagna` entry with identity `(volume, model, input level)`. Portable
provenance still records the producing backend. Publication remains an explicit
`vesuvius-atlas data-sync` operation; the manager never writes the public
bucket and leaves `atlas_publication = not_started`.
