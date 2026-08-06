# Plan: Lasagna inference manager CLI

## Scope and compatibility

This task implements the complete ordered sequence: Fiber 3D management,
portable inference provenance, shared output behavior, Atlas integration and
upload, and the Lasagna inference backend. No phase is lower priority or out of
scope merely because it comes later in dependency order. Persisted
manager/provenance documents are schema-versioned and extensible; Atlas schema
evolution may use an explicit migration. The manager orchestrates existing
Python APIs/CLIs; it does not fork downloader, inference, pyramid, or manifest
logic.

Command-prefix abbreviations apply only when a token uniquely identifies a
command at its current level (`sn ls`, `inf r`, etc.). Entity selectors use
exact match first and then unique prefix; ambiguity is an error that prints all
matches. Destructive or expensive actions are never selected by fuzzy match.

## Proposed delivery phases

### Phase 1 — shared foundations and read-only discovery

Implement packaging/dispatch, global config, catalog caching and volume
selectors, checkpoint-driven snapshot indexing, abbreviations, and read-only
completion. This provides `config`, `fetch`, `snapshot ls`, and `volume ls`
without starting work or mutating remote state.

Atlas/Lasagna requirements carried here:

- Preserve the complete catalog identity needed by Atlas: sample and volume
  IDs/long ID, license, original `DataEntry` origins/access roots, selected OME
  origin, raw catalog SHA-256, validators and fetch timestamp. Do not reduce a
  volume to only a download URL.
- Make snapshot records backend-neutral and namespace stable selectors by
  backend/run/checkpoint. Extract common model metadata plus backend-specific
  fields without assuming every checkpoint is Fiber. Include optional
  `atlas_model_id`; absence remains visible and must be resolved before upload.
  Preserve the candidate Atlas Model fields now: training/model creation UTC,
  process, task, architecture, patch shape, output schema/options, training run,
  checkpoint step/metric/hash, precision policy and code revision.
- Initialize Atlas checkout and staging-S3 config now, and design completion/
  command registration so `open-data` and Lasagna backend selectors plug into
  the same tree. Standard AWS credential/profile environment remains external
  and is never written into config or metadata.

### Phase 2 — prefetch and durable run orchestration

Add downloader reuse, `volume prefetch <volume> <scale>`, immutable run records,
Fiber command construction, tmux lifecycle, `inference ls`, `run ls`, and
contextual `tmux attach`. Validate first with fake tmux/downloader/inference
processes.

Atlas/Lasagna requirements carried here:

- Use a backend-neutral run schema with immutable run UUID, backend/artifact
  kind, source volume and level, snapshot/model identity, portable provenance
  path, artifact inventory, status/timestamps, and upload/Atlas state. Fiber
  and Lasagna must occupy the same run index without path or selector clashes.
- Make `artifacts/` a self-contained upload bundle: every path inside manifests
  and provenance is relative to the bundle, while logs/private host details stay
  one directory above it. The Atlas upload phase copies only `artifacts/`.
- Separate lifecycle dimensions: inference status, staging-upload status,
  Atlas-ingest status, and Atlas-publication status. Earlier phases may leave
  the latter three `not_started`, but the schema and state transitions exist
  from the first created run.

### Phase 3 — checkpoint-driven Fiber inference and provenance

Remove the need for a separate Fiber config by loading the authoritative config
embedded in current checkpoints. Retain explicit config only for legacy
checkpoints without one. Add direct `inference.json`, exact scale bookkeeping,
artifact inventory, redaction, and manager-to-inference provenance context.

Atlas/Lasagna requirements carried here:

- Define one versioned, backend-neutral inference provenance envelope and a
  product-specific section. Fiber is its first producer; Lasagna phase 7 must
  use the same writer/schema rather than translate manager logs afterward.
- Record all Atlas identity inputs at creation time: artifact kind, source
  volume/level, license/catalog digest, model/checkpoint identity, generated
  UTC time, effective inference settings/scales, repository revision, and
  structural artifact inventory. Mark whether `atlas_model_id` is asserted by
  trusted checkpoint metadata, supplied later, or unresolved.
- Keep exact source scale semantics generic: selected OME group, observed
  source-to-base factor, backend inference-output factor, effective output
  levels and crop. This supports Fiber's factor-4 default and Lasagna's own
  output products without conflating either with the input group.

### Phase 4 — shared output-format change

Change newly created Fiber and Lasagna inference OME-Zarrs to the shared exact
Blosc-Zstd default, preserve existing compressors on resume, add an override,
and validate every generated/pyramid level through zarr and TensorStore. This
is isolated because it changes persisted output bytes.

Atlas/Lasagna requirements carried here:

- The provenance inventory reads the actual codec metadata from every output
  level; it never merely repeats the requested default. Atlas validation can
  therefore verify what was persisted.
- Change the shared OME group creator used by both backends, and test both
  product layouts. No Fiber-only compressor option or manager-only rewrite is
  permitted.
- Resuming an older artifact preserves its existing compressor and records it;
  upload validation distinguishes a valid legacy bundle from a newly created
  bundle that violates the current default.

### Phase 5 — integration, documentation, and bounded real run

Finish dynamic completion, state reconciliation and failure handling; run the
unit/integration suites; perform a small real Fiber inference; and update specs,
docs, README, changelog, status, and task log.

Atlas/Lasagna requirements carried here:

- Validate portable Fiber provenance against a fixture built from the checked
  out Atlas Pydantic `DataEntry` models before phase 6 starts changing Atlas.
- Add a synthetic second backend fixture so generic run/provenance/completion
  code is proven not to rely on Fiber-only fields before Lasagna arrives.
- Test that copied `artifacts/` remains self-contained after moving it to a new
  directory, because staging upload and Atlas canonical copying change roots.

### Phase 6 — Atlas integration and upload

In Atlas reuse the existing Lasagna model/artifact type for Fiber output, add
the provenance parser and browser registration, then add
`las_manager open-data upload`: atomic staging-S3 upload, Atlas ingest, and
operator-controlled Atlas data-sync publication.

Detailed completion criteria:

- In the Atlas repository store Fiber and Lasagna inference as the existing
  `lasagna` copy-first volume artifact/model task, with canonical identity
  `(volume_id, model_id, input level)`, portable validation/browser support,
  and the existing CC BY-NC publication gate.
- Add an Atlas ingestion parser that consumes the portable envelope and writes
  `DataEntry.parameters` and a safe `creation_info` subset without local paths,
  tmux data, credentials, or arbitrary unvalidated metadata. Register/create
  the Atlas Model explicitly from checkpoint metadata when requested, or
  require an existing `--model-id`; never derive model identity from a filename.
- When explicitly registering a model, construct a normal Atlas `Model` using
  its UTC datetime ID convention and the carried task/architecture/patch/output/
  training/checkpoint metadata. Present the proposed JSON before mutation and
  refuse an existing-ID metadata mismatch.
- Implement `open-data validate <inference>` and `open-data upload <inference>`.
  Upload validates completion/model/license, creates an `_INCOMPLETE` marker at
  the final run-UUID staging prefix, uploads the bundle, writes a content
  `upload-manifest.json` last, verifies it, removes `_INCOMPLETE` as the commit,
  then ingests it into the configured Atlas checkout. Atlas never sees a
  partially committed prefix. Record every state transition in the run record.
  Public copying remains an explicit Atlas data-sync operation, not a direct
  manager write to the open bucket.
- Make upload idempotent by run UUID plus bundle/provenance digest. Refuse a
  same-identity/different-content collision. Build the detailed path/size/
  checksum upload manifest while traversing/reading local files for the upload;
  inference itself still does not hash every chunk. On retry, compare the local
  digest to the committed manifest object directly—never recursively list a
  remote Zarr.
- Test with a fake S3/Atlas command layer, then one staging target if authorized;
  validate the modified Atlas repository with its own tests and dry-run plan.

### Phase 7 — Lasagna manager backend

Add Lasagna snapshot discovery and launch arguments while reusing the same
catalog, cache, run, tmux, completion, provenance, artifact, and Atlas upload
infrastructure.

Detailed completion criteria:

- Detect/index Lasagna checkpoints separately from Fiber checkpoints, expose
  their training step/metric/architecture/patch metadata and optional Atlas
  model ID, and retain explicit `--backend` for genuinely ambiguous legacy
  snapshots. Stable selectors include the backend namespace.
- Implement the Lasagna backend using the existing `lasagna-preprocess
  predict3d` API and the shared download/tiled inference/output helpers. It
  supplies only product-specific arguments and uses the same tmux/run/status/
  completion machinery as Fiber.
- Make direct Lasagna inference emit the same portable provenance envelope with
  `artifact_kind = lasagna`, its four standard channel pyramids and any declared
  optional products. Preserve the existing `.lasagna.json` decoding fields
  (`source_to_base`, gradient encoding/factor, groups/scaledowns/crops) and map
  them explicitly into the provenance product section.
- Reuse the existing Atlas `lasagna` artifact/model task and publication rule,
  but add the same provenance-aware ingestion parser/validation used for Fiber.
  Reconcile its current exactly-four-pyramid validation with any optional
  declared Lasagna products instead of silently dropping or miscounting them.
- Run the same prefetch, manager launch, resume, moved-bundle, staging upload,
  Atlas ingest and publication dry-run tests for Lasagna. Phase 7 is complete
  only when backend-specific code is limited to snapshot interpretation,
  command construction, and product metadata—not duplicated orchestration.

## 1. Package and command architecture

1. Add a `lasagna.manager` package with small modules for config, catalog,
   snapshots, run records, tmux, completion, and inference backends. Register
   `las_manager=lasagna.manager.cli:main` in Lasagna packaging. Keep parsing and
   presentation separate from filesystem/network/tmux operations so all
   behavior is unit-testable.
2. Use one declarative command tree to drive argparse dispatch, unique-prefix
   resolution, help, and completion. Initial public commands:

   ```text
   las_manager config init [--force]
   las_manager config show
   las_manager fetch
   las_manager snapshot ls [--backend fiber3d|lasagna]
   las_manager volume ls [filters]
   las_manager volume prefetch <volume> <scale>
   las_manager inference ls
   las_manager inference run <snapshot> <volume> <scale> [--backend ...] [manager overrides]
   las_manager run ls
   las_manager tmux attach <run>
   las_manager open-data validate <inference>
   las_manager open-data upload <inference> [--model-id ...]
   las_manager completion bash|zsh
   ```

   `volume prefetch <scale>` from the request is made explicit as
   `volume prefetch <volume> <scale>` because a volume is otherwise
   unknowable; shell completion supplies the volume selector.
3. Add a backend protocol and a Fiber implementation. Every record has
   `backend = "fiber3d"`. It resolves the
   checkpoint-embedded config, selected catalog origin/group, artifact manifest
   path, and existing `fiber_trace_3d.infer` arguments. The phase-7 Lasagna backend
   implements this interface instead of adding a second manager workflow. The
   positional `<scale>` is strictly the source OME-Zarr group/index; it does not
   override Fiber's default `--inference-scaledown-power`. Record requested
   group, observed `input_sd`, inference factor, effective base level, and
   produced levels to prevent double application.
   Refactor the Fiber inference API so its config path is optional. Current
   checkpoints use their embedded config for architecture, patch shape,
   normalization, and inference policy. A legacy checkpoint without embedded
   config fails with a clear request for explicit `--config PATH`. Preserve the
   old positional-config invocation as a compatibility form, but the manager
   never stores or requires a global config.

## 2. Global configuration

4. Store config at `${XDG_CONFIG_HOME:-~/.config}/las_manager/config.toml`,
   overridable for tests/automation by `LAS_MANAGER_CONFIG`. Use stdlib
   `tomllib` to read and a small deterministic writer (no new config-library
   dependency). Resolve relative paths relative to the config file and expand
   `~` and environment variables at use time.
5. `config init` atomically creates a commented config with these stable keys:

   ```toml
   catalog_url = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json"
   open_data_bucket = "s3://vesuvius-challenge-open-data"
   snapshot_dirs = []
   cache_dir = ""
   output_dir = ""
   venv = ""
   atlas_dir = ""
   upload_staging_s3 = ""
   catalog_max_age_seconds = 3600
   ```

   `atlas_dir` and `upload_staging_s3` are initialized empty and used in phase 6; the
   public open-data bucket is not an upload staging location. Empty required
   local values remain present as requested and produce a
   precise validation error only for commands that need them. There is no
   global Fiber inference config: current checkpoints embed the authoritative
   config.
   Never overwrite an existing config without `--force`.
6. Validate paths/types centrally and expose the resolved config through
   `config show`, redacting future secret-valued fields. Launch the configured
   venv by absolute `<venv>/bin/python` rather than relying on interactive shell
   activation; record the equivalent activation command for humans.

## 3. Open-data catalog and volumes

7. Fetch the metadata document with HTTP conditional requests when possible
   (`ETag`/`Last-Modified`), a timeout, and atomic replacement. Store raw JSON
    plus a sidecar containing fetch time, SHA-256, validators, staleness, and
    last refresh error under the configured
   cache directory. `fetch` always revalidates; volume-dependent commands
    revalidate only when missing or at least 3600 seconds old. A conditional
    HTTP 304 atomically advances validation time. If refresh fails
   but a valid cache exists, warn and use it; malformed new data never replaces
   a valid cache.
8. Build a deterministic in-memory volume index from `samples.*.volumes` and
   `data[type="ome-zarr"].origins`. Preserve `sample_id`, `id`, `long_id`, shape,
   pixel size, data format, license, origins, and derived S3/HTTPS URLs. List all
   catalog volumes; prefetch/inference select a public S3 origin and clearly
   reject a record with no supported S3 origin in this first version.
9. Accept canonical selectors `sample_id/long_id`, full `long_id`, and volume
   `id` only when globally unique. Unique prefixes are accepted after exact
   matching. `volume ls` prints one stable line per volume with selector,
   sample, dimensions, voxel size, format, and available origin types.
10. `volume prefetch <volume> <scale>` computes
    `<cache_dir>/volumes/<sample_id>/<long_id>/<scale>` and invokes the existing
    `download_omezarr.download(...)` API for exactly that OME group. Add manager
    flags for downloader workers/force-refresh only as thin pass-throughs; do
    not implement a second S3 transfer path. Preserve `_download` metadata so
    the same local volume remains usable by inference auto-download/resume.
    Validate the destination convention against the downloader in tests rather
    than assuming whether it expects the OME root or `<root>/<scale>`.

## 4. Snapshot index

11. Treat each configured snapshot directory as either a run root containing
    `*/snapshots/*.pt`, an individual TensorBoard run containing
    `snapshots/*.pt`, or a snapshots directory itself. Deduplicate resolved
    paths and sort by run name then checkpoint name.
12. Read checkpoint metadata with CPU mapping, mmap where supported, and
    `weights_only=True`; never materialize/copy model state merely to list it.
    Cache extracted metadata by canonical path, size, and mtime. Record at least
    ordinal, stable selector (`run/checkpoint`), step, metric name/value,
    checkpoint timestamp/size and content SHA-256, patch shape, model branch/option count,
    precision policy, and path. Missing legacy metadata renders as `-`, not an
    error. Never fall back to unrestricted pickle for listing. The displayed
    ordinal is deterministic for the current sorted index but display-only;
    `inference run` uses stable names or unique prefixes.

## 5. Durable inference/run records and reproducibility

13. Give each launch an immutable UUID plus a collision-resistant, readable run name derived from
    Fiber, sample/volume, source scale, snapshot run/name, and UTC timestamp.
    Sanitize separately for tmux and filesystem limits. Atomically reserve:

    ```text
    <output_dir>/<run_name>/
      metadata.json
      command.json
      run.log
      artifacts/
        <run_name>.lasagna.json
        <channel>.ome.zarr/
    ```

14. Before tmux launch, write schema-versioned metadata containing status,
    run/tmux identity, timestamps, hostname/user, manager version/git revision,
    original user argv, resolved executable argv, cwd, relevant redacted
    environment, complete resolved catalog record and origin, selected source
    scale, snapshot path/cache identity/content SHA-256 and extracted metadata,
    runtime Fiber config content/fingerprint, Python/package/CUDA versions,
    output settings, and artifact paths. `command.json` stores both argv arrays
    (resolved is authoritative and shell-safe) and a display command.
    Stream stdout/stderr together to `run.log` while retaining the real child
    exit status. Secret-like config/environment fields are never persisted.
15. Run a tiny wrapper process in tmux that updates metadata atomically from
    `created` to `running`, then `completed` or `failed` with PID, start/end
    timestamps, and exit code. Forward SIGINT/SIGTERM to the child process group
    and record `cancelled/interrupted` where possible. Do not infer completion merely from tmux death.
    On manager restart, reconcile stale `created/running` records against tmux
    and PID plus process-start-time identity, marking them `interrupted/unknown` without destroying
    artifacts.
16. `inference ls` reads durable output records and shows created/running/
    completed/failed/interrupted runs. `run ls` is the live view: only records
    whose named tmux session currently exists, with session, status, age,
    volume, snapshot, scale, and log path.

## 6. Tmux behavior

17. Create sessions detached with a manager-owned prefix and one named window;
    fail before reserving a duplicate tmux session. Avoid quoting bugs by
    launching a generated argv-aware Python wrapper rather than interpolating
    user paths into `bash -c`.
18. `tmux attach <run>` behaves contextually: outside tmux it executes normal
    `tmux attach-session`; inside tmux it uses `tmux link-window` to insert the
    run window immediately after the current window (the requested adjacent
    tmux “tab”), then
    selects it. It must not nest tmux. If a linked window name collides, derive
    a unique visible name without renaming the source run window.

## 7. Completion and abbreviations

19. Provide generated Bash and Zsh completion scripts from the command tree.
    Static completion covers commands/options; dynamic completion invokes
    read-only hidden completion endpoints for snapshot, volume, inference, and
    live-run selectors. Cache/index reads are allowed during completion, but it
    never refreshes the network, opens checkpoints uncached, downloads data, or
    mutates run state, keeping double-tab responsive.
20. Document installation in Lasagna, plus one-time shell setup, e.g.
    `eval "$(las_manager completion bash)"` (and a `.bashrc` line). Completion
    returns stable selectors plus descriptions so shells supporting descriptions
    show snapshot metadata and volume dimensions.

## 8. Shared Fiber/Lasagna Zstd default

21. Add a shared OME-Zarr compressor choice to the existing group-creation
    helper, defaulting newly created Fiber and Lasagna inference arrays to
    exact Zarr-v2 codec `{id: blosc, cname: zstd, clevel: 3, shuffle: 1,
    blocksize: 0}`. Both callers must use the same helper; no manager-only
    compression behavior. Existing output arrays retain their encoded
    compressor on resume and a requested mismatch is reported rather than
    silently recreating data. Expose a shared CLI override for compatibility.

## 9. Direct inference provenance and Atlas contract

22. Add a shared `lasagna.inference_provenance` writer used directly by Fiber
    3D inference now and by Lasagna predict3d when that backend is added. Fiber
    writes `<bundle>/inference.json` atomically even when invoked without the
    manager. The manager passes catalog identity and its immutable run UUID by
    explicit arguments/context and records orchestration separately; it does
    not post-hoc invent inference facts.
23. The portable document records:
    - schema/artifact kind (`fiber3d-prediction`), generated time and status;
    - source sample/volume IDs and long ID, exact origin, catalog SHA-256/fetch
      metadata/license, requested OME group, observed input shape, derived
      `input_sd`, base shape and `source_to_base`;
    - requested inference-scaledown power/factor, effective base level,
      produced levels, base-coordinate crop, tile/border/overlap, and all
      effective numerical/output settings;
    - checkpoint SHA-256, run/snapshot, step, metric, architecture, embedded and
      runtime config digests, precision, repository revision/dirty state, and
      package/runtime versions;
    - a manifest-relative artifact inventory with manifest, product/channel
      semantics, OME roots/levels/shapes/chunks/dtype/exact compressor.
24. Do not recursively enumerate/checksum millions of Zarr chunks during
    inference. Hash root files; describe Zarr trees structurally, with optional
    S3 inventory/checksums added during upload. Keep absolute paths,
    hostname/user, tmux identity and full command only in private manager
    records. Portable metadata uses relative artifact paths and redacted
    settings. `LasagnaVolume.load/save` preserves the provenance reference and
    unknown forward-compatible manifest fields across resume.
25. Atlas inspection establishes that `lasagna` is a volume-level copy-first
    folder artifact parameterized by `(model_id, level)` and publicly copied
    only for CC BY-NC volumes. Fiber's Lasagna-style output deliberately reuses
    this artifact and model task; backend/product differences remain in
    portable provenance rather than creating a new Atlas data type.
26. The phase-6 Atlas patch validates portable bundles using `inference.json`
    and registers the existing Lasagna bundle type in the predictions browser.
    Its parser maps Fiber or Lasagna portable provenance to
    `DataEntry.parameters = {model_id, level}` and a safe `creation_info`
    subset. An arbitrary checkpoint filename cannot establish an Atlas model
    identity: upload resolves a model ID from explicit checkpoint metadata or
    config mapping, otherwise requires `--model-id`.
27. The phase-6 `open-data upload` flow is: validate a completed bundle; mark a
    unique final staging prefix incomplete; upload while building the content
    manifest; write/verify that manifest and remove the marker as commit; invoke
    Atlas ingestion on the source volume; then let Atlas data-sync copy to
    canonical/private/public targets. Never recursively list a remote Zarr,
    write directly to the public bucket, or mutate Atlas metadata unless upload
    was explicitly requested.

## 10. Tests and validation

28. Unit-test unique-prefix ambiguity, selector precedence, TOML init/validation,
    atomic writes, catalog schema/index/origin choice/staleness/fallback,
    snapshot-root discovery and legacy/current metadata, sanitized identities,
    run-state transitions/reconciliation, argv preservation, and completion's
    read-only/no-network guarantee.
29. Use fake downloader/inference/tmux executables for deterministic tests of
    prefetch, launch, log capture, exit propagation, inside/outside tmux attach,
    collisions, interrupts, and paths containing spaces. Add parser smoke tests
    for every documented shorthand and error on every ambiguous prefix.
30. Add shared inference tests asserting newly created Fiber and Lasagna Zarr-v2
    arrays use Zstd at every generated/pyramid level, resume preserves compressors, mismatch behavior is clear,
    and generated artifacts remain readable by zarr/TensorStore. Run existing
    Fiber/Lasagna inference unit suites to guard output semantics.
31. Add direct-Fiber provenance tests independent of the manager: exact scale
    bookkeeping (requested group versus input/output scaledowns), catalog digest
    pinning, redaction, failed/interrupted status, resume preservation, and an
    artifact-inventory fixture. Construct and Pydantic-validate the proposed
    Atlas `DataEntry` from that fixture against the checked-out Atlas models.
32. Perform a local end-to-end dry run using one real checkpoint/catalog record
    with fake inference first, then a bounded real crop if available. Record
    exact commands, inputs, versions, and results in `planning/task_log.md`.

## Spec update

Add a “Lasagna manager” section to `planning/specs.md` defining the command
families, exact/unique-prefix rules, XDG config and required keys, one-hour
catalog cache semantics, stable volume/snapshot selectors, downloader reuse,
durable run directory schema and state machine, tmux behavior, provenance,
    completion safety, backend extensibility, shared Zstd default, portable
    inference provenance, exact scale semantics, Atlas mapping, and copy-first
    publication. Define Atlas upload and Lasagna-manager launch as required
    phases 6 and 7 of the same deliverable.

## Docs updates

- Add `lasagna/docs/manager.md` with configuration, catalog/snapshot selectors,
  prefetch/inference examples, output layout, tmux behavior, status recovery,
  completion setup, and troubleshooting.
- Update `lasagna/README.md` installation/CLI sections with `las_manager` and a
  minimal first-run sequence.
- Update Fiber 3D and Lasagna inference docs for Zstd default/override and
  provenance boundaries.
- Document the backend interface and the Vesuvius Atlas upload workflow:
  staging -> Atlas ingest -> data-sync publication, required model ID, new
  Fiber data type, and that credentials are intentionally absent here.

## Changelog update

Add a dated entry for the manager/config/catalog/snapshot/prefetch/tmux Fiber
workflow, portable inference provenance, Atlas contract, and shared
Fiber/Lasagna Zstd default. Record upload, Atlas patch, and Lasagna manager
execution as explicit follow-ups.

## Planning workflow

Reset `planning/task_log.md` when implementation begins and record deviations,
findings, exact validation commands/results, platform limits, and any explicit
deviation. Keep `planning/status.md` current after each phase and
validate specs, docs, changelog, status, and task log before handoff.

## Confirmed interface decisions

- Prefetch syntax is `volume prefetch <volume> <scale>`.
- Inside tmux, attach links the run window immediately after the current window
  rather than splitting a pane or nesting tmux.
- Fiber configuration comes from the snapshot; it is not a global manager
  setting. Explicit config remains only for legacy/debug compatibility.
- Work is divided into the phases above so scope can be revised between phases.
