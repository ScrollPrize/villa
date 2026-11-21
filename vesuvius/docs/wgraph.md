# `vc_wgraph`: reusable winding-graph stages

For the complete command sequence, brief stage descriptions, and measured full-volume runtimes, see the [workflow quickstart](wgraph-quickstart.md).

`vc_wgraph` exposes the shared importer, extractor and solver stages. It does not embed datasets, host paths or alternative solver implementations. Install the normal `vesuvius` package to receive the console command; an existing development environment can also run `python -m vesuvius.wgraph` or `python vesuvius/scripts/vc_wgraph`.

```sh
# AFV source: import first. For an existing neural-tracer fibers.vcobj, skip this line.
vc_wgraph import import-settings.json -i fibers.afv -o fibers.vcobj
vc_wgraph extract extraction.json -i fibers.vcobj -o constraints.vcobj
vc_wgraph hvb solve-settings.json -i constraints.vcobj -o classes.vcobj
vc_wgraph sign solve-settings.json -i classes.vcobj -o signs.vcobj
vc_wgraph winding solve-settings.json -i signs.vcobj -o windings.vcobj
vc_wgraph uv solve-settings.json -i windings.vcobj -o uv.vcobj
```

Each stage accepts an optional profile name or arbitrary JSON configuration file; omission selects `default`. `-i/--input` and `-o/--output` override the actual input/output store paths. They never silently replace a separately configured report or working directory. JSON paths retain the existing native convention: relative paths are relative to the process's working directory, not the profile's directory.

## Current input and solver support

The `import` subcommand accepts AFV databases through the native parallel/sharded importer. An existing compatible vcobj fiber store, including the neural full-volume tracer's `fibers.vcobj`, starts at `extract`; no AFV conversion, reimport or producer changes are required. Independent Python adapters also accept completed compound stream-tracer tiles and reduced route pages, but those formats are not yet selectable backends of the `import` subcommand. See [accepted inputs](wgraph-quickstart.md#accepted-fiber-inputs) for their entry points and preparation limitations.

The current winding command consumes extracted measurements and saved classification/sign runs, not arbitrary pre-existing `fiber_branch` annotation links. H/V/broken and normal signs use wrapped IRLS; winding uses linear GPU IRLS; UV uses the fixed-assignment optimizer described below. The separate LEMON comparison implementation is not a selectable `vc_wgraph winding` backend. There is no `vc_wgraph compare` subcommand. Existing application reference-evaluation helpers do not change these command-line capabilities.

| Stage | Input configuration key | Output store key | Other outputs |
|---|---|---|---|
| `import` | `source` | `output` | `report`, default `<output-stem>.import.json` |
| `extract` | `fibers` | `database` | `output` report, default `<output-stem>.extraction.json`; `normal_prior_output`, default `<output-stem>.outward-priors.json` |
| `hvb`, `sign`, `winding`, `uv` | `database` | `result_store` | Stage work artifacts under `output_root` |

If not configured, `output_root` is `<output-store>.work`. Native stages reuse the existing workflow supervisor for frozen configurations, progress logs and run timing. Import/extraction JSON retains its normal native settings, including source provenance/coordinate binding and prediction locations. The CLI does not invent these dataset-specific values. If an umbilicus file is configured without its checksum, the native-stage adapter computes the checksum using the existing shared helper.

## Profiles

Use `vc_wgraph profiles` to list names and their source files. Names resolve in the following order:

1. Directories passed with repeatable global `--profile-dir DIRECTORY` options.
2. Directories in `VC_WGRAPH_PROFILE_PATH`, separated by the platform's path separator.
3. `$XDG_CONFIG_HOME/vc_wgraph/profiles` (normally `~/.config/vc_wgraph/profiles`).
4. Packaged profiles.

For example, a local `example.json` in one of these directories is selected by `vc_wgraph extract example -i fibers.vcobj -o constraints.vcobj`. Local profiles and paths need not be checked into the repository. Packaged `default` contains only generic algorithm/transport settings; runtime compute-core selection remains native, and there is no fixed host CPU count.

A JSON file can be a flat native/stage configuration, or a workflow profile combining common and stage-specific settings:

```json
{
  "version": 1,
  "common": {"seed": 0},
  "stages": {
    "import": {"volume_id": "catalog-volume-id", "packed": true},
    "extract": {"coordinate_space": "sample/volume", "normal_manifest": "prediction.zarr", "normal_provenance": "inference.json"},
    "hvb": {},
    "sign": {},
    "winding": {},
    "uv": {}
  }
}
```

Packaged stage defaults are merged first, then the selected common and stage settings, then input/output overrides. Nested JSON objects are merged rather than replacing unrelated keys. Workflow-profile root keys are restricted to `version`, `common` and `stages` to catch misspelled settings instead of ignoring them. Each individual stage still receives its ordinary flattened configuration.

`vc_wgraph extract example -i fibers.vcobj -o constraints.vcobj --print-config` prints resolved settings without accessing volumes, importing GPU solvers or running the stage. The local no-argument approval runner used in experiments is separate from this public CLI; it imposes no limitation on normal input/output selection.

## Output and reuse

Stages produce new stores rather than mutating inputs. Derived result stores reference their chosen upstream stores; classification, sign, winding and UV results can therefore be rerun independently of extraction. The native extraction stage publishes normal-aligned standard measurement data; staging internals are not a public graph format. See [fiber workflows](fiber_workflows.md) for the graph schemas and detailed stage output fields.

The store chain is `import → extract → hvb → sign → winding → uv`. `winding` consumes typed classification and normal-sign runs from its input store references, builds the existing fractional-target L1 graph, and runs the historic linear GPU IRLS solver (15 reweighting rounds, 500 CG iterations, epsilon .05 unless explicitly overridden). It records integer readout, continuous potential, connected component and residual metrics. These are point estimates, not calibrated winding probabilities. No GT labels, spiral reference solution or external solver-source file is required. The measured geometry and umbilicus working-coordinate binding remain required inputs for seam transport.

`uv` consumes a selected saved winding run, fixes those integer assignments, and invokes the existing straight fixed-length UV optimizer. It records evaluated sampled UV positions, widths and fiber assignments in a referenced result store, without copying geometry. Generic `vc_wgraph` UV uses `component_selection: "all"`; explicitly choose `"largest"` to reproduce the historical largest-component experiment. Fibers without perpendicular support are excluded by the existing fixed-length model rather than assigned fallback geometry. Exact packed-path prior preparation uses the measured geometry and interpolation; solver weights and GPU-manager settings can be provided under `parameters` and `gpu_manager` respectively.

Each store-to-store solver stage prepares and optimizes inside one managed process. Native operations return owned memory buffers through the Python API; NumPy views retain their native owners and PyTorch consumes those buffers directly before transferring numeric columns to the GPU. There is no `.npy`, NPZ, mmap, RAM-filesystem or other file-based handoff of solver inputs or results. Parallel native operations filter and remap events, derive support, sample measured geometry and construct targets. Configure `workers` for native preparation separately from GPU optimizer settings. Only authoritative result stores and small configuration/history/timing reports are persisted by these stages.

`graph_cache`, `geometry_cache` and `tensor_cache` are not used for store-to-store stage handoffs. Geometry is read through the native store API into process-owned RAM under the consumed source snapshot. Large working allocations belong to process memory, not files under `/dev/shm` or scratch.

Bulk geometry reads use `CollectionReader.geometry_array_columns` with the configured preparation workers. Sealed or history-enabled inputs permit independent native worker transactions sharing one routing topology; each transaction processes coarse UUID blocks without reopening the collection for every block. History-enabled sources may remain unsealed. Extraction captures and retains the consumed publication, and derived store references continue reading that publication after source edits while the required history remains available. An explicit saved `position` also supports reopening a historical source for parallel reads; an unpinned mutable root instead rejects advancement during a parallel read. Unsealed history-off sources are not valid reusable dependencies because replaced states cannot be retained. Generated H/V axial priors consume these coordinate buffers with the same shared midpoint-quadrature and umbilicus-center kernels as native extraction, returning only the needed per-fiber scalar array. See [store reading](../../vc_object_store/docs/reading.md) for position selection and capability limits.

The batch solver workflow automatically uses `trusted_immutable` geometry reading only when the complete referenced scope is sealed and history-off: this fast path is not an integrity audit. History-enabled scopes use checked reads. If a source has advanced, retained metadata and membership changed since the pin are reconstructed in connection-local RAM; geometry payloads remain referenced in their source store. Historical selection therefore has additional setup cost, not a promise of identical throughput to an immutable import. Generated axial evidence is an in-process typed buffer bound to the consumed graph snapshot, rather than a simulated external feature file requiring repeated full-fiber digest scans. Explicit legacy feature files retain their original provenance validation. Bulk worker startup disables SQLite's global memory accounting by default (`sqlite_memory_statistics: false`); set it to `true` for diagnostic accounting. `sqlite_pagecache_bulk_bytes` remains zero unless explicitly configured. Both are startup options, not changes to the objective or numeric precision.

### Bulk preparation and UV optimization

Winding and UV use the same descriptor-only point-count catalog, typed result columns and packed polyline interpolator. Setup does not read every XYZ coordinate merely to calculate point counts. Run selection is pushed into native extension reads; numerical preparation consumes owned NumPy columns instead of per-fiber dictionaries. A single dense graph-to-catalog index is shared read-only by workers. Endpoint evaluation uses per-fiber native arc searches, not repeated textual UUID joins or searches of the entire packed point array. These temporary solver indices do not change persistent UUID identity, store-reference routing, geometry or extension formats.

The shared graph reader uses `fiber_graph_columns` for native node selection and binary-UUID endpoint joins. Its owned buffers contain dense numeric endpoints and binary edge UUIDs, not repeated textual UUID columns per constraint. Diagnostic endpoint-ID views format only requested rows, and numerical stages do not materialize those views. Persisted vcobj measurement formats are unchanged.

UV's `parameters.uv_optimizer` selects `gauss_newton_cg` (default) or the existing `torch_lbfgs` comparison backend. Gauss–Newton preserves the squared fixed-assignment objective; it does not substitute the winding solver's L1 loss. Its matrix-free CG is shared with wrapped IRLS. The independent linear V block is solved first; horizontal physical anchors and the same learned width parameters use Gauss–Newton with Levenberg damping and actual-objective acceptance. Integer labels, exact straight fiber lengths, phase/Z priors, masks, class balancing and width penalties remain unchanged. No dense event-by-parameter Jacobian is built.

The width preconditioner uses a small exact width-by-width block accumulated from interval-pair moments, not one full graph pass per width. Reductions into the few width bins use independent contiguous partial sums before the final reduction, while node reductions retain direct indexed accumulation. This bounds temporary partial storage to roughly one scalar per 32 contributions and avoids a giant array of group indices, sorting or source copies. All arithmetic remains float64; parallel reduction order can differ, so CUDA results are not promised bitwise identical to the original atomic accumulation.

The CG controls are `uv_cg_iterations` (100), `uv_cg_tolerance` (`1e-6`), `uv_v_cg_iterations` (1000) and `uv_v_cg_tolerance` (defaults to the CG tolerance). `steps` remains an outer-iteration cap, not a convergence claim. `uv_lm_damping` defaults to `1e-3`, and five outer iterations below the `uv_tolerance` improvement threshold (default `1e-9`) terminate the solve. The GN backend requires a positive width parameterization; unbounded `direct` widths fail explicitly. All numeric graph columns are transferred once to the solver device, so memory scales with the complete graph rather than the old host batch size; measure VRAM before large runs. Reports retain full-loss/term and meeting-error statistics, optimizer iteration counts and peak VRAM.

Run `pytest vesuvius/tests/test_fiber_flow_uv.py vesuvius/tests/test_fiber_wrapped_irls.py` in the configured environment to check objective equivalence, analytic gradients/normal products, fixed labels/lengths and shared-CG behavior. Native descriptor and interpolation coverage lives in `vc_object_store/tests/python/test_selected_values.py` and `test_polyline_arc.py`. CUDA tests require an available GPU; workflow launches continue to use the configured GPU manager.

`constraint_batch` counts measured rows per native preparation task, not the numeric width of a source-ID interval. Sparse chunk-stable IDs therefore do not create empty tasks. Native workers consume the same read-only in-memory graph buffers; they do not exchange graph-part files.

For example, after configuring the source volume, prediction channels and umbilicus in a profile:

```sh
vc_wgraph hvb settings.json -i measurements.vcobj -o classified.vcobj
vc_wgraph sign settings.json -i classified.vcobj -o signed.vcobj
vc_wgraph winding settings.json -i signed.vcobj -o winding.vcobj
vc_wgraph uv settings.json -i winding.vcobj -o uv.vcobj
```

When multiple terminal input runs exist, select the appropriate run explicitly (`normal_sign_run_uuid` for winding, and UV's `input_run_uuid` for its winding run). Ancestor classification/sign runs are resolved from the selected run's pinned inputs; optional explicit `classification_run_uuid` or `normal_sign_run_uuid` selectors must agree with those pins. Inconsistent combinations fail before graph preparation. Ambiguity is an error; the workflow does not silently choose the latest result. CUDA solves use the existing managed launcher; `devices` selects the GPU(s), defaulting to device 1. Grants default to non-exclusive; an explicit `gpu_manager.exclusive` setting is preserved. Any host-specific GPU reservations belong in local profiles, not the reusable API.

The final console marker is `WGRAPH_STAGE_COMPLETE` followed by JSON with the stage, actual input/output paths and overall stage wall time, plus the persisted `run_uuid` for result stages. Native reports retain the separate ingestion and final-persistence timings; do not add overlapping worker timings or confuse final flush with extraction. Tests in `vesuvius/tests/test_fiber_wgraph.py` cover all six mappings, profile precedence, JSON validation, explicit overrides, dry inspection and dispatch through the existing native supervisor.
