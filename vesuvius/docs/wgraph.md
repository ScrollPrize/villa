# `vc_wgraph`: reusable winding-graph stages

`vc_wgraph` exposes the shared importer, extractor and solver stages. It does not embed datasets, host paths or alternative solver implementations. Install the normal `vesuvius` package to receive the console command; an existing development environment can also run `python -m vesuvius.wgraph` or `python vesuvius/scripts/vc_wgraph`.

```sh
vc_wgraph import import-settings.json -i fibers.afv -o fibers.vcobj
vc_wgraph extract extraction.json -i fibers.vcobj -o constraints.vcobj
vc_wgraph hvb solve-settings.json -i constraints.vcobj -o classes.vcobj
vc_wgraph sign solve-settings.json -i classes.vcobj -o signs.vcobj
vc_wgraph winding solve-settings.json -i signs.vcobj -o windings.vcobj
vc_wgraph uv solve-settings.json -i windings.vcobj -o uv.vcobj
```

Each stage accepts an optional profile name or arbitrary JSON configuration file; omission selects `default`. `-i/--input` and `-o/--output` override the actual input/output store paths. They never silently replace a separately configured report or working directory. JSON paths retain the existing native convention: relative paths are relative to the process's working directory, not the profile's directory.

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

`uv` consumes a selected saved winding run, fixes those integer assignments, and invokes the existing straight fixed-length UV optimizer. It records evaluated sampled UV positions, widths and fiber assignments in a referenced result store, without copying geometry. Generic `vc_wgraph` UV uses `component_selection: "all"`; explicitly choose `"largest"` to reproduce the historical largest-component experiment. Fibers without perpendicular support are excluded by the existing fixed-length model rather than assigned fallback geometry. Work arrays and packed geometry are disposable caches, not additional public inputs. Exact packed-path prior preparation reuses the same measured geometry and interpolation as winding preparation; solver weights and GPU-manager settings can be provided under `parameters` and `gpu_manager` respectively.

Store-to-store UV preparation and optimization run in the same managed process. Parallel native operations filter and remap the event graph, derive perpendicular support, sample measured geometry, construct fixed targets and fill owned numeric buffers. Python passes those arrays directly to the solver; it does not iterate over the fiber/event corpus or serialize intermediate input, selected-input or prepared tensor NPZ files. Final initial/solution checkpoints are outputs, not an interprocess data-transfer mechanism. Fresh packed-geometry caches use the same native seam and per-fiber arc construction as prior sampling. Configure `workers` for native preparation separately from the GPU optimizer settings.

Place large working directories and persistent caches on disk with sufficient free space. A RAM-backed filesystem such as `/dev/shm` consumes memory in addition to the solver's process allocations and is not an appropriate unbounded working directory for full-volume caches.

To reuse the winding stage's packed coordinates, set UV's optional `geometry_cache` to its `output_root/geometry` directory. The adapter verifies the exact base-store snapshot and selected classification/sign runs, then checks the catalog's working-coordinate frame, umbilicus axis, IDs, labels, signs, point offsets and array headers. A verified catalog is reused without rescanning source point counts. Only missing geometry blocks are scheduled; already completed blocks do not spawn workers. This is only a cache hint: omitting it rebuilds geometry from the referenced input store, without needing a saved solver checkpoint.

### Bulk preparation and UV optimization

Winding and UV use the same descriptor-only point-count catalog, typed result columns and packed polyline interpolator. Setup does not read every XYZ coordinate merely to calculate point counts. Run selection is pushed into native extension reads; numerical preparation consumes owned NumPy columns instead of per-fiber dictionaries. A single dense graph-to-catalog index is shared read-only by workers. Endpoint evaluation uses per-fiber native arc searches, not repeated textual UUID joins or searches of the entire packed point array. These temporary solver indices do not change persistent UUID identity, store-reference routing, geometry or extension formats.

The shared graph reader uses `fiber_graph_columns` for native node selection and binary-UUID endpoint joins. Disposable graph-cache format 3 stores dense numeric endpoints and binary edge UUIDs, not three repeated textual UUID columns per constraint. Older caches must be rebuilt explicitly; persisted vcobj measurement formats are unchanged. Diagnostic endpoint-ID views format only the requested rows, and numerical stages do not materialize those views.

UV's `parameters.uv_optimizer` selects `gauss_newton_cg` (default) or the existing `torch_lbfgs` comparison backend. Gauss–Newton preserves the squared fixed-assignment objective; it does not substitute the winding solver's L1 loss. Its matrix-free CG is shared with wrapped IRLS. The independent linear V block is solved first; horizontal physical anchors and the same learned width parameters use Gauss–Newton with Levenberg damping and actual-objective acceptance. Integer labels, exact straight fiber lengths, phase/Z priors, masks, class balancing and width penalties remain unchanged. No dense event-by-parameter Jacobian is built.

The width preconditioner uses a small exact width-by-width block accumulated from interval-pair moments, not one full graph pass per width. Reductions into the few width bins use independent contiguous partial sums before the final reduction, while node reductions retain direct indexed accumulation. This bounds temporary partial storage to roughly one scalar per 32 contributions and avoids a giant array of group indices, sorting or source copies. All arithmetic remains float64; parallel reduction order can differ, so CUDA results are not promised bitwise identical to the original atomic accumulation.

The CG controls are `uv_cg_iterations` (100), `uv_cg_tolerance` (`1e-6`), `uv_v_cg_iterations` (1000) and `uv_v_cg_tolerance` (defaults to the CG tolerance). `steps` remains an outer-iteration cap, not a convergence claim. `uv_lm_damping` defaults to `1e-3`, and five outer iterations below the `uv_tolerance` improvement threshold (default `1e-9`) terminate the solve. The GN backend requires a positive width parameterization; unbounded `direct` widths fail explicitly. All numeric graph columns are transferred once to the solver device, so memory scales with the complete graph rather than the old host batch size; measure VRAM before large runs. Reports retain full-loss/term and meeting-error statistics, optimizer iteration counts and peak VRAM.

Run `pytest vesuvius/tests/test_fiber_flow_uv.py vesuvius/tests/test_fiber_wrapped_irls.py` in the configured environment to check objective equivalence, analytic gradients/normal products, fixed labels/lengths and shared-CG behavior. Native descriptor and interpolation coverage lives in `vc_object_store/tests/python/test_selected_values.py` and `test_polyline_arc.py`. CUDA tests require an available GPU; workflow launches continue to use the configured GPU manager.

`graph_cache` selects the shared memory-mapped measurement columns across H/V/B, sign, winding and UV stages. By default it is `<base-measurement-store>.graph-cache`. Packed-preparation workers validate and reuse this base-store snapshot rather than creating another full copy of the graph under each stage's work directory.

`constraint_batch` counts measured rows per preparation task, not the numeric width of a source-ID interval. Sparse chunk-stable IDs therefore do not create empty tasks. Resume markers must match the complete task interval before their graph part is reused.

For example, after configuring the source volume, prediction channels and umbilicus in a profile:

```sh
vc_wgraph hvb settings.json -i measurements.vcobj -o classified.vcobj
vc_wgraph sign settings.json -i classified.vcobj -o signed.vcobj
vc_wgraph winding settings.json -i signed.vcobj -o winding.vcobj
vc_wgraph uv settings.json -i winding.vcobj -o uv.vcobj
```

When multiple terminal input runs exist, select the appropriate run explicitly (`normal_sign_run_uuid` for winding, and UV's `input_run_uuid` for its winding run). Ancestor classification/sign runs are resolved from the selected run's pinned inputs; optional explicit `classification_run_uuid` or `normal_sign_run_uuid` selectors must agree with those pins. Inconsistent combinations fail before graph preparation. Ambiguity is an error; the workflow does not silently choose the latest result. CUDA solves use the existing managed launcher; `devices` selects the GPU(s), defaulting to device 1. Grants default to non-exclusive; an explicit `gpu_manager.exclusive` setting is preserved. Any host-specific GPU reservations belong in local profiles, not the reusable API.

The final console marker is `WGRAPH_STAGE_COMPLETE` followed by JSON with the stage, actual input/output paths and overall stage wall time, plus the persisted `run_uuid` for result stages. Native reports retain the separate ingestion and final-persistence timings; do not add overlapping worker timings or confuse final flush with extraction. Tests in `vesuvius/tests/test_fiber_wgraph.py` cover all six mappings, profile precedence, JSON validation, explicit overrides, dry inspection and dispatch through the existing native supervisor.
