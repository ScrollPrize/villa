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

The winding command consumes extracted measurements with saved classification/sign runs, or the fibers' stored H/V tags, and optionally the scaffold's `fiber_branch` annotation links (see [winding inputs and weights](#winding-inputs-and-weights)). H/V/broken and normal signs use wrapped IRLS; winding uses linear GPU IRLS or the CPU LEMON min-cost-flow backend (see [LEMON backend](#lemon-backend)); UV uses the fixed-assignment optimizer described below. There is no `vc_wgraph compare` subcommand. Existing application reference-evaluation helpers do not change these command-line capabilities.

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

### Crossing evaluation

`vc_wgraph crossings CONFIG` scores solved windings against the crossing
measurements themselves, one row per measured crossing: no fractional target
expansion, no aggregation by fiber pair and no clamping of the measured
separation. The relation is `k_b - k_a = target`, where the target is the
measured winding separation corrected for the H/V offset and the seam
transport, and the residual is reported both unrounded (against that measured
target) and rounded (against the target rounded to a whole turn, the relation
the L1 problem can state). Four numbers per group: unrounded mean and median,
rounded median and the share of crossings the solve satisfies exactly. A
nonzero rounded median is a systematic winding shift; an unrounded mean away
from zero without it is a sub-turn displacement.

The config names `graph_store` (the sign-lineage store the solve read),
`winding` (the same settings the solve used), `solves` (label to result
store), optionally `focus` (a JSON file of fiber UUIDs, or a list: only
crossings between a focus fiber and another fiber are scored) and `output`.
Results are grouped by distance cut (`|round(measured separation)| <= 0, 1,
2`) and, for the tightest cut, by the winding the solve gives the focus side.
Only the largest connected component is scored unless
`largest_component_only` is false.

The store chain is `import → extract → hvb → sign → winding → uv`. `winding` consumes typed classification and normal-sign runs from its input store references, builds the existing fractional-target L1 graph, and runs the historic linear GPU IRLS solver (15 reweighting rounds, 500 CG iterations, epsilon .05 unless explicitly overridden). It records integer readout, continuous potential, connected component and residual metrics. These are point estimates, not calibrated winding probabilities. The IRLS solve runs in double precision: in single precision the CG loses the true residual once weakly attached groups (total weight far below 1) sit next to strongly weighted ones, as with `scaffold_reference`, and leaves those groups hundreds of windings off. Double precision costs about 200 MiB more GPU memory and 20 % more solve time on the full PHerc0332 graph (6.4 million constraints: 505 MiB, 7.7 s); a mixed scheme (double solution and residual, single-precision CG corrections) gave the same result at the same memory, so this choice is open to revisit for consumer GPUs. Every reweighting round measures how far its solution still is from solving its linear system: the true residual divided by the diagonal, the distance in windings one more Jacobi step would move a fiber. A weakly attached group left behind shows there while it vanishes in the residual norm (full PHerc0332 graph with `scaffold_reference`: at most 0.0015 windings per round in double precision, up to 6 in single). The run records the largest value and the largest solution range (`irls_largest_node_update`, `irls_largest_solution_range`) and fails when a round stays above `max_node_update` (default `0.05` windings). No GT labels, spiral reference solution or external solver-source file is required. The measured geometry and umbilicus working-coordinate binding remain required inputs for seam transport.

### Winding inputs and weights

`winding` assembles one weighted integer L1 arc list from up to three constraint types and solves it with the selected backend:

| Setting | Values | Meaning |
|---|---|---|
| `weights.parallel`, `weights.perpendicular` | `>= 0`, default 1 | Automatic measured constraints from extraction (same-family and H/V), per expanded fractional-target arc. Zero omits the type. |
| `weights.enclosure` | `>= 0`, default 0 | Exact H->V relations derived from signed perpendicular events of one pair (largest lower bound equal to smallest upper bound, after seam and connector transport); one extra arc per accepted pair, the witnessing perpendicular arcs remain. Pairs that are one-sided, contradictory or non-unique are counted, not used. |
| `normal_dot_weight_power` | `>= 0`, default 0 | Weighs every automatic crossing by `\|dot\|^power` of its two normal axes, the rule the [sign stage](#sign-stage-weights) uses. 0 is the flat per-type weight. |
| `winding_distance_weight` | `inverse_square`, `none` (default) | Damps every automatic crossing by `min(1, 1/d^2)` of its measured winding separation, again as the sign stage does. |
| `fixed_object.windings` | path, optional | A previous solve's windings JSON (`ids`, `selected_winding`, `component`). Each of its components with two or more fibers in this store is contracted into one node, so the solve may move the group but never its members against each other. |
| `weights.sign` | `>= 0`, default 0 | One-sided sign constraints: every signed perpendicular crossing states on its own that the V fiber lies outside the H fiber there, or inside, which bounds the pair's H->V step from one side (the bound an enclosure needs from both). It costs only when the solve breaks it, needs no winding separation, and is scaled by the crossing's evidence like the equations. Both backends take it (IRLS as a one-sided term, LEMON as a one-directional arc). |
| `weights.manual` | `>= 0`, default 0 | The scaffold's `fiber_branch` links read from the resolved geometry store. Ordinary links assert the same winding at their anchors; adjacent links assert the V fiber one winding inside the H fiber (`W_V = W_H - 1`, from the stored tags only). Same-family adjacent links, links to disabled fibers and links outside the selected approval states are skipped and counted. |
| `scaffold_reference` | object | A separately solved scaffold takes part as one fixed object, without a union graph: `constraints` is the cross-constraint dump `vc_scaffold_solve` writes for `field_comparison` (with its `.json`), `windings` that run's `windings_output`. Only the scaffold's largest component is used. Each cross crossing ties its volume fiber to one extra node at the scaffold fiber's own winding, as an equation or a one-sided sign constraint, re-weighted with this stage's `weights` (`file_weights` states the weights the dump was written with, default perpendicular 1, sign 100). The volume fibers connected to it are numbered in the scaffold's own windings. Not together with `fixed_object`. |
| `manual_links.approval_states` | list, default `["approved"]` | Approval states of links to use. |
| `manual_links.include_auto` | bool, default false | Also use pending automatic (`meta.auto`) links. |
| `manual_links.max_anchor_distance` | voxels, default 5 | Anchors farther from their fiber polyline fail the run (coordinate-frame guard). |
| `classification` | `run` (default), `record`, `run+record` | Families from the pinned classification run, from the fiber records' `family` column (`record_family`: `any`, `manual`, `automatic` selects by `manual_family`), or the run with the tagged fibers overriding it. With `record` no `hvb` run is needed; `run+record` pins the same run as `run` and keeps every tagged fiber enabled with its tagged family, which is how a hand-tagged scaffold imported into a traced volume keeps its own fibers when the classifier calls them broken. |
| `normal_sign` | `run` (default), `none` | A pinned normal-sign run, or none. `none` is only valid with zero automatic weights (manual links only) and `classification: record`; such a run pins no input run. |
| `umbilicus` | path or `store` | The legacy umbilicus document, or the input store's own umbilicus object (which retains that document's metadata). |
| `handedness` | `+1`, `-1` | The scroll's winding direction for seam transport: `+1` when the winding index grows with increasing angle around the umbilicus, `-1` otherwise. Normally not set here: the stage reads it from the store umbilicus object's `winding_handedness` meta (`vc_wgraph handedness -i store.vcobj --set -1`), else from the open-data catalog orientation (`-1` when exactly one of `z_direction_is_top_to_bottom` and `left_handed_coordinates` holds, as VC3D reads it). A volume whose handedness is stated nowhere is an error, never a guess. The run records the value used. |

The scaffold solve (`vc_scaffold_solve`, volume-cartographer `docs/fiber_map_scaffold_solve.md`) builds its constraints from the same native function (`windingEventConstraints` in vc_object_store `geometry/FiberWindingGraph.hpp`): the per-crossing weight, the equation arcs, the enclosure bound and the sign constraint mean the same in both. Its defaults are perpendicular 1, sign 100, parallel and enclosure 0, with the crossing evidence on.

Extraction with `winding_distances: false` (no winding manifest, see [fiber workflows](fiber_workflows.md)) yields measurements without distances: they contribute no parallel/perpendicular arcs (counted as `without_distance`) but still witness enclosures and feed the sign stage, where an unmeasured distance counts as within one winding.

### Sign stage weights

The sign stage uses the perpendicular crossings only. Each crossing's weight is `|dot|^normal_dot_weight_power` of its two normal axes (default power 2, so well-aligned crossings dominate) times a winding-distance factor: `winding_distance_weight: inverse_square` (default) multiplies by `min(1, 1/d^2)` of the measured winding distance, so crossings more than one winding apart still count but only decide a sign when nothing closer does; `none` disables the factor. The run records both settings. Besides the weighted disagreement, every run reports the strict internal metric `strict_disagreement_percent`: the unweighted share of disagreeing crossings among those whose axes are within 30 degrees and whose winding distance is below 1. On the PHerc0332 traced volume (1.86 M crossings) the defaults lowered that metric from 0.074% (`|dot|` weights, no distance factor) to 0.054%. The same objective also exists as a native CPU solver (wrapped IRLS and a Ceres backend, no GPU) for interactive callers: see [native CPU normal-sign solve](../../volume-cartographer/docs/fiber_normal_sign.md). It is not a backend of this stage.

A crossing's evidence weight is its type weight times `|dot|^normal_dot_weight_power` times the separation damping, so the same crossing can be weighed here exactly as the sign stage weighs it. The defaults (power 0, damping off) keep the flat per-type weights every earlier run used, and each run records both settings (`normal_dot_weight_power`, `winding_distance_damped`).

A `fixed_object` names a solve whose groups this run must keep: every component of that solve with two or more fibers in this store becomes one node. An arc inside a group is dropped (the group already fixes that relation) and counted, with the turns it disagrees by; an arc leaving a group moves to the group's root with its target shifted by the member's offset; after the solve each member takes its root's winding plus that offset and the root's component. The run counts the groups, the fibers they fix, the single-fiber components that fix nothing, and the dropped internal arcs under `fixed_object`. This is how a manual scaffold enters a full-volume solve as a rigid body rather than as fibers that may drift apart.

The anchor relation of a manual link is converted to a whole-fiber step with the same seam and connector transport the automatic constraints use (`target = step + connector - turns_b + turns_a`). Arcs of all types are aggregated by (a, b, target) before solving, so the weights add up per identical relation. Each run records the weights, link counts and enclosure counts in its settings; the former single `perpendicular_relative_weight` is rejected. Without automatic constraints no extraction is needed: the nodes are the store's active H/V fibers, and the geometry is read from the fibers' bound volume. The `sign` stage accepts `classification: record` the same way.

### LEMON backend

`optimizer: lemon` solves the assembled arc list exactly as a capacitated minimum-cost circulation with LEMON (`lemon.backend`: `network_simplex` (default) or `cost_scaling`), built into `vc_object_store` (`VC_OBJECT_STORE_WITH_LEMON`, `liblemon-dev`). Weights become integer capacities `round(weight * lemon.capacity_scale)` (default 1000); arcs rounding to zero are dropped and reported. The result is an exact integer optimum of the quantized objective: `selected_winding` holds the integer windings directly, `winding_mean` repeats them, and the run metrics report the original-weight L1 residuals. It runs on the CPU in the calling process with no GPU-manager grant, torch or `devices` setting, and writes the same frozen `config.json` (with code provenance) and `summary.json` under `output_root`. It is intended for small problems such as manual scaffolds: LEMON's int indices bound nodes and arcs below about 500 million, the solver is single-threaded, and memory grows with the arc count; full-volume graphs stay with `optimizer: irls`.

Manual scaffold solve from a store that holds fibers, tags, links and an umbilicus:

```sh
vc_wgraph winding lemon-scaffold -i scaffold.vcobj -o windings.vcobj
```

The packaged `lemon-scaffold` profile selects the LEMON backend, tags, no sign run, the store umbilicus, manual links (approved and pending, no automatic links) and zero automatic weights; add a `sign` run and nonzero automatic weights in a local profile to combine scaffold links with extracted constraints.

The same solve, with the scaffold's own crossings measured in one process and its residuals ranked for review, is the native `vc_scaffold_solve` and the VC3D fiber map's solver dock: see [scaffold winding solve](../../volume-cartographer/docs/fiber_map_scaffold_solve.md). It reads one store, needs no extraction run beforehand, and records every run in the store it solved.

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
