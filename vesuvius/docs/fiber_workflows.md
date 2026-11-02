# Fiber-layout workflows

The preparation, tuning, diagnostics and surface-export workflows are shared
Python modules, not local runner scripts. Configuration and datasets remain
external. The solver implementations and scientific defaults are retained;
this restoration does not select a new model or change its losses.

## Run

From an environment configured for this checkout:

```sh
python -m vesuvius.neural_tracing.fiber_trace_3d.fiber_workflows --config /configs/fiber-layout/job.json
```

The JSON contains **one** workflow key and its settings, as below. Relative paths
are relative to the working directory, not the config file. Use absolute paths
for jobs that launch child processes. A local approval wrapper should only choose
the interpreter, environment and configuration; all workflow logic lives here.

Use the existing Vesuvius ML environment, `lasagna` and `villa_gpu_manager` from
this repository. Individual tools additionally use SciPy, threadpoolctl, psutil,
matplotlib, tifffile/imagecodecs, scikit-image (connectivity) or boto3 (S3 download).
No command installs dependencies or downloads inputs implicitly. CPU diagnostics
do not acquire GPUs. GPU solve stages use the existing manager launcher; configure
`devices` and `gpu_manager` in their solver configs. Non-exclusive grants are
supported. The optional matrix performance harness requires an exclusive parent
grant, unlike ordinary search trials.

Native extraction uses `vc_fiber_constraints` on `PATH`, or `native_binary`.
Command-list jobs also accept `native_binaries` (name-to-path mapping) and
`binary_dir`. Old `vc_fiber_trace_chunk afv-constraints CONFIG` requests are routed
to the same standalone exporter. Supply `native_regression_binary` to the
normal/axial-prior wrappers to run the native regression executable first.
Other native tools, such as reference replay, must be available separately.

Normal-orientation publication uses a bounded producer/writer pipeline: endpoint
JSON validation and serialization run in parallel, while a single SQLite writer
updates up to 32 changed rows per statement. Byte-identical rows are skipped;
axis/validity checks, transactional publication and final database integrity
checks are retained. Older SQLite versions without `UPDATE FROM` use scalar
updates with the same parallel producer. Publication statistics include scanned,
changed and unchanged rows, plus read, encoding and write timings (overlapping,
so these are not additive wall time).

Build the native `test_fiber_normal_publication` target to validate exact output
against the previous scalar implementation. Run it without arguments for
regressions, or with `--benchmark 1000000 8 32` for three paired measurements
(rows, encoding threads, SQL batch size). This is an in-memory microbenchmark;
full-database cloning, disk I/O and integrity checks require end-to-end timing.

## Preparation and evaluation

`fiber_selected_solver_comparison` can reuse an existing `selection_root` with
frozen sign/H/V/broken classifications and extracted constraints. Optional
`minimum_length_mm` and `minimum_width_base` strictly filter whole fibers before
the existing largest-component/perpendicular-support selection. Length uses the
AFV's explicit `root.voxel_size_mm`; width uses the stored piece-local
length-weighted `fiber_attributes.width_mean` in base voxels (not the parent
coverage-policy width). Missing widths are excluded and counted. The selection
report records threshold counts and final mean/median/total physical length.
No sign estimation, H/V classification or constraint extraction is repeated.
Use `include_sgd:false`, `exact_lp:false`, `capacity_scales:[]` and
`uv_methods:["irls"]` for IRLS only followed by fixed-label U/V fitting.
These filters must match when reusing a `prepared_root`.

### Same-input IRLS winding comparison

`fiber_irls_benchmark` compares an external winding-field `sheetgraph.solve_l1`
implementation with the existing LEMON backend on the identical aggregated
fractional-target graph. Supply `solver_source` and its `solver_sha256`, a
`dataset` (`afv`, `tensor_cache`, `umbilicus`, measurement calibration/weights),
`sgd_solution`, `events`, `gt_config`, `gt_solution`, and a fresh `output_root`.
Optional `sgd_report` validates the historical SGD score and records its timing.
GPU settings use the standard nonexclusive `devices`/`gpu_manager` launcher.

The external module is imported directly and must already have its dependencies
installed. No code/data is downloaded. Defaults are 15 IRLS reweightings,
500 CG steps, epsilon .05, and the upstream 1e-6 zero-prior weight. Both raw
continuous potentials and upstream-style component-phase-rounded labels are
reported. The tiny prior and finite epsilon mean IRLS is an approximate solver,
not a certified exact L1 optimum. The exact fractional-capacity LP and LEMON
capacity scales 1000/1 provide controls on the same graph.

No track splitting, surface inference, winding-inference seeds, or UV refinement
is used. The saved independent SGD winding checkpoint is scored, not retrained;
its source IDs, intervals and families must exactly match. Continuous endpoint
GT L1/correctness and geometric sign errors use existing shared helpers and a
single global integer gauge. Report full graph-solver coverage and the identical
SGD-retained event subset separately. SGD has rejection and geometry losses;
its training objective is not numerically comparable to graph L1. Timings from
nonexclusive grants are operational, not isolated hardware benchmarks.

`fiber_length_histogram` takes `comparison` (the per-fiber NPZ from
`fiber_hv_label_comparison`) and `report` (output JSON). It reports source arc
lengths in base voxels: count histograms and quantiles for all/H/V/broken fibers.

For a CPU-only 3D preview of changed source-family labels, use:

```json
{"fiber_hv_crop": {
  "afv": "/data/fibers.afv",
  "comparison": "/results/hv_comparison.npz",
  "output_root": "/results/change_crop",
  "size_base_voxels": 2048,
  "minimum_each_change": 3
}}
```

The comparison is the per-fiber output of `fiber_hv_label_comparison`. The crop
is selected deterministically from cells containing both change directions;
alternatively supply `bounds_xyz: [[xmin,ymin,zmin],[xmax,ymax,zmax]]`.
The new output directory contains five line OBJs (`h`, `v`, `broken`,
`h_that_were_v`, `v_that_were_h`), an MTL and `index.json` with bounds and source
IDs. Changed subsets also occur in their final H/V group. Original raw curves
are segment-clipped, without repairs, rescaling or translation. Pairwise OBJ
line elements avoid accidental connections across gaps and work with MeshLab.
For a whole-volume preview instead, replace the size/minimum-change options
with `"sample_per_group": 100, "seed": 42`. This independently samples each
of the five groups uniformly without replacement, reads the original full
fibers without clipping, and records the seed and selected IDs in the index.
Changed-group samples need not also be selected in the independent H/V samples.

| Workflow key | Inputs / result |
|---|---|
| `fiber_json_solve` | Reviewed JSON directory + AFV metadata template + explicit ordered stages; creates AFV and dispatches extraction, normal/HV/direction and layout requests. |
| `fiber_anchored_dataset` | Reference-curated source graph; constructs an anchored dataset with fixed reference events. |
| `fiber_selected_dataset_eval` | `dataset`, `base_config`, `gt_root`, `output_root`; filter unsupported sources, split if configured, initialize, solve and score. |
| `fiber_selected_score` / `fiber_selected_checkpoint_report` | Score completed or published intermediate states without rerunning optimization. |
| `fiber_joint_gt_eval` | `baseline_root`, `base_config`, `gt_root`, `output_root`; attach GT, with `force_gt_enabled: true` or `false`. `report_only` reads completed output. |
| `fiber_gt_inpainting` | Fixed-reference solve and optional post-solve deformation. |
| `fiber_neighbor_expansion` | `seed_solve`, whole `parent_tensors`, `afv`, `gt_root`, `z_range`, extraction/orientation templates; MAP-select neighbors, re-solve, re-filter per round. |
| `fiber_neighbor_pair` | Run and compare no-GT/forced-GT variants on the expanded dataset. |
| `fiber_full_slab` / `fiber_full_slab_soft` / `fiber_full_slab_progress` | Whole-slab dataset construction, soft-only control and checkpoint reporting. |

Minimal selected-set job:

```json
{
  "fiber_selected_dataset_eval": {
    "dataset": "/data/fiber-layout/selected",
    "base_config": "/configs/fiber-layout/base.json",
    "gt_root": "/data/fiber-layout/gt",
    "output_root": "/data/fiber-layout/results/free",
    "devices": "0"
  }
}
```

`base.json` wraps the solver config under `fiber_implicit_winding`. The selected
dataset contains `selected.afv`, `tensors.npz` and `cross/cross_events.json`.
`gt_root/uv_whole` contains the trusted `config.json`, `solution.npz` and referenced
tensors. Joint evaluation uses a prepared baseline's `tensors.npz` and
`gt_events.json`. These schemas are also checked by the
[three-arm replay benchmark](fiber_layout_benchmark.md).
Editable samples: [selected-set evaluation](examples/fiber_workflows/selected_dataset.json),
[GT inclusion](examples/fiber_workflows/joint_gt.json), and
[neighbor expansion](examples/fiber_workflows/neighbors.json).

For reviewed JSON, provide `expected_manifest_tokens` (source volume/scan tokens)
and supply `umbilicus` in stages that require it. Do not infer scan identity from array shape alone.
Use `stages`, an ordered list of `{name, config_file}` or `{name, request}`
entries. Each request is a complete workflow object (or native command-list
job). Stage names are caller-selected output labels, not preset selectors.
`{afv}` and `{output_root}` are expanded in request string values; referenced
native configuration files must already contain their own correct paths. No
solver settings, enablement policies, pass counts or GPU reservations are injected.
An empty list converts JSON to AFV only. `layout_output` optionally requests the
categorical-layout report; omit it for other checkpoint formats. See the
[JSON pipeline example](examples/fiber_workflows/json_pipeline.json).

For surface projection, `reference_xyz_scale` is required in anchored-dataset and
full-slab jobs (including slab scoring jobs). Supply the reference-to-base-voxel
scale for the actual input, rather than assuming a particular scan resolution.
Projection stride, candidate count and sampling distance are configurable.
`fiber_gt_inpainting` runs deformation only when `deformation_parameters` is
provided. `fiber_current_gt_solve` renders only when `visualization` is supplied.
Both settings objects pass caller choices through to the existing implementations.

## Iterative search and regression problems

`fiber_split_weight_search` restores asynchronous three-arm scheduling, screening,
candidate reuse, score recomputation, coupled proposals, discrete split choices
and coarse-to-ultrafine coordinate search. The objective remains summed GT
perpendicular sign violations **divided by** selectable retained-GT arc fraction;
summed GT perpendicular L1 breaks ties. The threshold is probability >= 0.5.

The [minimal search example](examples/fiber_workflows/split_search.json) shows
bounds, discrete choices and a coupled proposal. It is an interface illustration,
not a tuned parameter set. Unspecified solver settings come from `base_config`.
Set `background: true` for a persistent coordinator; it writes its PID,
config and log. Otherwise the process remains attached. `accepted_improvements:
null` keeps searching; an integer is an explicit stop limit. Resume with identical
settings and code in the same output root. Changed scientific inputs/code require
a new root; reuse checks must not be bypassed. Linux process-control actions verify
the exact module/config before signalling; use normal process control elsewhere.

The earlier `fiber_paired_weight_search`, `fiber_density_weight_search`,
`fiber_selected_weight_search` and `fiber_uv_layout.loss_weight_search` remain
available. They retain their own ranking protocols; do not mix their scores.

The small-GT matrix is data-independent code:

1. `fiber_current_gt_solve`: construct all-event or enclosure-only graphs.
2. `fiber_gt_width_initialization`: generate explicit width-factor variants
   (`kind: "scale"`, factors 0.5 and 2.0), preserving all other initialization arrays.
3. `fiber_gt_convergence_probe`: run a `trials` list from supplied checkpoints
   using the configured optimizer and full objective.
4. `fiber_reference_check`: construct/use consensus regression references;
   reports geometry, signs, length, angle and meeting checks.
5. `fiber_uv_weight_search.regression_guard`: apply the recorded acceptance
   limits. Supply limits/reference data and any excluded source IDs explicitly.

Choose regression modes, initialization variants, sampling and acceptance limits
in external configurations. The [width-regression sample](examples/fiber_workflows/gt_width_regression.json)
illustrates two caller-selected trials; it is not a required test matrix.
The [no-initialization-width-prior rule](fiber_layout_no_width_prior.md) applies
to every workflow. In paired searches, `gt_parameter_overrides` explicitly maps
regression names to mode-specific overrides; names never select hidden behavior.
Non-screened paired searches use two large-run GPU lanes and the remaining lanes
for the supplied regression suite. `gt_gpu_manager` optionally overrides regression
resource requests; otherwise those come from each supplied regression request.

## Diagnostics, visualization and export

| Workflow/API | Functionality |
|---|---|
| `fiber_sign_rejection_report` | Compare supplied `roots` against `baseline`; `internal_audits` maps each root to its audit summary path. No implicit grids or audit-directory discovery. |
| `fiber_density_sign_report` | Internal or frozen-GT sign counts, distributions and detailed error breakdown. |
| `fiber_implicit_match_report` | Same-sheet meeting metrics over probability cutoffs. |
| `fiber_reference_rounding_audit` | Floating vs integer scoring and alternative references. |
| `fiber_forced_selectable_audit` | Exact parent-geometry/hypothesis transfer and full objective comparison on selectable graphs. |
| `paired_winding_review` | Matched-cohort changed-winding plots, dotted fibers and sign-neighbor highlighting. |
| `fiber_direction_review` | Direction disagreement audits and OBJ exports; `normal_visualization` configures the shared normal-arrow exporter. |
| `fiber_hv_label_comparison` | Compare an H/V-stage `solution` NPZ with `afv` family labels. Writes JSON to `report`; optional `per_fiber_output` NPZ and `probability_cutoffs`. Reports count/arc disagreement and confusion by family, keeping normal-stage and H/V-stage exclusions separate. Source labels are comparison-only. |
| `fiber_uv_refinement_inspection` | Matched before/after deformations, movement in voxels and windings, plots. |
| `fiber_fixed_metric_probe`, `fiber_optimizer_probe`, `fiber_lbfgs_audit` | Fixed-meeting/length/angle probes and optimizer/conditioning comparisons. Fixed-metric rendering is opt-in via `visualization`. |
| `fiber_uv_surface` / `fiber_uv_surface_audit` | H-anchored tifxyz surface construction and audit from historical categorical U/V solutions. |
| `fiber_reference_surface` | Explicit S3 reference acquisition with integrity/provenance checks. |
| `fiber_pair_connectivity` | Existing two-fiber threshold/component/skeleton diagnostic with reference strips and OBJ export. |

Reference downloads may resume or refresh the same S3 prefix. Reusing an output
directory for a different source, or leaving local files absent from the current
source listing, is rejected before downloading. Use a new/clean output directory;
the downloader does not delete old reference files automatically.

Surface usage:

```json
{
  "fiber_uv_surface": {
    "solutions": [{"label": "candidate", "run_root": "/data/fiber-layout/uv-solve"}],
    "normal_curves": "/data/fiber-layout/normal-curves",
    "coordinate_space": "YOUR_VOLUME/YOUR_SCAN",
    "output_root": "/data/fiber-layout/surfaces",
    "grid_step": 100, "max_search": 1000, "joint_threshold": 0.5
  }
}
```

This exporter retains its original checkpoint contract; it does not silently
reinterpret a fixed-length/probability-model checkpoint as categorical geometry.
The separate `fiber_exact_deformation` and `fiber_uv_surface_refinement` workflows
retain their own checkpoint schemas.

Extra read-only tools have module CLIs (use `--help`):
`fiber_constraint_database_report`, `fiber_constraint_database_validation`
(requires AFV and the matching C++ header), `fiber_oriented_normals_validation`,
and `fiber_violation_neighborhoods` (consumes `paired_winding_review` output).

Performance modes restored in `fiber_uv_layout`: `benchmark_pair_fusion`,
`diagnostic_benchmark`, `distributed_suite` and `experiment_matrix`.
`indexing_audit` is restored in soft/density optimization. They reuse the normal
solver algebra; enable them explicitly. No new speed or convergence claim is made.

## Local archive and verification

### Whole-fiber winding-range selection with minimum-cost flow

`fiber_flow_uv` fits conditional U/V geometry after flow selection. It takes
`flow_root`, `umbilicus`, `output_root`, `devices`, and `gpu_manager`. Fibers with
no perpendicular events are explicitly removed and recorded in `selection.json`;
neither remaining integer labels nor enabled status are optimized. A common
integer gauge shift puts width lookup indices in range; all relative assignments
are unchanged. The conditional targets at event bases include the flow's exact
seam transport, not rounding of evolving U/V coordinates.

This mode reuses `SharedCurve` straight fixed-length geometry, monotone widths,
width regularizers, the scaled PyTorch L-BFGS optimizer and TIFF/PDF drawing sinks.
It does **not** replay the probabilistic assignment objective: fixed integer
event separations supervise approximate cross-winding coordinates, while fixed
same-sheet perpendicular events additionally supervise physical U/V meetings.
Parallel constraints supervise transverse coordinates only. Angular/Z priors
are sampled along curves; radii initialize widths but never provide a width prior.
`meeting_weight`, `meeting_scale`, `distance_weight`, `metric_weight`,
`prior_weight`, `v_prior_weight` and `width_smooth` control this conditional fit.
There is no free bending or length change in this first geometry mode.

`initial_solution` continues a compatible fixed-label fit after checking every
fiber/target identity. `visual_initial` optionally renders the original initial
checkpoint alongside a continued solution. Outputs include `fibers.afv`, tensors,
`solution.npz` (fixed labels, widths, shared U/V anchors), complete loss/mismatch
history, and equal-scale `visualization/{initial,solution}.{tif,pdf}`. TIFFs use
LZW compression. Re-running an identical completed request only exports/renders;
it does not allocate a GPU again.

The `fiber_min_flow_selection` workflow combines an oriented constraint database
with final normal/HV decisions and a small intact GT fiber set. Required settings:
`afv`, `database`, `normal_solution`, `comparison`, `gt_root`, `umbilicus`,
`extract_template`, `orient_template`, `native_binary`, `output_root`, `z_range`.
Optional settings include `threads`, `seed_spacing` (64), `maximum_distance` (128),
`max_rounds` (10), `perpendicular_relative_weight` (1),
`perpendicular_offset` (0.5), and `lasagna_winding_scale` (1).

It finds spatial GT neighbors, extracts their GT cross-events using the existing
native workflow, and reads source/source measurements through the AFC indexes.
Each round jointly solves candidates, filters to GT-connected fibers intersecting
the GT H winding range within the Z slab, then jointly re-solves the filtered set.
Subsequent rounds include one-hop eligible neighbors. All output curves remain
whole; only the selection test is restricted to the slab. The flow solver has no
probability/disabling model: upstream normal/HV rejections apply, and all seven
(or however many supplied) GT fibers remain enabled. No SGD confidence threshold
is substituted. Stable membership or the explicit round limit ends expansion;
a monotone prune/re-solve cleanup then ensures the exported set satisfies its
own final range/connectivity test. `finalize_only: true` can finish this cleanup
from an existing `selected_tensors.npz` and `summary.json` without re-expansion.

`history.json` records counts, timings, geometric-order violations and certified
flow objectives. Its `measured_target_l1` is fit to calibrated measured targets,
**not** error against independent GT winding labels. `selected.afv` contains
original source IDs/geometry and the resolved H/V family labels;
`selected_with_gt.afv` adds remapped GT IDs documented
in `gt_id_mapping.json`. Matching tensors and integer solution arrays are saved
separately. Original source files are read-only.

An optional `frozen_gt_cross` directory can reuse historical signed GT normal
samples only after checking the GT AFV, normal solution, tensors and solution
hashes against its `inputs.json`; this avoids silently changing the GT normal
orientation when extraction implementations have evolved.

The `fiber_min_flow_full` workflow solves all upstream-eligible source fibers,
without a spatial crop or added GT. Required settings are `afv`, `database`,
`normal_solution`, `comparison`, `umbilicus`, and `output_root`.
`experiment_link` optionally creates a convenient output-directory symlink.
Preparation uses `workers` (32), `geometry_batch` (4096 fibers),
`constraint_batch` (500000 database IDs), and `read_batch` (32768 rows).
The same `perpendicular_relative_weight`, `perpendicular_offset`, and
`lasagna_winding_scale` settings as selection control the shared flow objective.
Set `prepare_only: true` to publish the packed geometry and `graph.npz` without
starting HiGHS. The summary then records `prepared_only: true` and no solution is
written. This graph can be passed as `source_root` to `fiber_irls_benchmark` with
`full_graph: true` for an IRLS-only solve of its largest connected component.

Exact original polylines and seam transport are packed into memory-mapped arrays;
workers stream disjoint constraint-ID ranges. Only identical L1 arcs are combined,
preserving fractional-target terms. Preparation blocks are resumable with matching
input identities and configuration; the optimizer itself restarts if interrupted.
`progress.json` tracks preparation/solve state, `solver.log` retains native solver
progress, and `summary.json` is written after completion. `solution.npz` contains
source IDs, resolved family, integer winding, flow and connected-component IDs.
Component winding origins are arbitrary and normalized independently. This is a
CPU solve with no GPU request; full-volume optimizer time need not scale linearly
from cropped runs. Source AFV/AFC files remain read-only.

`fiber_flow_threads` benchmarks the saved full-volume graph without extracting
anything again. Set `source_root`, a fresh `output_root`, optional
`experiment_link`, `thread_counts` (default `[1,2,4,8]`) and `iterations`
(60000 per trial). It presolves once, benchmarks serial dual simplex and parallel
PAMI from the same reduced LP, then continues the selected trial's basis to
completion. Both pivot throughput and logged objective improvement per second
are recorded in `trials.json`; selection uses objective improvement per second
so unproductive cheap pivots do not automatically win. Native progress intervals
provide min/median/p95/max pivot rates, not independent repeated-run confidence
intervals. The installed HiGHS simplex concurrency is capped at eight even if
more scheduler threads are requested; both are recorded. The best basis is
checkpointed to `best.basis`. The final solution must pass the same original-graph integer
primal/dual certificate as the ordinary solver.

This workflow uses SciPy's private bundled HiGHS interface for presolve/postsolve
and basis retention; compatibility is covered by backend-equivalence tests and
must be checked when updating SciPy. It runs synchronously and must not share a
process with another active HiGHS solve (thread scheduler is global). Optional
`stop_previous_pid` plus exact `previous_command` can replace an earlier owned
workflow after preparation succeeds; process identity checking uses Linux
`/proc`. Omit these options elsewhere. Benchmark logs, selection and final solve
log are retained separately; it never overwrites the source graph or old logs.

### Optional specialized integer-capacity flow benchmark

`fiber_lemon_benchmark` compares LEMON network simplex and cost-scaling on an
already solved dataset. It requires the optional system LEMON development package,
`pkg-config`, and a C++17 compiler; it never installs these automatically. The
explicit benchmark builds its small native adapter in the output directory with
Release optimization and records source hash, compiler and LEMON versions.
No GPU is required. Configure, for example:

```json
{
  "fiber_lemon_benchmark": {
    "output_root": "/results/flow-backend-comparison",
    "dataset": {
      "afv": "/data/fibers.afv",
      "tensor_cache": "/data/tensors.npz",
      "umbilicus": "/data/umbilicus.json",
      "perpendicular_relative_weight": 1,
      "perpendicular_offset": 0.5,
      "lasagna_winding_scale": 1
    },
    "baseline_solution": "/data/solution.npz",
    "baseline_certificate": "/data/certificate.json",
    "scales": [1000, 1],
    "repeats": 3
  }
}
```

Fractional target expansion is unchanged. Identical arcs are aggregated first,
then capacities become `floor(scale * weight + 0.5)`; zero-capacity arcs are
dropped, not clamped to one. Scale 1 is direct integer rounding, while larger
scales preserve finer weight distinctions. This deliberately changes the objective
and is not the earlier variant that rounded measured winding targets.

`results.json` records dropped arcs/weight, repeated native graph/setup/solve/export
timings, complete certification wall time, and original-weight L1 and sign metrics.
Each result must pass the existing primal/dual certificate for its quantized graph
and match HiGHS on that exact graph. Original-weight scores are evaluated separately;
certification of the quantized graph is not certification of the original objective.
Overall mean L1 divides by the total contributing weight, including valid unsigned
constraints; sign diagnostics necessarily include only known-sign constraints.
If quantization disconnects components, comparison gauges are explicitly aligned
to the baseline and cross-component original events are reported.

Run the optional native regression tests with:

```sh
python -m pytest -q vesuvius/tests/test_fiber_lemon.py
```

They skip if the optional compiler/package is unavailable; no setup is performed.

### Full cached-graph LEMON solve

`fiber_flow_perpendicular` creates a perpendicular-only experiment from a full
graph and calls the same LEMON runner. Configure `source_root`, `output_root`,
`minimum_perpendicular_constraints` (default2), `scale`, and `reference_mean`.
It relies on the full builder's H/V compatibility filter: H–V arcs are
perpendicular, same-family arcs parallel. Before capacity rounding, summed
fractional-target arc weights divided by the original perpendicular event weight
recover original constraint counts. Integrality is checked, not assumed.
Fibers below the minimum are recursively removed until the remaining induced
graph satisfies it. Multiple measured events to the same neighbor count
separately; target-expansion arcs do not. Selection counts and original IDs are
saved. The derived `geometry/catalog.npz` contains identities only, not a new
packed-geometry copy; use original packed geometry for subsequent evaluation.
This changes the objective/cohort, so a normalized factor against a mixed
parallel/perpendicular reference is only a diagnostic, not a like-for-like score.
Optional `minimum_length_base` filters by strict full-polyline arc length before
recursive degree pruning, using the original packed-geometry arc coordinates.
`fiber_flow_length_sweep` accepts `output_root`, optional `experiment_link`,
`lengths_base`, `workers` and a `common` perpendicular-workflow configuration.
It executes independent trials in a bounded process pool. Optional
`common.reference_evaluation` selects cached reference scoring after each solve;
its `comparison_solution` can name the prior no-length-filter solution for
matched-cohort comparisons. Errors are recorded per trial in `results.json`.

`fiber_lemon_full` reuses `graph.npz` and `geometry/catalog.npz` from a completed
full-volume preparation. It does not repeat extraction or run HiGHS first.

```json
{"fiber_lemon_full": {
  "source_root": "/results/full-preparation",
  "output_root": "/results/cost-scaling",
  "scale": 1,
  "reference_mean": 0.27,
  "progress_interval": 20
}}
```

Choose `reference_mean` from the actual small reference report, not the example
above. Capacity quantization is explicit and reported, including discarded arc
weight. Progress and final L1 use **all original arc weights**, including arcs
discarded by quantization. `factor_vs_small` divides that normalized L1 by the
supplied reference mean. Intermediate snapshots evaluate rounded current node
potentials: they are not the intermediate flow objective printed by HiGHS and
are not a convergence percentage. Excess imbalance is separately recorded in
quantized capacity units. Final output requires the ordinary primal/dual,
conservation and capacity certificate for the quantized problem.

The optional progress build generates a local instrumented copy of the installed
LEMON header, preserving its license and algorithm. It inserts read-only hooks
at cost-scaling phase starts and periodically within partial augmentation; it
does not edit system headers or change solver arithmetic. Header source hashes
are recorded, and incompatible layouts fail closed. Reporting is same-thread
and time-throttled, so no unsynchronized access to solver state occurs. A long
internal heuristic may delay a snapshot. Scanning the original graph has some
overhead; no per-arc Python callbacks are used. The backend remains single-core.

### Reference evaluation of cached whole-fiber flow solutions

`fiber_flow_reference_eval` accepts `geometry` (the packed geometry directory),
`solution`, and `output_root`. Optional `gt` supplies `config` (GT tensor/AFV
configuration), `reference` (explicit frozen offsets/exclusions), and `events`
(cached cross-constraint JSON). It reuses the existing sign evaluator and reports
both single-global-offset and explicitly separate component-aligned diagnostics.
`spiral` supplies `reference` (tifxyz), `xyz_scale`, `surface_stride`, `workers`,
`candidates`, `sample_spacing`, `max_distance`, and `fiber_batch`.
Set `allow_solution_subset: true` for a component-only solution drawn from the
full geometry catalog; source IDs and family labels are validated before mapping
to the original geometry. Set `continuous_gt_protocol: true` to additionally
report the same continuous-endpoint, single-global-integer-offset GT metrics as
the IRLS comparison workflow, alongside the explicit-integer reference metrics.

The spiral check samples all fibers uniformly in arc length, including endpoints,
and reuses the nearest-surface projection implementation. Continuous fitted
winding combines the saved integer label, exact polyline seam transport, and
local umbilicus phase. The primary error uses the arc-weighted median pointwise
reference-minus-fit per fiber, then one global median translation. It does not
align disconnected components independently. Reports include coverage, L1,
within-two-windings percentage, largest-component and half-length-covered subsets,
and the historical median-coordinate metric separately. This is approximate
reference association, not optimization or a certificate of reference correctness.
Projection batches are cached for resuming identical requests.

`fiber_flow_reference_analysis` reads `evaluation_root`, writes `output_root`,
and accepts `winding_band_width` (default10). It reports absolute-error
histograms, quantiles, reference-winding bands and low-error subset counts.
Per-band median removal and a held-out linear drift fit are diagnostic only:
they never change the saved solution or replace the single-global-offset score.
It also exports `per_winding.md` and `per_winding_{all,main_component}.csv` using
nearest-integer reference winding IDs. Each row includes the L1-optimal additional
shift, total shift from raw labels, support count, pre/post-shift L1 and residual
p90. Positive additional shifts increase fitted winding. These reference-derived
regional shifts are diagnostic only; they are not applied to the saved solution.

`fiber_flow_reference_rescore` accepts `evaluation_root` from a prior full
projection, a new whole-fiber integer `solution` on the same IDs or a subset,
and `output_root`. It reuses reference geometry exactly: changing a fiber's
constant integer label translates all its projected residuals by that constant.
It validates source identities and H/V classifications, recomputes GT signs,
and reports old/new scores on the identical retained cohort, each with one
global offset. It is not suitable for solutions that change fiber geometry.

`fiber_flow_stop` accepts an explicit `pid` and exact `expected_command`; it
reuses the guarded process-stop helper and never terminates a pattern of jobs.

### Optional parallel network-simplex benchmark

The `fiber_pns_benchmark` workflow compares the external MIT-licensed
[PNS implementation](https://github.com/gokcehan/pns) against a completed
`fiber_lemon_benchmark` report. Supply the upstream `pns.cpp` explicitly; the
workflow does not download or install dependencies.

```json
{"fiber_pns_benchmark": {
  "source": "/dependencies/pns/pns.cpp",
  "comparison_root": "/results/lemon-comparison",
  "output_root": "/results/pns-comparison",
  "thread_counts": [1, 2, 4, 8, 16],
  "scales": [1000, 1], "repeats": 3, "time_limit": 10,
  "avx2": false
}}
```

The adapter reuses upstream initialization and pivot routines. Signed flow
bounds `[-q,q]` become `[0,2q]`, with corresponding node supplies; recovered
flows and potentials receive the same primal/dual, conservation and bound
checks as the other backends. This bypasses the upstream printer, which omits
nonpositive-cost arcs. Every completed result must match the prior certified
quantized optimum. Time-limited trials are marked incomplete, never optimal.
The limit is checked every 64 pivots and may overshoot. Reports include repeated
timings, pricing/tree/potential-update profiles, original-weight metrics and
compiler/source hashes. OpenMP pivot tie-breaking may select different optima.

This optional backend requires an OpenMP C++17 compiler and Boost headers.
Upstream unconditionally includes x86 intrinsics, so other architectures are
explicitly rejected without affecting the portable default backends. AVX2 is
opt-in and CPU-checked. Conservative int32 flow/potential range guards reject
unsupported inputs rather than truncating them; this is not yet an unrestricted
full-volume backend. Parallel pricing does not parallelize every pivot step.

```sh
FIBER_PNS_SOURCE=/dependencies/pns/pns.cpp python -m pytest -q vesuvius/tests/test_fiber_pns.py
```

### Parallel measured-prior preparation

The predicted-only managed L-BFGS solver accumulates independent edge, phase-prior,
Z-prior and global-regularizer objective parts. `lbfgs_prior_batch_size` defaults
to `lbfgs_edge_batch_size`; original sample weights and full-fiber denominators
are preserved. Each part's gradients are accumulated before releasing its graph,
and priors/globals are evaluated once per full objective, not once per edge batch.
This changes floating-point reduction order, not the mathematical objective.
`lbfgs_objective_progress_every` optionally reports first-objective batch progress.

Geometry initialization uses sorted, segmented exact-location topology for zero
merge distance and batched lexicographic interpolation queries rather than
whole-array scans per fiber. Nonzero-distance greedy merging retains its original
semantics. Exact-location duplicates use their shared coordinate directly, avoiding
roundoff from averaging repeated identical coordinates. Straight fixed-length
queries do not construct unused knot-interpolation indices or packed query copies;
the same physical winding/arc evaluation and losses are retained. Startup logs
separately report data loading, geometry construction and query-plan readiness.

`fiber_soft_winding.prepare_samples_geometry` accepts `prior_sample_workers`
(default 1) and `prior_sample_batch` (default 512). It batches all interpolation
queries for each fiber in native NumPy, and uses a bounded process queue when
workers exceed one. Results retain input fiber/sample order and identical weights.
Linux workers share read-only preparation arrays through fork; other platforms
can use spawn, which copies worker initialization state and requires more memory.
Run this CPU preparation before initializing CUDA. The full-volume SGD workflow
defaults to 64 prior workers; `prepare_only: true` stops before optimization and
records compute/save times in `prior_sample_timing.json`. Sample output is published
by atomic rename so interrupted writes cannot be reused as completed caches.

Only reusable implementations, specifications, synthetic tests and minimal usage
examples belong in the shared tree. Dataset/scan names, snapshot IDs, historical
trial names, tuned winner configurations and experiment-result tables do not.
Keep those artifacts externally, for example under ignored `experiments/docs/`
and `experiments/data/`. Workflows must take experimental choices as explicit
inputs; report metadata must describe those inputs, not a historical experiment.
Changing a request requires a new output root when provenance checks reject reuse.

```sh
python -m pytest -q vesuvius/tests/test_fiber_*.py vesuvius/tests/test_paired_*.py
```

Tests require the configured Vesuvius import path. GPU tests skip without CUDA;
the two-rank CPU Gloo test needs local socket access. Preparation/extraction and
full GPU solves require user-provided data and native binaries. Old tests that
required the deleted width prior/fallback geometry were updated to assert the
current rejection contracts, not used to reintroduce those models.

Read-only real-checkpoint regression (recomputes every full-objective term as well
as all historical scoring metrics):

```sh
FIBER_LAYOUT_REFERENCE_SPEC=/configs/fiber-layout/benchmark.json python -m pytest -q vesuvius/tests/test_fiber_workflows_real_data.py
```

## Route-page import

`fiber_stream_import` imports completed `fiber-stream-attributes-v2` tile exports.
Set `tiles`, `output_root`, `frame_template`, and optionally `workers` (16).
Optional `z_range: [lower, upper]` selects whole fibers intersecting that slab;
omit it or use null for the full volume. The importer verifies completion,
coordinate-space identity, unique source IDs, and source counts, preserves the
native base-voxel geometry and IDs, and sets family to unknown for independent
H/V classification. Tile decoding is parallel with bounded outstanding work;
the shared AFV writer still buffers the selected records before database export.

`fiber_route_import` converts native `vc-fiber-routes-v4-reduced` page exports
into AFV using the shared AFV writer. Configure `routes`, `output_root`,
`frame_template` (an AFV with the independently verified same coordinate frame),
`z_range`, and optional `workers`. It keeps whole intersecting routes, preserves
`points_base` without rescaling, and does not read component indexes as fibers.
Family labels are unknown until the normal/H/V preparation stages are run.

## Registered AFV comparison

`fiber_afv_transform` applies a supplied XYZ affine registration to AFV curves.
Set `afv`, `transform` (published `transformation_matrix` JSON), `invert`,
`fixed_volume`, `source_coordinate_space`, `target_coordinate_space`,
`frame_template`, `z_range`, and `output_root`. It validates frames, preserves
source IDs and whole polylines intersecting the target slab, and recomputes
native-coordinate bounds and lengths. Source metadata and the applied matrix
are retained in output provenance; original H/V labels remain available for
comparison with independently inferred labels.

After the normal/H/V and `fiber_min_flow_selection` workflows,
`fiber_selected_solver_comparison` consumes `selection_root`, `base_config`
(the independent winding model), `gt_root`, `umbilicus`, the hash-bound external
IRLS `solver_source`/`solver_sha256`, `devices`, `gpu_manager`, and `output_root`.
GT fibers are removed before solving. All methods use the same largest
perpendicular-supported connected component, with exclusions reported explicitly.
The existing independent winding model is initialized from radial geometry, not
selection-time winding assignments. LEMON and IRLS use the same fractional-edge
graph; reports include both all-event and common SGD-enabled GT cohorts.
Set `include_sgd: false` to run only LEMON/IRLS winding assignment: no SGD
checkpoint is required, and scores use all eligible GT events (no SGD-based
filter). `exact_lp: false` also skips the optional LP control solve. Both flags
default to true to preserve existing three-way recipes. `prepared_root` can
reuse completed preparation in a new output directory after checking selection,
reference, initialization-config and coordinate-axis bindings; it does not reuse
the old winding solution. Keep the old run directory intact for provenance.
`uv_parameters` controls the separate fixed-label geometry fit, and `uv_methods`
selects `lemon_1000` and/or `irls`. These produce equal-axis-scale TIFF/PDF UV maps.
Keep dataset-specific recipes and outputs in the external experiment archive.

To fit/render an existing graph solution without rerunning winding assignment,
use `fiber_flow_uv_saved` with `tensor_cache`, `solution` (NPZ containing matching
`source_id`, integer `winding` and `component` arrays), source `afv`, `umbilicus`,
`output_root`, `devices`, `gpu_manager`, and UV `parameters`. It selects the saved
largest component, converts aligned endpoint arcs back to unsigned input arcs,
and calls the same fixed-label UV workflow. Input hashes are recorded; labels
are not re-estimated. Only whole unsplit fibers are supported.

`fiber_dataset_union` combines prepared independently classified cohorts. Supply
`sources` (each with `name`, `afv`, `tensor_cache`, saved graph `solution`, frozen
GT `events`, and `normal_solution`), native extraction/orientation templates,
`native_binary`, `output_root` and nested `comparison` settings. Source IDs are
remapped with an explicit reversible mapping; coordinate frames must agree.
Within-source constraints and reference events are retained exactly. Native
extraction plus the existing local normal sweep supplies only new cross-source
constraints, using the inherited inward normal signs and family compatibility.
This requires unchanged source geometry and the same normal sampler/sweep settings
as the original classification. No deduplication, GT-in-solve, winding warm start,
or reclassification occurs. The saved solution selects membership only; its
winding labels are not passed to the combined solve.
# Enclosure-only constraints and frozen-reference normal signs

The `fiber_eval_enclosures` workflow imports a directory listing of annotated
fiber JSONs, preserves full `line_points`, and uses the existing native local-event
extractor and fiber-local normal sweep. It does not fit winding widths or UV.

Set `sign_alignment: "reference_constraints"` and provide
`reference_constraint_alignment` (the `fiber_reference_constraint_signs` workflow
configuration). The reference specifies its `afv`, original oriented `database`,
normal `solution`, and prepared `geometry` cache. Only new signs are fit.
The existing native line/normal construction exports frozen normal curves,
validating their gauge against the original constraint endpoints. The native
extractor's `cross_query_prefix` mode queries new lines against reference lines
only, in parallel, with no reference-reference search. Spatial reference selection
uses conservative block bounds, without nearest-fiber or per-fiber sample caps.
The shared `CurveBank` interpolates both frozen normal curves at actual extracted
constraint base points. A single batched dot-product reduction independently fits
each new sign. `minimum_abs_dot` filters weak evidence and `minimum_agreement`
leaves contradictory signs unresolved. Perpendicular evidence is the default,
matching the original sign stage; `include_parallel` explicitly adds parallel
evidence. Reference signs are never reoptimized. The slow Python nearest-endpoint
lookup implementation was removed; there is no per-sample SQLite normal lookup.

`constraint_mode` selects `ordinary`, `additional`, or `only`. Ordinary native
events are always saved unchanged in the raw/oriented databases. Additional
relations have `constraint_type="perpendicular_enclosure"`, one exact integer
anchor step per H/V pair and two witness event indices into `ordinary_tensors.npz`.
The local relation is same winding from the inner H encounter to its backing V.
The anchor step includes full polyline seam transport AND connector transport.
Neither measured Lasagna distance nor an estimated winding assignment is used
to establish these exact bounds. Opposite signs alone are insufficient: maximum
lower bound must equal minimum upper bound. Contradictory and nonunique pairs are
reported separately. Connected-component outputs include isolates and audit
integer cycle consistency without silently deleting conflicting relations.
Saved `relative_winding` values are a spanning-tree audit gauge, not a solved
assignment for components with conflicts. Disconnected components have independent
integer offsets; the enclosure-only graph does not determine their mutual placement.

Required configuration also includes `url`, `output_root`, `frame_template`,
`umbilicus`, `extract_template`, `orient_template`, and `native_binary`.
An explicit `allowed_source_shape_zyx` permits a known annotation metadata shape
discrepancy only; coordinates are never rescaled or clipped. H/V uses manual tags
where present, otherwise saved automatic tags. Fibers without tags are reported.

Run through the normal `fiber_workflows --config FILE` entry point (or a locally
approved fixed runner). Output `graph.npz` is the selected tagged graph;
`enclosures.json` retains witnesses/rejections; `enclosure_components.npz` and
`results.json` describe the enclosure graph independently of ordinary edges.
# Enclosure-component reference checks

`fiber_enclosure_reference_eval` compares a saved enclosure graph with a frozen
full-volume IRLS solution and a reference tifxyz surface. It does not run a new
solver or modify either reference. Supply `enclosures`, `cross_root`,
`irls_solution`, `irls_geometry`, `reference_normal_solution`, `umbilicus`,
`spiral` (the usual projection configuration), and `output_root`.

The evaluator reuses native new-to-reference cross events and frozen signed
normal caches. It reports perpendicular/parallel distance L1, rounded-step
agreement and geometric-order violations against IRLS, plus arc-weighted
surface winding errors, projection coverage and internal constraint checks.
One L1-optimal translation is fitted per component/reference (integer for the
integer IRLS realization, real for continuous spiral coordinates);
there is no rescaling or per-fiber correction. Internal pair differences are
offset-invariant. Contradictory enclosure components are explicitly flagged:
their spanning-tree potentials are diagnostic, not a consistent solution.
IRLS cross-event metrics and spiral projection metrics have different cohorts
and are not an apples-to-apples ranking of the references. Full counts and
definitions accompany the JSON/Markdown output.
