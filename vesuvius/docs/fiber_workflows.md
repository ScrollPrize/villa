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
`binary_dir`. Supply `native_regression_binary` to the
normal/axial-prior wrappers to run the native regression executable first.
Other native tools, such as reference replay, must be available separately.

## Preparation and evaluation

| Workflow key | Inputs / result |
|---|---|
| `fiber_json_solve` | Reviewed JSON directory + `volume` id and `volume_shape_zyx` + explicit ordered stages; creates a fiber store (`fibers.vcobj`, each fiber keeping its document `uuid`) and dispatches extraction, normal/HV/direction and layout requests. |
| `fiber_anchored_dataset` | Reference-curated source graph; constructs an anchored dataset with fixed reference events. |
| `fiber_selected_dataset_eval` | `dataset`, `base_config`, `gt_root`, `output_root`; filter unsupported sources, split if configured, initialize, solve and score. |
| `fiber_selected_score` / `fiber_selected_checkpoint_report` | Score completed or published intermediate states without rerunning optimization. |
| `fiber_joint_gt_eval` | `baseline_root`, `base_config`, `gt_root`, `output_root`; attach GT, with `force_gt_enabled: true` or `false`. `report_only` reads completed output. |
| `fiber_gt_inpainting` | Fixed-reference solve and optional post-solve deformation. |
| `fiber_neighbor_expansion` | `seed_solve`, whole `parent_tensors`, `fibers`, `gt_root`, `z_range`, extraction/orientation templates; MAP-select neighbors, re-solve, re-filter per round. |
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
dataset contains `selected.vcobj`, `tensors.npz` and `cross/cross_events.json`.
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
`{fibers}` and `{output_root}` are expanded in request string values; referenced
native configuration files must already contain their own correct paths. No
solver settings, enablement policies, pass counts or GPU reservations are injected.
An empty list converts JSON to a fiber store only. `layout_output` optionally requests the
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
(requires `--fibers`, the source fiber store, and the matching C++ header), `fiber_oriented_normals_validation`,
and `fiber_violation_neighborhoods` (consumes `paired_winding_review` output).

Performance modes restored in `fiber_uv_layout`: `benchmark_pair_fusion`,
`diagnostic_benchmark`, `distributed_suite` and `experiment_matrix`.
`indexing_audit` is restored in soft/density optimization. They reuse the normal
solver algebra; enable them explicitly. No new speed or convergence claim is made.

## Local archive and verification

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
