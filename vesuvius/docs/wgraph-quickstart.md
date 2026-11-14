# Fiber winding workflow: commands and example runtimes

Use `vc_wgraph` to run each stage separately. In a configured development environment, `python -m vesuvius.wgraph` is equivalent. The native importer/extractor, `vc_object_store` bindings, and CUDA solver dependencies must be available in that environment.

## Stages and commands

Run these in order, using a workflow profile such as `settings.json`:

```sh
vc_wgraph import settings.json -i input.afv -o fibers.vcobj
vc_wgraph extract settings.json -i fibers.vcobj -o constraints.vcobj
vc_wgraph hvb settings.json -i constraints.vcobj -o classes.vcobj
vc_wgraph sign settings.json -i classes.vcobj -o signs.vcobj
vc_wgraph winding settings.json -i signs.vcobj -o windings.vcobj
vc_wgraph uv settings.json -i windings.vcobj -o uv.vcobj
```

| Stage | What it does | Result |
|---|---|---|
| `import` | Imports traced polylines through the sharded bulk importer. | Fiber geometry store. |
| `extract` | Finds/refines parallel and perpendicular candidates, samples prediction data, and stitches piece-local normal orientations. | Standard measured constraints and outward-normal prior report. |
| `hvb` | Classifies fibers as horizontal, vertical, or broken using wrapped IRLS. | Per-fiber classification and relation-consistency metrics. |
| `sign` | Aligns whole-fiber normal signs with classification fixed, using wrapped IRLS. | Per-fiber signs and disagreement metrics. |
| `winding` | Solves the fractional-target L1 graph using GPU IRLS/CG. | Continuous winding potentials, integer readout, components, and residual metrics. |
| `uv` | Fixes winding assignments and fits straight, fixed-length fiber layouts and winding widths using Gauss–Newton/CG. | Sampled UV positions, fitted widths, assignments, and meeting-error metrics. |

These are separate commands, not one automatic pipeline invocation. A stage creates a new output store; it does not modify its input. Derived stores reference upstream stores rather than copying their geometry. Keep those referenced stores accessible. To rerun a later stage, reuse the existing upstream output and choose a new output path; there is no need to repeat import or extraction.

### Accepted fiber inputs

`vc_wgraph import` currently dispatches to `vc_object_import_afv`: its source must be an AFV database, not an arbitrary fiber directory. The rest of the pipeline consumes vcobj fiber stores and is not AFV-specific. Existing non-AFV adapters produce the same geometry-store input for `vc_wgraph extract`:

| Input | Existing entry point | Notes |
|---|---|---|
| AFV database | `vc_wgraph import` | Native parallel/sharded bulk import. |
| Compound stream-tracer tiles (`fiber-stream-attributes-v2`) | `fiber_stream_import.run(config)` | Completed tile export with its run/summary metadata; optional whole-fiber Z-slab selection. |
| Reduced neural-tracer route pages (`vc-fiber-routes-v4-reduced`) | `fiber_route_import.run(config)` | Route manifest and compressed pages; whole-fiber Z-slab selection. |
| Neural full-volume tracer `fibers.vcobj` | Start at `vc_wgraph extract` | Use the tracer's native output directly, including an unsealed history-enabled store. |
| Existing compatible vcobj fibers | Start at `vc_wgraph extract` | No format conversion or reimport needed. |
| Application-owned polylines/records | Shared initial-fiber / native initial-import APIs | Requires a source adapter; not an additional auto-detected CLI file format. |

The two tracer adapters are Python modules under `vesuvius.neural_tracing.fiber_trace_3d`, not selectable input backends of `vc_wgraph import`. They use the shared native bulk writer but currently materialize records in Python first; their preparation cost is not covered by the AFV runtime below. Both preserve source-derived UUIDs and import unknown family labels for subsequent H/V classification. See [tracer adapter settings](fiber_workflows.md#route-page-import) and the [source-independent native construction API](../../vc_object_store/docs/initial-import.md). Coordinate frames must match; a file containing polylines alone is not sufficient to infer the volume binding.

### Neural tracer vcobj workflow (skip import)

The neural full-volume tracer's native `fibers.vcobj` is already the fiber geometry store. Start at extraction, then use the same downstream commands:

```sh
vc_wgraph extract settings.json -i tracing-output/fibers.vcobj -o constraints.vcobj
vc_wgraph hvb settings.json -i constraints.vcobj -o classes.vcobj
vc_wgraph sign settings.json -i classes.vcobj -o signs.vcobj
vc_wgraph winding settings.json -i signs.vcobj -o windings.vcobj
vc_wgraph uv settings.json -i windings.vcobj -o uv.vcobj
```

This does not change the tracer output or run AFV import. The source may be sealed, or unsealed with history enabled. Extraction captures one coherent source publication and publishes its actual consumed pin. Later source commits do not alter the derived result: downstream reads resolve the retained older metadata, geometry and membership. Missing required history fails explicitly. Parallel geometry loading still rejects mixed source publications during capture; it never combines old and new fibers. Unsealed history-off sources are rejected because their consumed versions are not retained. History-enabled scopes use checked geometry reads rather than the history-off trusted fast path.

## Configuration

The positional argument accepts either a JSON file or a named profile. `-i/--input` and `-o/--output` override store paths. Relative paths are resolved from the current working directory. Profiles hold dataset-specific coordinate bindings, prediction locations, umbilicus input, solver settings, and GPU-manager requests; the built-in defaults alone do not specify a dataset.

Use this workflow-profile structure, filling the stage settings for your inputs:

```json
{
  "version": 1,
  "common": {},
  "stages": {
    "import": {},
    "extract": {},
    "hvb": {},
    "sign": {},
    "winding": {},
    "uv": {}
  }
}
```

Import needs the source volume/coordinate binding; extraction additionally needs the normal and winding prediction sources and umbilicus configuration. H/V classification can use `native_config` pointing to the resolved extraction configuration for geometry-derived axial priors. Sign alignment can use `priors` pointing to extraction's outward-normal prior report. Winding and UV also need the measured-geometry/umbilicus coordinate binding. See [the full configuration guide](wgraph.md) and [fiber workflow formats](fiber_workflows.md) for the supported settings.

Inspect configuration without loading data or launching solvers:

```sh
vc_wgraph profiles
vc_wgraph extract settings.json -i fibers.vcobj -o constraints.vcobj --print-config
```

For downstream stages, `workers` controls native CPU preparation, `devices` selects the GPU, and `gpu_manager` configures its managed grant. UV optimizer settings belong under `parameters`, including `uv_optimizer`, `steps`, and `log_every`. Import/extraction retain their own native worker/shard settings. GPU grants are non-exclusive by default. If multiple eligible upstream runs exist, select them explicitly; see the full guide for the run-UUID selectors.

## Example full-volume runtimes

Example workload: approximately 3.54 million fibers, 384.7 million polyline points, and 135.1 million constraints (108.7 million parallel and 26.3 million perpendicular). Native code used a Release build. Downstream preparation used 32 CPU workers and one managed CUDA GPU with a non-exclusive 32-GiB reservation. These are single measured runs, not repeated benchmark averages or runtime guarantees; import/extraction were completed earlier and reused for the downstream run.

| Stage | End-to-end wall | Output size |
|---|---:|---:|
| Import | 1 min 14 s | 19.83 GB |
| Extract | 12 min 29 s | 15.93 GB |
| H/V/broken | 6 min 36 s | 0.69 GB |
| Normal signs | 3 min 19 s | 0.66 GB |
| Winding | 2 min 56 s | 0.67 GB |
| UV | 42 min | 2.20 GB |

Output sizes are decimal GB and describe each output file, not a self-contained copy of all referenced inputs. End-to-end times include preparation and result writing. The downstream stages total about 55 minutes. Adding the separately measured import/extraction gives about 69 minutes, not a newly measured uninterrupted pipeline run. UV preparation alone took 2 min 7 s. Numeric arrays passed directly from native RAM buffers to NumPy/PyTorch; no tensor-file staging was used.

The example UV run completed its configured iteration cap, **not convergence**. Its objective fell from 154.34 to 0.935, with no integer winding assignments changed. Final same-sheet U meeting error was mean/median/p95/max 32.28 / 6.85 / 135.07 / 14,953.99 voxels, or 0.00136 / 0.00017 / 0.00584 / 1.37946 windings using the fitted width mapping. V meeting error was 18.72 / 5.92 / 77.72 / 1,809.03 voxels. Large outliers remained; these runtimes do not imply a fully converged fit.

## Progress and outputs

Each command prints `WGRAPH_STAGE_COMPLETE` with its input/output paths, overall wall time, and the result run UUID where applicable. Import defaults to `<output-stem>.import.json`; extraction defaults to `<output-stem>.extraction.json` and `<output-stem>.outward-priors.json`. Solver work/report files default to `<output-store>.work`, unless `output_root` is configured. Consult the stage summary/history for stop reasons, objective terms, iteration counts, memory, and residual statistics rather than assuming that a successful exit means convergence.
