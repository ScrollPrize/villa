# Surface Tracer Evaluation

This evaluates the overall surface tracer pipeline by running seed- and expansion-patch growing, surface tracing, winding number estimation, and metrics.

## Usage

```bash
python eval_surface_tracer.py <config_file>
```

## Configuration Options

### Data and paths
- `surface_zarr_volume`: Path to surface predictions (.zarr)
- `z_range`: [min_z, max_z] - range of slices to process; restricts seeds, tracer and metrics
- `wrap_labels`: Path to ground-truth wrap-labels JSON file
- `bin_path`: Path to compiled VC3D executables
- `out_path`: Output directory for results

### Patch growing
- `use_existing_patches`: Skip seed/expansion phases, use existing patches from `out_path/patches`
- `existing_patches_for_seeds`: Path to existing patches whose seeds will be re-used here; or json file produced by get_seeds_from_paths
- `max_num_seeds`: Maximum number of seed points to process
- `num_expansion_patches`: Number of expansion runs to perform
- `seeding_parallel_processes`: Number of parallel processes for seeding
- `vc_grow_seg_from_seed_params`: Parameters for seed growth; children `seeding` and `expansion` are each a copy of standard `vc_grow_seg_from_seed` params json; only `mode` field is overridden

### Surface tracing
- `min_trace_starting_patch_size`: Minimum area for tracer start patches; we select arbitrarily from those exceeding this threshold and do one trace from each
- `num_trace_starting_patches`: Maximum number of patches to use as trace starting points
- `vc_grow_seg_from_segments_params`: Parameters for surface tracing; same as standard `vc_grow_seg_from_segments` params json; only `z_range` is overridden

### Metrics and logging
- `trace_ranking_metric`: Metric name to rank traces by (e.g. "winding_valid_fraction"); assumes higher is better
- `num_best_traces_to_average`: Number top-ranked traces to average for final metrics
- `wandb_project`: Weights & Biases project name for logging (optional). Only the wandb summary and config are written, not per-step metrics
