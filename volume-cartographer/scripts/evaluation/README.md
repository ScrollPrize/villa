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
- `target_bboxes`: Non-empty list of boxes as `[x, y, z, sx, sy, sz]` (origin + size) used to rank starting patches by face coverage
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
- Patch selection ranking: boxes_hit across `target_bboxes`, then face coverage, then overlap count, then area
- `starting_traces_selection_mode`: Set to `"mask"` to enforce per-bbox coverage diversity when picking starting patches (default `"none"`)
- `starting_traces_top_m`: When using mask selection, only diversify within the top M ranked candidates (default 40_000, `0` = use all)
- `starting_traces_max_per_mask`: Hard cap on how many patches to take per exact bbox bitmask when in mask mode (default `1`, `0` = uncapped)

### Metrics and logging
- `trace_ranking_metric`: Metric name to rank traces by (e.g. "winding_valid_fraction"); assumes higher is better
- `num_best_traces_to_average`: Number top-ranked traces to average for final metrics
- `wandb_project`: Weights & Biases project name for logging (optional). Only the wandb summary and config are written, not per-step metrics
