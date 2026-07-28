# Plan: Native 3D Trace2CP Scaled Inference Field

## Implementation

1. Add `NativeTrace2CpConfig.inference_scaledown_power` and a CLI flag
   `--inference-scaledown-power`.
2. Validate the power at cache construction:
   - power must be a non-negative integer;
   - factor is `1 << power`;
   - every inference patch axis must be divisible by the factor;
   - `core_margin_voxels` must be divisible by the factor;
   - the native patch must still be larger than `2 * core_margin_voxels`.
3. Keep model input loading unchanged: read the requested native selected-level
   block with `CoordinateSampler.sample_block_zyx(...)`, preprocess it, and run
   the model exactly as before.
4. Before storing inferred products in the CPU field cache, apply 3D box
   downsampling with `torch.nn.functional.avg_pool3d` to each raw product tensor.
   Downsample the validity mask with the same factor and mark a scaled output
   voxel valid only when all source voxels in the box were valid.
5. Store scaled cached blocks with:
   - scaled output spatial shape;
   - scaled crop margin;
   - native-coordinate sample origin and sample spacing equal to the scaledown
     factor.
6. Update point sampling so it converts native selected-level point coordinates
   to cached scaled-grid coordinates via `(point - sample_origin) / spacing`
   before `grid_sample`.
7. Include the scaledown power/factor in startup output and JSON summaries.

## Spec Update

Add a native 3D Trace2CP spec entry documenting `--inference-scaledown-power`,
the power-of-two factor, box filtering, divisibility checks, scaled trusted
core margin, and default no-op value.

## Docs Updates

No standalone docs update is needed beyond the specs: this is a CLI option on
the native 3D Trace2CP tool and does not change training or data loading.

## Tests

Add focused tests for:

- CLI/dataclass default remains exponent `0`.
- Invalid scaledown combinations fail loudly.
- Scaled cache storage uses box-filtered raw output and routes point sampling
  through the scaled grid.
- Default exponent `0` preserves existing cache shapes.

Run the relevant 3D Trace2CP tests and `py_compile`.

## Changelog

Add one changelog line for the native 3D Trace2CP scaled inference option.
