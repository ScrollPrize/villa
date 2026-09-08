# Plan: native lower-resolution Fiber3D U-Net output

## Architecture

1. Add an explicit `model_3d.output_stride` setting to the shared 3D U-Net.
   Keep stride 1 as the compatibility default and initially support isotropic
   strides represented by an encoder skip resolution.
   Define one shared lattice helper: output index `k` aligns to input index
   `k * output_stride`, and output shape is ceil-divided per axis, matching the
   padded odd-kernel strided-convolution lattice.
2. For stride 4, construct and execute only the decoder prefix required to
   reach the quarter-resolution encoder skip. Do not construct or run the
   half- or full-resolution decoder stages.
3. Attach the existing output activation/head semantics directly at that
   decoder resolution. Preserve the exact stride-1 module structure and state
   keys so existing checkpoints remain loadable.
4. Retain GroupNorm in the benchmark configuration; do not change the normal
   configuration default or silently reinterpret old checkpoints.

## Training targets and losses

1. Materialize presence, validity, and sparse direction targets directly on
   the model output lattice. Coarse positives use source-voxel-cell occupancy:
   a coarse cell is positive when it contains a positive source-grid sample.
   CP targets always include their nearest in-bounds coarse cell if a
   sub-voxel radius otherwise produces none. Dense lines retain source-lattice
   rounding before deterministic coarse-cell deduplication.
2. A coarse validity cell is valid only when all of its in-bounds source voxels
   are valid. A negative-loss cell is eligible only when its complete source
   footprint is inside the configured edge margins; derive lower and upper
   bounds independently for non-divisible margins and shapes.
3. Keep deterministic sparse-direction deduplication and existing loss math.
   Validate model-output and target shapes explicitly with actionable errors.
4. Apply the same output-stride handling to training, test, and visualization
   paths through one shared target-materialization API.

## Benchmark

1. Extend the existing two-GPU profiling launcher with an explicit output
   stride; do not create a second training implementation.
2. Compare stride-1 GroupNorm with stride-4 GroupNorm using the same S1a
   128-cubed inputs, batch 8/rank, BF16, eager execution, optimizer, seeds,
   captured per-rank batches, 30-step warmup, and 90-step measured window.
3. Run both replayed-batch compute and real-sample measurements on GPUs 0 and
   1. Report global/per-GPU samples/s and iteration mean, median, and p95.
4. Record model parameter count, output shape, and peak allocated GPU memory so
   the source of the speedup is auditable. This phase makes no convergence or
   quality claim.

## Inference compatibility

1. Store output stride in checkpoints through the existing saved config.
2. Keep product inference scaledown as the requested total factor. Expose the
   checkpoint's native model stride separately and define residual downscale as
   `requested_total / native_stride`. Build blend weights on the native lattice
   before applying residual downscaling; persisted output scale remains the
   requested total factor.
3. Reject incompatible combinations where native model stride and requested
   output scale cannot be represented exactly. Keep old stride-1 checkpoints
   and inference behavior unchanged. Apply the same composition or an explicit
   rejection in Trace2CP/native field consumers; no consumer may silently
   downsample twice.

## Tests

1. Test stride-1 state keys and output shape against the compatibility path.
2. Test stride-4 construction, omitted decoder stages, output shape, direct
   target materialization, loss/backward, and config validation on CPU. Target
   cases include shifted CPs, radius below stride, short/diagonal lines,
   non-divisible margins/shapes, invalid source regions, and duplicate coarse
   indices.
3. Test checkpoint config recovery and inference scale composition without a
   full-volume run.
4. Test total/native/residual scale composition for legacy 1/4, native 4/4,
   native 4/8, and rejection of native 4 with requested 1 or 2. Require
   fixed-seed stride-1 output equality and strict legacy checkpoint loading,
   not state-key comparison alone.
5. Run focused Fiber3D model/training/inference tests and `git diff --check`,
   followed by the authorized two-GPU benchmarks.

## Spec update

Document the aligned ceil-shaped lattice, occupancy/validity/margin semantics,
`model_3d.output_stride`, checkpoint behavior, and total/native/residual
composition for whole-volume and Trace2CP inference.

## Docs update

Add the architecture and measured stride-1/stride-4 results to the Fiber3D
two-GPU training profile document, including the speed-only/quality caveat.

## Changelog update

Add one concise entry for native lower-resolution Fiber3D output once the
implementation and measurements are complete.
