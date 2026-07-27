# 128 sd2 Single-Output 3D Fiber Config

Prepare `train_s1a_nml_all_128_sd2.json` for a regular single-output 3D fiber
training run instead of the recent direction-conditioned decoder experiment.

Requirements:

- Confirm the trainer still supports the non-conditioned legacy/single-output
  path.
- Update only the 128 sd2 config unless runtime code changes are required.
- The 128 sd2 config should build one seven-channel output: six Lasagna 3x2
  direction channels plus one presence channel.
- The trainer should therefore use the existing `compute_losses(...)` path,
  not `compute_conditioned_losses(...)`.
- Keep current 128 sd2 data, augmentation, BatchNorm, and BF16
  mixed-precision settings.
- Use a run name that does not collide with the conditioned 128 sd2 run.
- Update planning/spec docs if the active config description changes.
