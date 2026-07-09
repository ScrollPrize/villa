# Current Task

Verify and harden prefetch for full-augmentation 2D fiber-strip training.

- Training needs to download addressed base-volume chunks before running, using
  the same full augmentation footprint as actual batch loading.
- Prefetch must account for the oversized source strip, coordinate-space
  geometric augmentation, deterministic per-strip-z augmentation parameters,
  and all configured strip-z offsets.
- Prefetch must remain base-volume-only; Lasagna manifest channels are not part
  of the VC3D base-volume prefetch path.
- Add a convenient training-oriented way to prefetch the samples needed by a
  training run or a specified number of training steps.
- Allow prefetching the full configured training run, for example with
  `--prefetch-steps 0`.
- Add tests proving prefetch chunk requests are generated from the same final
  augmented coordinates used by loading.
- Update specs/docs/local commands so the correct prefetch command is clear.
