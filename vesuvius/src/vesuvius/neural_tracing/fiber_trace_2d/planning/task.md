# S1A NML All-Data Training Config

Rename and clean up the S1A NML config so it clearly loads all available S1A
NML files from the local `fiber_vols` source as training data.

Requirements:

- Rename the config away from the generic/test example name.
- Use the full S1A NML glob:
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fiber_vols/fibers_s1a_*.nml`.
- Keep the PHercParis4 base volume, selected scale, Lasagna manifest, and
  existing S1A affine transform settings.
- Avoid silently mixing the old JSON held-out test dataset into this S1A config.
- Keep `training.max_sample_index: 0` so training/prefetch can cover the full
  deterministic S1A CP stream.
