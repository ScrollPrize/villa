# 3D Augmentation Stream And 2D Spec Parity

Fix the 3D fiber loader/trainer so training augmentations are deterministic but
seeded by the unbounded training iteration/sample stream, not by the bounded
data sample identity. Repeated use of the same CP/data sample must still produce
fresh deterministic augmentation parameters over training.

Requirements:

- Split 3D data sample selection from 3D augmentation seeding.
- Add the 2D-style bounded deterministic sample-prefix behavior for 3D where it
  is missing: `training.max_sample_index` limits CP/data selection only.
- Keep augmentation seeding keyed by the unbounded training stream index even
  when `training.max_sample_index` wraps CP/data samples.
- Fix dense 3D test evaluation so it does not inherit training augmentations by
  default. Add an explicit opt-in key for augmented tests.
- Add 3D training CLI resume support so continuing a run can be done with a
  command-line `--resume /path/to/current.pt` argument instead of editing JSON.
- Match 2D `training.max_steps: 0` semantics in 3D: training should keep
  repeating the deterministic dataset stream instead of converting the run into
  one finite dataset pass.
- Match 2D dense-test sentinel behavior in 3D: `training.test_control_points: 0`
  means evaluate every held-out CP sample once in flat order from zero.
- Match 2D prefetch CLI semantics in 3D, including positive
  `--prefetch-steps` overriding config, `--prefetch-steps 0` selecting all
  applicable CPs, omitted `--prefetch-steps` using config, negative values
  failing clearly, and `training.max_sample_index` bounding the prefetched
  training prefix.
- Align console progress wording for both 2D and 3D so the first 100 training
  steps are printed, then `training.scalar_log_interval` is used.
- Ensure train and test loaders always carry full transformed fiber-line segment
  metadata for visualization/context; only the loss mode decides whether dense
  line supervision or CP-only supervision is applied.
- Review the shared 2D specs and report which non-dimensional requirements were
  ignored or only partially carried over to 3D.
- Update specs/docs/tests for the above.
