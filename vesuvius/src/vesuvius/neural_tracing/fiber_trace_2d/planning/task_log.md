# Training-feedback GT refinement — 2026-09-15

- Approved opt-in: current detached raw presence, immutable original tether,
  40/80 ellipse, normal-plane projection, tapered 1/8 per-iteration cap,
  arclength resampling. Train/test reuse updated geometry through existing queues;
  model checkpoints reference matching geometry archives. Input files untouched.
- Independent review corrected output-stride lattice, unoriented normal metric
  interpolation, material quadrature and legacy model-config validation. Shared
  vc3d/C++ normal sampler; no Python/Zarr loading fallback or protocol changes.
- Solver benchmark: initial sparse Torch mapping 7.33 ms/patch, native scalar
  2.98 ms; batched 16-patch min/median/max 34.43/34.52/34.65 ms, 464 patches/s
  on ONE CPU thread. Synthetic 24-node curves, four iterations, affine+smooth,
  ten repeats. Excludes normal loading, D2H, commits and snapshot writes.
- Approved runner `vesuvius/scripts/run_fiber3d_cooperative_tests`:
  refinement/sampling-regression/unified-managed/cuda-refinement suites. Results
  `/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260915_093453_577298`:
  12 CPU passes, one CUDA skip; exclusive manager-gated CUDA 1 pass. Full rerun
  `20260915_093658_395692`: sampling 317 passed/11 skipped/2 xfailed;
  managed lifecycle 57 passed/2 skipped. Redundant CUDA repeat was cancelled
  while waiting behind another nonpausable task; earlier CUDA check passed.
  `20260915_093936_870829`: final refinement + V3 export roundtrip 13 passed,
  one CUDA-only skip. `git diff --check` clean.
- Fixed an existing constructor-failure test fixture to provide an actual config
  file before its injected failure; no manager production code changed.
- Explicit limitations/deviations: native CPU batched solver, not GPU optimizer;
  full training overhead not yet measured. Test feedback commits per wave so it
  survives pause (raw/EMA use same prepared targets within each wave). Nearest
  connected in-patch visit only. Invalid normals skip refinement. Prefetched GT
  stays immutable, stale overlaps rejected, original sampling weights fixed;
  multiworker timing is not bit-identical. Unmanaged multi-GPU, conditioned models,
  historical test sets and output-upsample testing reject enabled refinement.
- Initial implementation did not launch training; settings documented in
  docs/training_refinement.md.
- User subsequently authorized launch: new refinement config copies raw snapshot
  settings from run 5scroll_json_eligible_128_sd1_16x8_ema_20260914_105436,
  current.pt update11800. Preserves LR0.01, optimizer/EMA, effective batch128 and
  eligible JSON data; adds existing normals manifests and refinement.enabled.
  Approved command: `vesuvius/scripts/run_fiber3d_experiment production-start`.
  Launch root2761852, manager task1be334ed-b1a4-4e6b-84ad-1c6c30ff1f71;
  log `/ephemeral/hendrik/fiber3d_train2/runs/.launches/5scroll_json_eligible_refinement_128_sd1_16x8_ema-20260915_095338/train.log`.
