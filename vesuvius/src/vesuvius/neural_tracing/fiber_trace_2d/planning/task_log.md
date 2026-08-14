# Task Log: straightened fiberlet DP and scored graph joins

## Findings

- The current 0.5-prediction-voxel grid is finer spatially but retains only the
  26 world-axis immediate-neighbor directions. Its unoriented angular covering
  radius is about 27.55 degrees, exceeding the hard 25-degree prediction gate.
- Alternating discrete directions incurs multiplicative alignment loss, while
  a constant nearest diagonal avoids that turn term. This can favor a drifting
  quantized diagonal over the correct average fiber direction.
- Existing graph transitions enforce only a strict 45-degree angle. Their angle
  and local prediction/normal evidence do not contribute to beam-search loss.
- A straight chord-normal layered domain would retain an inappropriate straight
  baseline. The corrected design follows a cubic-Hermite centerline fitted from
  both anchor directions and uses normal planes with a parallel-transported
  transverse frame.
- Independent review required explicit incoming-step DP state, actual mapped
  edge lengths, deterministic frame/error rules, globally deduplicated native
  interpolation corners, a shared complete metric wrapper, exact anchor samples,
  and exactly-once graph transition accounting. These are incorporated.
- The old fiberlet free angle is 45 degrees, which would make tangent/normal
  smoothness zero at every admissible graph join. The curved floating-point
  lattice removes the quantization rationale, so DP and join defaults change to
  the greedy tracer's zero-degree free angle.
- The current Paris4 half-grid baseline is 424.67 s wall, 1,507.13 s user, and
  699,060 KiB peak RSS. It preloads 33,527 native voxels and 221,964 derived DP
  nodes, accepts 2,062 of 8,706 searched fiberlets, and produces a 486-node,
  2,062-edge graph. The replay exceeds the 20-base-voxel error threshold,
  reaches 62.19 base voxels maximum error, and completes diagnostic postroll.
- The curved-domain replay accepts 3,684 of the same 8,706 searched pairs and
  produces a 487-node, 3,684-edge graph with 61,008 scored transitions. Its
  14-edge route reaches the end of the 1,023.97-base-voxel reference interval;
  maximum, mean, and final reference errors are 11.00, 6.20, and 9.52 base
  voxels. No route point crosses the 20-base-voxel failure threshold.
- The curved-domain run takes 66.88 s wall, 414.65 s user, and 501,516 KiB peak
  RSS. This is a 6.35x wall-time reduction and 28% lower peak RSS than the
  half-grid baseline. It samples 38,008 native prediction voxels and evaluates
  about 8.54 million candidate-local floating positions. Candidate-domain
  enumeration and interpolation are now the primary optimization opportunity.
- Route loss is split into 22.13 edge loss and 3.49 transition loss, totaling
  25.62 (0.199 per prediction voxel). Thirteen join costs are accounted exactly
  once for the fourteen selected graph edges.

## Deviations

- None.

## Validation

- Independent review completed before implementation; its state, interpolation,
  determinism, and exactly-once join-accounting requirements were incorporated.
- Built with 32 jobs:
  `cmake --build volume-cartographer/build -j32 --target vc_fiberlets test_fiberlet_paths test_fiber_replay`.
- `ctest --test-dir volume-cartographer/build --output-on-failure -R fiber`
  passes all 9 selected tests.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src pytest -q
  vesuvius/tests/test_view_fiber_presence.py` passes all 55 tests. Plugin
  autoload was disabled because this host's unrelated `zarr` pytest entrypoint
  imports the absent `zarr.testing` module.
- `PYTHONPATH=vesuvius/src python -m py_compile
  vesuvius/src/vesuvius/scripts/view_fiber_presence.py` passes.
- The strict replay loader accepts the benchmark artifact and reports
  `reference_end`, 71 route points, and all 5 anchor diagnostic stages.
- Paris4 benchmark command, run from `data/workdir3`:
  `/usr/bin/time -v $SRC/volume-cartographer/build/bin/vc_fiberlets
  fiberlet-replay $FIBER
  $VES/data/fibers/david/Paris4_fibers/dj_20260805T025256484_000003.json ./
  --normal-manifest $NORMALS --along 512`.
- Benchmark artifact: `runs/4bbb9d5d36688f3b`.
