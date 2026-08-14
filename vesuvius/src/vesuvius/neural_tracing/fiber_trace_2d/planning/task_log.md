# Task Log: stage-parallel fiberlet extraction

## Findings

- The former aggregate sampling timer was misleading. On the fixed Paris4
  interval at 32 threads and 32 curves per batch, 22.72 of 24.09 seconds was
  serial domain/node enumeration and ordered corner insertion. Prediction and
  normal reads took 0.83 and 0.49 seconds.
- The same domain and local nodes were then rebuilt during DP. Preparation was
  therefore both duplicated and serial, while the duplicate DP-side build was
  parallel.
- Larger curve batches reduced aggregate requests only because native corners
  shared by candidates in different batches were requested repeatedly. This is
  an invalid operational dependency: batch size must not change the global
  unique sample population.

## Deviations

- The plan's initial wording retained the search corridor into DP. DP does not
  consume the corridor, so it is built once from the retained domain for node
  enumeration and then released. Retaining dead corridor storage would only
  increase the global prepared-set memory.
- Portable owned-payload estimates are reported instead of OS-specific RSS.
  They cover the retained prepared vectors and global sampled arrays but do not
  pretend to include allocator/hash-table overhead exactly.

## Validation

- Independent plan review completed before implementation. It required strict
  global stage ordering, retained interpolation stencils, reuse of the original
  domain in corridor construction, phase-labelled progress, and independent
  prediction/normal request-order tests.
- `cmake --build volume-cartographer/build --target test_fiberlet_paths
  vc_fiberlets -j32`: passed.
- `volume-cartographer/build/bin/test_fiberlet_paths`: 36 test cases passed.
- `ctest --test-dir volume-cartographer/build --output-on-failure -R
  '(fiber|lasagna_line)'`: 11/11 passed.
- `PYTHONPATH=vesuvius/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
  -q vesuvius/tests/test_view_fiber_presence.py`: 55 passed. Disabling external
  pytest plugin autoload avoids the system environment's missing
  `zarr.testing` plugin; no dependency installation was performed.
- Fixed Paris4 interval `[4800.0154976922377, 5312.0154976922377]`, radius 64,
  32 threads: both `--batch 1024` and `--batch 65536` requested exactly 19,835
  unique voxels and produced 2,011 successful paths from 4,258 searches.
- Three `--batch 65536` runs measured fiberlet-stage wall time mean 1.969 s,
  median 1.965 s, range 1.958-1.985 s. Preparation mean was 1.461 s (median
  1.449, range 1.449-1.484) and DP mean 0.098 s (median 0.097, range
  0.096-0.101). Whole-stage effective CPU use averaged 27.59 cores.
- The directly comparable prior `--batch 32` run took 27.137 s for the
  fiberlet stage: 22.718 s serial preparation, 2.866 s DP, and 605,898
  aggregate requests. The new median is 13.81x faster and uses the true 19,835
  coordinate union (30.55x fewer sampler requests) without numerical or
  artifact changes.
