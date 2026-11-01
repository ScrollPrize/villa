# Normal-grid probability threshold and resume

Generate normal grids directly from an 8-bit probability Zarr using
`vc_gen_normalgrids generate -i probabilities.zarr -o grids --threshold 51`.
Voxels at or above the threshold count as surface. The default, 1, preserves
the previous behavior of accepting any non-zero voxel.

Resuming into the same output directory requires the same threshold on every
shard. Existing metadata without a threshold field means 1. A mismatch stops
before creating output directories, rewriting metadata, or generating grids;
use a fresh output directory to try a different threshold.

With an existing configured build and dependencies, run from `volume-cartographer`:

```sh
ninja -C build vc_gen_normalgrids
VC_TEST_REQUIRE_NETWORK=1 ctest --test-dir build --output-on-failure -R '^test_normal_grid_threshold_resume$'
```

The regression downloads one real PHerc0172 level-5 chunk using Python's standard
library. It checks that threshold changes leave metadata and grids untouched on
both shards, and that same-threshold resume restores a missing grid. Network
failures skip the test unless `VC_TEST_REQUIRE_NETWORK=1` is set. To use an
existing local real-data crop instead:

```sh
python3 core/test/test_normal_grid_threshold_resume.py build/bin/vc_gen_normalgrids --volume /path/to/crop.zarr
```
