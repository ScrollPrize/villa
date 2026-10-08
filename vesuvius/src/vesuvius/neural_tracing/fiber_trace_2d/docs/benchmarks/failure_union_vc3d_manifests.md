# Failure-union VC3D manifests

The completed `full_failure_union_20260921_v3/vc3d_channels` export contains
`<scroll>/inference.lasagna.json` and its `inference.json` provenance for
PHerc0139, PHerc0332, PHerc1299, PHerc1451 and PHercParis4.

Attach the relevant `inference.lasagna.json` as fiber predictions in VC3D.
Relative group paths point to the sibling presence/nx/ny Zarr arrays at level
1. Source manifests supply the exact original base dimensions (including odd
dimensions), `source_to_base=1` and `scaledown=1`, giving two base voxels per
prediction voxel. These manifests describe partial inference coverage; they
do not add sheet-normal predictions or additional pyramid levels.

The normal failure-union export now writes these manifests automatically.
Refresh only these small metadata files in a completed export with:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py failure-union --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/failure_union_full_20260921_v1.json --export-manifests-only
```

No inference or extraction runs in this mode. Export checks relative paths and
all channel shapes against source base dimensions and scale. Paris4 uses the
subset's recorded source manifest but binds to the local subset channel paths.
