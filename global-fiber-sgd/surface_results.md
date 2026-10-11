# First H-anchored surfaces — saved20-sweep maps

The tifxyz files are loadable-format outputs for inspection, **not clean final
surfaces**. There are many holes, large inter-vertex jumps and twisted quads.
No surface render was made. Existing100-sweep optimizations were left running.

## Outputs

Root:`data/surfaces20_v1/`, with `ccw/` and `cw/` tifxyz directories.
Each contains x/y/z.tif(float32 LZW),mask.tif(0/255),meta.json,
surface_statistics.json and validation.json. Roughly12MiB each.
Updated20-sweep fiber plots: `data/uv_categorical_v1/visualization20/{ccw,cw}/layout.tif`
and layout.svg. TIFF height1000px, equal voxel scale, winding boundaries.

|Metric|CCW|CW|
|---|---:|---:|
|Selected joint>50% pieces|294,927|294,020|
|H pieces|152,913|152,196|
|V pieces|142,014|141,824|
|Grid rows × columns|790 ×43,661|803 ×43,168|
|Valid vertices|584,950|576,328|
|Valid/full bounding-grid vertices|1.6959%|1.6626%|
|Complete quads|332,497|325,793|
|Twisted or degenerate quads|19,807|19,338|
|Median3D edge length(basevx)|157.90|161.03|
|P95edge length(basevx)|2,877.33|2,926.86|
|Maximum edge length(basevx)|21,258.96|19,350.43|
|Construction+write(seconds)|44.83|50.59|

Bounding-grid coverage is not a percentage of true papyrus area. The unfolded
bounds include unsupported space and all100winding hypotheses.
7.86M/7.84Mgrid points have boundedH brackets,1.14M/1.13Malso have spanningV
brackets; invalid/opposed normal support removes another0.55M/0.56M. No claim
that these diagnostics prove the interpolation or assignment anatomically right.

## Method

Grid100base voxels, max1000base voxels per side inUV. Independent H/V radius
overrides supported. Search at exact orthogonal coordinate, real curve span;
V must span selectedH interval. Transfer V's normal-projected deviation from
its own endpoint chord onto ruled H surface. Zero deviations atH endpoints.
Original AFV curves are mapped using shared hard_geometry and signed source
arc. No 3D geometry is derived from visualization straight lines.
Normals are regular16vx native samples, same original sweep merged with all
constraint endpoints, then stage1 inward signs. All14,736,670stored endpoint
normals matched. Invalid interpolation intervals and opposedH normals yield
holes. No geometry repair or smoothing was silently applied.

Tifxyz scale=.01,.01 means100vx per grid sample, not100. UV origin is stored in
metadata; XYZ remains native Paris4/20260411134726base coordinates.

## Reproduce

```sh
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_export_normal_curves.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_surfaces20.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_surfaces20_audit.json
```

Use fresh output paths to reproduce extraction/construction; completed outputs
are protected. Audit is repeatable and checks source hashes plus TIFF contents.
Normal export106.04s wall,37.06s native sweep,512,698fibers/45,649,609samples.
Native build RelWithDebInfo; surface constructionCPU NumPy, one run each.
Interactive vc3d loading was not tested; format contract and roundtrip verified.
