# Gray top/bottom bands: physical coverage diagnosis

Read actual reference z.tif (not merely metadata), scale4 to base voxels:
reference Z16019.7207–70987.4688. Retained AFV samples Z1549.0820–75751.8438.
No projection or solver rerun. Diagnostic approved runner config:
report_reference_projection_coverage.json; output:
data/uv_fullpass151_v1/projection_coverage/coverage.json.

20.9776%retained sampled fiber length is gray. Of gray length:
91.9741%is outside reference global physical Z bounds;
6.6601%has nearest unrestricted surface within200vx but fails chosen-winding
projection. These are diagnostic categories, not an exclusive partition.

Thus large top/bottom bands predominantly reflect absent physical reference
coverage. Full reference image height is flattened V, not CT-Z0 to scan end;
rendering both images1000px high visually obscures this distinction.

|Physical Z band|Nearest surface accepted %|Chosen winding accepted %|
|---|---:|---:|
|0–8000|0|0|
|8000–16000|0.0110|0.0072|
|16000–24000|98.5820|91.4891|
|24000–32000|99.8827|98.0690|
|32000–40000|99.9680|99.4361|
|40000–48000|99.9407|99.3325|
|48000–56000|99.8969|99.2107|
|56000–64000|99.9394|99.4630|
|64000–72000|90.1034|89.2705|
|72000–80000|0|0|

Why the upstream supplied surface was limited to this physical extent is not
established here; its actual coordinates, not just a projection filter, are limited.
