# Fiber-local normal orientation — 2026-10-08

Native second pass over the existing extraction, not a re-extraction. Uses the
same Paris4_v1 AFV and Lasagna normals; no cross-fiber alignment or BP/SGD solve.

## Method

Original AFV point order; regular arc sampling every16base voxels including both
ends, merged with exact constraint base points. Flip current normal iff dot with
last valid oriented normal is negative. First valid normal anchors an arbitrary
gauge. Store signed normals only at constraints, plus independent fiber
`normal_sign=+1`. No normal arrays along fibers are persisted.

Schema, semantics, limitations and exact build/run/validation commands are in
[constraint_database.md](constraint_database.md#second-pass-fiber-local-signed-normals-schema2).
The normal sampler is parallel (128workers); batches contain256fibers. Build is
RelWithDebInfo, GCC13, workspace-native runner. Each timing is one run, not a
statistical performance comparison.

## Pilot

6,612fibers;19,889constraints;831,845samples. Sweep1.0476s, native second-pass
total1.5180s, runnerwall2.1802s. Endpoint flips13,611/39,778(34.2174%). No invalid
samples. All old measurement fields and source-fiber fields exact; all endpoint
axes preserved up to sign, integrity/FKs valid.142native test cases passed.

Pilot artifact: `data/oriented_normals_v1/pilot.afc.sqlite`.
Validation: [oriented_normals_pilot_validation.json](oriented_normals_pilot_validation.json).
Runner provenance/logs: `data/oriented_normals_v1/pilot_runner/`.

## Full volume

Output: `data/oriented_normals_v1/full.afc.sqlite`.
Input remains `data/raw_winding_v1/full.afc.sqlite` unchanged.
Runner provenance/logs: `data/oriented_normals_v1/full_runner/`.

| Metric | Full volume |
|---|---:|
| Fibers |512,698|
| Constraints (unchanged) |7,368,335|
| Exact constraint endpoint normals |14,736,670|
| Total sweep samples |60,386,279|
| Flipped endpoint normals |3,797,156 (25.7667%)|
| Flipped samples including regular grid |15,459,564|
| Invalid intermediate samples |5|
| Invalid constraint endpoints |0|
| Fibers without any valid normals |0|
| Sampling time |41.39s|
| Sweep including sample construction/sign propagation |45.46s|
| Native second pass including DB read/copy/write/check |189.34s|
| Total runner wall including AFV load |222.89s|
| Output size |12,264,431,616bytes (11.42GiB)|
| Peak RSS (runner measured) |10,219,200KiB (9.75GiB)|

The five invalid regular-grid samples were skipped, retaining last valid
orientation. No invalid endpoint normals were invented or interpolated. Sign
continuity across those gaps is not guaranteed; this is not global alignment.

Independent whole-database validation completed in137.08s:every original fiber
and non-normal constraint column is exact, all14,736,670normal axes unchanged
except sign, all512,698fiber signs+1, integrity/FKs pass. Source database retained.
Results: [oriented_normals_full_validation.json](oriented_normals_full_validation.json).
