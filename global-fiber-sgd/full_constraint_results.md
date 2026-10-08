# Full Paris4 unsigned constraint extraction — 2026-10-08

Complete reusable database:
`/home/hendrik/villa2/global-fiber-sgd/data/constraints_v1/full.afc.sqlite`

Size:4,249,571,328bytes (4.25GB / 3.96GiB). Retain `data/Paris4_v1.afv` for
source geometry. This is an AFV-like SQLite constraint artifact, not a VC3D AFV
geometry file. Schema/usage: [constraint_database.md](constraint_database.md).
Database SHA256: `98d08c49c795c8f089677ecc36ea446f1da5e0b9788c419bf696626218276c07`.

## Result

All512,698original source fibers,5,265,142stored points,719.132Mbase-voxel arc.
No cropping, cutting, sign alignment, tracing, inference or winding assignment.
Same unsigned local-event defaults as preceding slab benchmark,128workers.

| Metric | Result |
|---|---:|
| Accepted constraint events | 2,334,188 |
| Parallel | 1,180,410 (50.57%) |
| Orthogonal | 1,153,778 (49.43%) |
| Fibers with at least one constraint | 417,673 (81.47%) |
| Isolated fibers, retained in database | 95,025 (18.53%) |
| Mean incident events per fiber | 9.106 |
| Median / p95 / max incident events | 5 / 33 / 537 |
| Distinct connected fiber pairs | 756,560 |
| Mean distinct neighbors per fiber | 2.951 |
| Median / p95 / max distinct neighbors | 2 / 9 / 49 |
| Invalid endpoint normals | 0 of4,668,376 |

Mean incident events is2E/N, counting an event at both participating fibers.
Local repeated encounters explain why event degree exceeds distinct-neighbor
degree. Isolated fibers mean no event survived these extraction settings, not
that the underlying papyrus/fiber is anatomically disconnected.

Approximate unsigned distance distribution (nearest half-winding bins, not
ground-truth classes; continuous measurements are retained unchanged):

| Nearest half-winding | Parallel | Orthogonal |
|---:|---:|---:|
| 0.0 | 605,461 | 20,635 |
| 0.5 | 238,469 | 456,930 |
| 1.0 | 265,051 | 503,455 |
| 1.5 | 71,429 | 172,758 |

All accepted measured distances remain strictly below1.5; the last bin rounds
values near that bound. No signed winding-order information is inferred.

## Measured performance

GCC13 RelWithDebInfo, one complete extraction,128workers configured, no GPU.
Source/filesystem caches were not flushed; not a cold-storage benchmark.

| Stage | Seconds |
|---|---:|
| Catalog selection | 1.755 |
| Geometry loading | 29.849 |
| Constraint extraction | 201.810 |
| Database export/endpoint normals/indexes/checks | 76.685 |
| Entire native process | 312.46 |

Extraction phases: prepare0.881s,search34.717s,orientation10.420s,winding
integration/certification146.660s; remaining9.131s is normalization/remapping/
sorting and other extraction bookkeeping.30,412,865candidates were measured;
no invalid winding integration. Endpoint normal sampling during export14.234s.
PeakRSS49.67GiB. TotalCPU2519.76s across312.46s wall (~8.1average cores);
not all stages scale to128workers. The prior slab-based2–3minute projection was
optimistic: actual extraction3m22s, plus loading/export, totals5m12s.

## Validation and provenance

- Pilot: all19,889measurement records preserved every C++ field (including
  optional nulls), matched typed columns;200endpoints matched source geometry.
- Full: SQLite integrity/FK checks passed; all AFV source IDs present exactly;
  1001deterministically sampled measurement records and200source endpoints passed.
- Full eventdegree sum4,668,376=2E; isolated count agrees independently.
- JSON summary `data/constraints_v1/full.json`; command/resources/input and binary
  hashes/source patch `data/constraints_v1/runner/full.*`. New exporter source is
  separately hashed as an input. Full stats generated programmatically into
  [full_constraint_statistics.json](full_constraint_statistics.json).
- Run via approved runner/config `run_extract_full.json`. Existing output paths
  are refused; change paths for a new run. Reuse current database without rerunning.

No serialization of fiber geometry or downstream optimization in this task.
