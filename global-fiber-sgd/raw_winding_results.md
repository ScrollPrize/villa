# Continuous winding extraction

## What changed

The AFV extraction path uses moving-window16 refinement and now defaults to:

- `certify_winding=false`: no variation, rounded-category or signed-order rejection.
- `enforce_maximum_winding=false`: no1.5winding cutoff.
- `closest_parallel_estimator=false`: preserve the computed median rather than
  replacing it with the central connector value.

Existing native measurement/extraction code is reused. Shared BP processing
defaults remain unchanged, and historical AFV configs explicitly retain their
previous flags. Uncertified local adaptive refinement now considers geometry
validity/type transitions, not rounded winding categories. The AFV path is
unsigned, so it never invokes the separate signed orthogonal-consensus option.

`winding_distance` stores the continuous central-connector integral;
`parallel_winding_distance` stores the continuous median over central and probe
integrals (the field also exists for perpendicular observations). Use the median
field for the parallel estimate; the perpendicular central measurement stays
separate. Neither field is rounded. Invalid central or probe measurements still
reject the observation; they are not silently omitted from this raw local mode.

This is not unrestricted all-pairs extraction: the128voxel discovery radius,
tangent/orientation/support tests, geometric probe-distance limit and event
deduplication remain. No inferred labels or reference fibers are involved.

## Probe connector explanation

A connector is the straight segment between two selected points on the source
fibers, along which the Lasagna winding integral is sampled. Parallel probes
follow correspondences at offsets±32and±64base voxels. Perpendicular probes
move one side at a time at offsets±8and±16around the refined encounter. Any
probe connector longer than128base voxels currently rejects the whole event,
even if its central connector is within128. This is a geometric locality guard,
not a winding threshold. It was explained but not removed in this task.

The historical variation and rounded-category checks applied to BOTH types.
They compared every probe to the central value, not the median, and rounded
parallel values to integers/perpendicular values to half-integers for rejection.
Those processing gates are now disabled during AFV extraction.

## Reproduction and verification

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_build
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_raw_winding.json
/home/hendrik/.venv_las/bin/python global-fiber-sgd/validate_constraint_database.py global-fiber-sgd/data/raw_winding_v1/full.afc.sqlite
/home/hendrik/.venv_las/bin/python global-fiber-sgd/summarize_rejection_audit.py global-fiber-sgd/data/raw_winding_v1/full.json --output global-fiber-sgd/raw_winding_statistics.json
```

Full512,698fiber Paris4_v1 AFV, same unsigned Lasagna normal cache;
RelWithDebInfo/GCC13,128threads, one extraction. Native141tests pass, including
both relation types with variable/high winding, continuous median preservation,
huge finite inputs and invalid probes. Legacy real-data pilot reproduces all
19,889serialized measurements exactly. No inference or normal alignment.

The exporter refuses to overwrite an existing database. New artifact:
`data/raw_winding_v1/full.afc.sqlite`; original filtered DB is unchanged.
Run metadata/configuration, hashes, timings and logs: `data/raw_winding_v1/`.
No macOS/arm64 execution was performed; shared portable C++ code is used.

## Full-volume result

| Mode | Parallel events | Perpendicular events | Total |
|---|---:|---:|---:|
| Previous moving16, winding filters on | 3,062,208 | 483,271 | 3,545,479 |
| Continuous extraction, moving16 | 6,047,725 | 1,320,610 | 7,368,335 |

32,697,203candidate attempts (30,412,865initial+2,284,338refinement).
First-failure ledger, denominator ALL attempts:

| Outcome | Count | Percent |
|---|---:|---:|
| Initial tangent support | 6,367,038 | 19.4727% |
| Initial orientation | 861,553 | 2.6349% |
| Refined tangent support | 1,130,784 | 3.4584% |
| Refined orientation | 295,791 | 0.9046% |
| Central connector distance | 161 | 0.0005% |
| Probe endpoint support | 2,380,851 | 7.2815% |
| Probe tangent support | 28,755 | 0.0879% |
| Probe orientation | 819,583 | 2.5066% |
| Probe connector distance | 508,671 | 1.5557% |
| Winding variation/category/cutoff | 0 | 0% (disabled) |
| Reciprocal | 0 | 0% |
| Duplicate encounters | 12,935,681 | 39.5620% |
| Accepted | 7,368,335 | 22.5351% |

The exact ledger sums to all attempts. Adaptive candidate counts change because
geometry-only refinement and retained observations alter neighboring states;
these are not fixed-candidate independent ablations. No invalid winding samples
occurred in this run. All14,736,670endpoint normals were valid.

20,608of512,698fibers have no constraints (4.02%); mean degree28.7434,
median16,p95=101,max1566. Database12,097,462,272bytes (~11.27GiB).
Extraction247.59s, export182.04s; runner-inclusive462.60s total. These are
single-run timings, not a statistically established speedup.

Validation complete: full SQLite integrity/foreign-key and source-ID checks;
1,001measurement rows round-trip checked,200endpoints checked against AFV
geometry, full degree sum14,736,670. Full-table SQL confirms2,899,420central
values and2,896,504median values >=1.5;3,328,863central/median pairs differ.
All7,368,335central values are off the half-integer grid (tolerance1e-8),
confirming continuous rather than quantized storage on this dataset.
