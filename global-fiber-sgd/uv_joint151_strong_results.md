# Single fresh joint 151-winding solve

Status: running. This is one optimizer run per convention, not a staged pipeline.
No previous U/V checkpoint is loaded. Upstream normal/H/V/direction results and
all physical constraints are unchanged.

All 151 winding probabilities, rejection probabilities, shared U/V anchors,
per-winding offsets and widths are optimized jointly. Strong ordering charge16
is applied from the first update; rejection has no warmup. Existing entropy
regularization retains its within-run ramp. There is no later repair stage.

Budget: 16,129 batches of4,096 sampled events =10.000217 event-pass equivalents
over6,606,100 constraints. This is importance-weighted sampling with replacement,
not ten exhaustive sweeps. The older10–100 categorical iterations were exhaustive
passes with fixed geometry, so their iteration counts are not minibatch counts.
Both directions use separate GPUs, shared24GiB reservations. Measured startup
allocator use ~11.5GiB/device. Final metrics use the full graph.

## Reproduce

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_joint151_strong_ccw.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_joint151_strong_cw.json
```

Output directories: `data/uv_joint151_strong_v1/ccw` and `cw`.
Final audit config: `report_uv_joint151_strong.json` with the same approved runner.

Independent plan/config review approved. Existing40tests passed earlier;24solver
regressions rerun after teaching the audit to reconstruct fresh initialization
when there is no checkpoint. The audit change does not alter optimization.
