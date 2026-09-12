# Full classic-checkpoint evaluation and source-prefetched repeat

Status: first pass running; final timings and second-pass results pending.

## Protocol

- Revision5a50f4760, classic best91_5k snapshot (SHA256 in workload/audit).
- Six collections/five scans,50locked fibers,100whole-fiber directions.
- Native beam8/lookahead2/step4, fixed reference threshold20base voxels;
  no changes to tracer/model/precision or input geometry between passes.
- Eight H100s,224logical CPU exclusive manager priority100. No concurrent tests
  or training during timed runs; initial external GPU-compute occupancy checked.
-512cubed inputs,border32,overlap96,scaledown4,32cubed output chunks;
  bf16 inference,f16raw/f32weights,chunk-major accumulators;
  128tile readers,256input slots,128result slots,native64IO/8decode workers.
- Historical successful bidirectional PHerc0139 pilot gate reused. It is not a
  new-binary pilot; current native regressions and short real-model profile had
  already passed before this full run.

First pass commands (all succeeded through launch):

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full
```

Output:
`/ephemeral/hendrik/fiber3d_train2/evaluations/classic_s1_91500_20260910_managed_v3/full`.
Audits/full workload live one level above. Command launched2026-09-11T10:14:42UTC;
exclusive task0e69e03d-19eb-4df7-b7bb-f3b6043fb7a0 granted all224CPUs and8GPUs
with84,447,068,160managed bytes each. Five new prediction caches created under
global `/ephemeral/hendrik/vc3d_cache/live_inference`, model key
`fc798aff0bee43b9e9fdf60f357b71a8d6a46d6d24534dbdc2cf8df7e9b9f2c4`.
Source-fetch caches were not cleared; this is not a guaranteed fully cold run.

## Second pass: source fetches only

User explicitly requires all network inference to run again. Finished inference
chunks must NOT be reused. CT and sheet-normal fetch caches remain at the same
global root, first-pass predictions and results remain untouched.

Optional top-level evaluation `inference_cache_namespace` creates a fresh
single-use `live_inference/_namespaces/<namespace>/<source>/<model>` tree.
Default path/reuse behavior is unchanged. Names are strict safe1–64character
components; existing namespaces and symlink escapes fail. A failed startup
consumes its namespace; retry with a new name, never delete/adopt old evidence.
The option changes only prediction-cache placement, not source cache roots,
frozen snapshot, network evaluation, accumulator math or tracer parameters.

The small path-helper change also changes the hashed `live_evaluate.py`
implementation identity for pass2; this is recorded rather than pretending the
entire code identity is identical. First pass retains its original loaded code.
The new helper/config have independent review; regression tests await release of
the first timed run to avoid CPU contamination.

Prepared second workload in repository:
`fiber_trace_3d/configs/eval_classic_collection_warm_inputs.json`.
Compared with pass1, only output_root and operational namespace differ.

Planned commands, only after pass1 completes and helper tests pass:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py prepare --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_warm_inputs.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py validate-full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_warm_inputs.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py full --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_warm_inputs.json
```

## Interim observations, not final results

Final status of this attempt: cancelled at user request after poor tracing
scaling, with2complete case directories. Evaluation launcher/coordinator/native
tracer exited; manager records task26ae78a2-c299-4e05-ac22-fe3c0c810f74 finished
with zero resources, and training received its grant again. A3s shutdown wait
timeout was reported before final cleanup confirmation. Published inference,
source fetches and completed case outputs were preserved. Not a completed run.

Restart submitted2026-09-11T11:34:04UTC using the warm_inputs_v1 commands above.
Prepared/validated all50fibers/100directions. Manager task
`26ae78a2-c299-4e05-ac22-fe3c0c810f74` received224logical CPUs and all8GPUs;
training yielded through the manager and holds zero resources. Fresh products
created for all5source volumes, preserving failed-run outputs and fetched inputs.
Source cache is partially warm, not guaranteed fully prefetched. Run remains
active (exec session62273); no completed runtime or ETA yet.

Update: first run failed at approximately24min with `live inference scratch
budget exhausted` (64GiB/source configured cap). Last progress:1408.3s,
2632tiles,4861published chunks. No successful full-run timing or fully prefetched
input set exists yet. Second pass is not launched. The two20s process probes
caught cleanup, not active tracing/inference, and cannot quantify the requested
network-blocked fraction. First probe TCP totals also have a corrected parser
identity bug; do not use them. GPUs are now idle. See task_log for details.

User-authorized follow-up removes that application scratch cap entirely,
including touched-brick accounting and per-tile filesystem polling. Actual
filesystem errors/worker failures still abort; shared input/result RAM queues
remain bounded. Both repository workload configs now omit the obsolete setting.
Historical prepared JSON and failed-run artifacts are unchanged. This code fix
has not been measured in a new full run yet.

Scratch-cap removal validated with78CPU regression tests in24.24s (demand store,
shared profile/evaluator settings and inference namespace). Tests cover planning
a >64GiB padded brick without physical allocation, unchanged padded-edge values,
storage-error propagation and cleanup. Exact command and sandbox limitation are
in planning/task_log.md. Independent code review passed.

At~18min:about2000inferred512tiles,3590published chunks,no completed direction
artifacts. Native logs show previously uncached normal downloads. All128readers
often run with zero prepared reads and nearly empty GPU queue. Sampled process
averages near14min were5002%CPU(input/coordinator) and6764%CPU(native tracer),
so remote-fetch cost is not the only plausible explanation of the difference
from the523equivalent128tiles/s buffered short profile. No final runtime/ETA or
tracing-quality metric is yet claimed. Native completion and subsequent
reference-prefetch/inference drain will be reported separately.
