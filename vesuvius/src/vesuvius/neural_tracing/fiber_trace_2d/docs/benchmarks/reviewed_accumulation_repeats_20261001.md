# Float16 versus float32 accumulation repeatability — 2026-10-01

## Outcome

Six fresh evaluations complete; no infrastructure/route-limit failures. Float32 reduced observed restart variability (5varying cases→1), but final-voxel repeatability improved only modestly. This does not establish float16 accumulation as the main cause of voxel nonrepeatability. N=3perprecision is exploratory, not a statistically established quality improvement.

## Fixed workload and protocol

Frozen raw u31500/checkpoint SHA256 ffd7aff24441e82e53b5f3fef69ab19d24a7b632e1dee4dac6487401694d244d,91reviewed fibers/182directions/five scrolls. Same locked cohort and beam8/lookahead2 as reviewed_chunk_profile_20261001.md; no embedding/reverse-consistency filters. Native RelWithDebInfo build unchanged. Source CT/normal caches reused; prediction namespaces fresh and all warm_chunks=0.

Tile256³,overlap96,border32,outputchunk64³,bf16 model,maxbatch4. GPUmanager exclusive0–5,192traceworkers,128input/accumulator/publicationworkers,96inputslots,32resultslots. Three independent runs each, interleaved16_r1,32_r1,16_r2,32_r2,16_r3,32_r3. No inference/tracing implementation or parallelization changes.

Weight sums alreadyfloat32. Seven raw product planes change16→32bit: record4,718,592→8,388,608bytes. Hot slots114688→64512 and spillsegment slots2048→1152 preserve504GiBhotpayload and9GiBspillsegmentpayload. Runtime record bytes/capacity verified. Shared buffers17.625→18.0625GiB (not identical totalRAM); actual spill/timing/order can differ. Arrival-order policy unchanged, actual tile order and batch composition uncontrolled.

## Restarts and timing

|Precision|Repeat1|Repeat2|Repeat3|Mean|Sample std|Cases varying /182|Mean eval wall s|
|---|---:|---:|---:|---:|---:|---:|---:|
|float16|22|24|25|23.667|1.528|5|234.96|
|float32|24|23|23|23.333|0.577|1|240.67|

Wall time is native evaluation summary wall_seconds, not outer process startup/teardown. Float32mean was2.43%longer; no isolated throughput claim. Float16 min/median/max233.13/234.20/237.56s;float32237.51/240.37/244.13s.

|Scroll|Float16 repeats|Float32 repeats|
|---|---|---|
|PHerc0139|8, 10, 9|8, 8, 8|
|PHerc0332|0, 0, 0|0, 0, 0|
|PHerc1299|6, 6, 6|6, 6, 6|
|PHerc1451|0, 0, 0|0, 0, 0|
|PHercParis4|8, 8, 10|10, 9, 9|

## Actual cached values

Decoded uint8 values, not compressed-file bytes. All common chunk coordinates, including zero voxels. Missing chunks excluded and coverage reported. Per-pair1.14–1.17billion common voxels/channel. All15pairs computed (3within each precision,9crossprecision); below is mean of the3within-precision pair metrics. Quantiles computed from exact full voxel histograms, not averages of chunk quantiles.

|Channel|Float16 changed voxels|Float32 changed voxels|Float16 MAE (byte units)|Float32 MAE|Max absolute difference in either precision|
|---|---:|---:|---:|---:|---:|
|presence|0.143584%|0.139775%|0.00143800|0.00140006|2|
|nx|1.022354%|0.960445%|0.03921102|0.03720802|254|
|ny|0.835707%|0.788343%|0.03485161|0.03185573|254|

Presence maxima are2/255. Nx/nymax254 is an encoded-byte difference, NOT an angular error: hemisphere sign transitions can produce large byte changes. No full-volume angular claim is made. Unchanged background is included in denominators.

Chunks containing any difference decreased: presence20.53%→9.46%,directions20.57%→9.46%; differences remain concentrated in fewer chunks rather than vanishing.

|Precision|Pair|Common / union chunks|Presence changed voxels|nx changed voxels|ny changed voxels|
|---|---|---:|---:|---:|---:|
|float16|r1–r2|4407 / 4651|0.151699%|1.071729%|0.889436%|
|float16|r1–r3|4415 / 4625|0.142511%|1.024004%|0.828835%|
|float16|r2–r3|4414 / 4648|0.136543%|0.971329%|0.788849%|
|float32|r1–r2|4430 / 4596|0.141427%|0.917451%|0.780180%|
|float32|r1–r3|4432 / 4595|0.143660%|1.000380%|0.798563%|
|float32|r2–r3|4460 / 4597|0.134236%|0.963505%|0.786285%|

Crossprecision presence differences average0.357920%: changing precision shifts predictions, not merely their run-to-run variability.

## Cases whose counts differ

|Scroll|Fiber|Direction|Float16 repeats|Float32 repeats|
|---|---|---|---|---|
|PHerc0139|0007_kb_20260908T172232906_000117_run0.json|reverse|0, 1, 0|0, 0, 0|
|PHerc0139|0010_lt_20260831T061702750_000026_run0.json|reverse|0, 1, 0|0, 0, 0|
|PHerc0139|0024_lt_20260909T084022344_000109_run1.json|forward|0, 0, 1|0, 0, 0|
|PHercParis4|0058_anon_20260901T170547855_000317_run1.json|forward|0, 0, 0|1, 1, 1|
|PHercParis4|0071_anon_20260902T214326941_000399_run0.json|reverse|0, 0, 0|1, 0, 0|
|PHercParis4|0079_anon_20260903T214715058_000404_run1.json|reverse|0, 0, 1|0, 0, 0|
|PHercParis4|0084_anon_20260905T163237868_000469_run5.json|forward|0, 0, 1|0, 0, 0|

Float32 still has one varying direction: Paris4fiber0071reverse. Paris4fiber0058forward fails consistently in32but not16. Thus smaller variability is not a guarantee every fiber improves.

## Reproduce and inspect

Approved no-argument command:
```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
```
Control file scripts/training_failure_mining_control.json action=reviewed_accumulation_sweep. Existing fully completed runs are validated/skipped; partial runs or preexisting prediction namespaces without complete outputs are rejected to prevent warm-cache comparisons. All original workloads/caches retained. Float32 is enabled in the experiment workloads; production-wide defaults were not changed.

Artifacts: /ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/accumulation_repeats_u31500_v1/
Each float16_r1..3 /float32_r1..3contains immutable workload.json, full/summary.json, inference_summary.json,pipeline_profile.json,queues and per-case traces.
Full per-scroll histograms/all15comparisons: comparison.json in that directory.
Compact durable results: reviewed_accumulation_repeats_20261001.json beside this report.

CPU comparison uses16threads/32bounded tasks, actual chunk metadata, source/model/protocol/producer identity checks; no GPUcomparison work.11focused tests passed:
```bash
/home/hendrik/.venv_las/bin/python -m pytest -q scripts/test_reviewed_accumulation_sweep.py scripts/test_check_reviewed_divergence.py scripts/test_reviewed_profile_sweep.py
```
Synthetic tests cover immutable configs/payload budgets, missing chunks, uint8differencehistograms includingzeros, partialvolume boundaries, incomplete-run guards. git diff --checkpassed. Independentplan/code review incorporated9/16budget correction,bounded comparison scheduling,identityvalidation and cold-run guards.

## Interpretation / remaining work

Observed restart spread narrowed22–25→23–24, but presence and directional-byte disagreement only fell roughly3–6%. Float32 accumulation alone is not a reproducibility fix. Model inference remainsbf16; changing accumulation precision also changes scheduling/batchcomposition, so this is an end-to-end precision comparison, not an isolated arithmetic proof. Definitive attribution requires reusing identical raw predicted tiles to isolate accumulation order, and/or comparing model tile outputs across reruns. Saved outputs are final quantized caches, not prequantization accumulators. None of those isolation experiments were silently substituted for this requested comparison.
