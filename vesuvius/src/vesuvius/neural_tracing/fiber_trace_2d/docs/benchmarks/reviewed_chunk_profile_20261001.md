# Reviewed evaluation: GPU profile and output-chunk sweep (2026-10-01)

## Outcome

Keep the existing 64³ output chunks. Smaller 32³ chunks save inference tiles but make coordinator/accumulator handling slower and fragment GPU batches. Larger 128³ chunks keep GPUs busy but require much more inference. No production defaults changed.

Same frozen raw u31,500 snapshot (`ffd7aff24441e82e53b5f3fef69ab19d24a7b632e1dee4dac6487401694d244d`), 91 reviewed fibers / 182 directions across five scrolls. Every run completed without infrastructure or route-limit failures. No embedding or reverse-consistency filter. Same native beam8/lookahead2, step4 prediction voxels,40/80base acceptance ellipse,160base excursion grace,80base restart grouping.

Only **output Zarr chunk geometry** changes. Inference tiles remain256³,overlap96,border32,batch limit4,bf16. These are not inference tile/context-size trials. Output voxels are2base voxels wide:32/64/128³output chunks span64/128/256base voxels per edge.

## Timing and work

One exploratory fresh-inference run per variant, sequential order64,32,128. Source CT/normal caches reused. Timing is not a statistical confidence estimate; later variants can benefit from source-cache warming. Output chunk changes also change requested coverage and scheduling.

| Output chunk | Coordinator s | Change vs64 | Full runner s | Inference tiles | Published spatial chunks | Restarts | Mean mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32³ | 310.38 | 28.6% | 344.68 | 4034 | 10934 | 25 | 53.314 |
| 64³ | 241.34 | 0.0% | 277.08 | 5033 | 4520 | 24 | 55.447 |
| 128³ | 356.82 | 47.8% | 390.89 | 7448 | 1963 | 23 | 57.757 |

Coordinator time includes its startup/manager wait and final profile serialization. Full runner additionally includes prepare, binding validation and cleanup; snapshot was already frozen. Serialization overhead was chunk32: 1.53s, chunk64: 1.15s, chunk128: 1.14s. Previous unprofiled64run was240.12s versus241.34s here.

## GPU utilization

Measured with nvidia-smi at1Hz over six H10080GB GPUs (0–5). Values are sampled kernel-active time, **not FLOP or bandwidth efficiency**. Pre-grant samples can belong to another job and are excluded. The inference interval starts at the first logged GPU submission (1Hz coordinator observation), includes model initialization/tail, and ends at the summary. Alignment uses summary mtime minus coordinator wall time, hence is approximate.

| Chunk | Inference interval s | Mean GPU utilization | Median sample | P95 sample | Eval-attributed utilization over full coordinator time (approx.) |
|---|---:|---:|---:|---:|---:|
| 32³ | 264.45 | 53.0% | 59% | 100% | 45.2% |
| 64³ | 211.87 | 73.4% | 100% | 100% | 64.4% |
| 128³ | 312.36 | 74.1% | 100% | 100% | 64.8% |

The last column assigns zero evaluation GPU work before first submission; it is not an average including another tenant's pre-grant activity. Allocation/startup timing differs: first GPU submission occurred at chunk32=45.93s, chunk64=29.47s, chunk128=44.47s.

## Worker stage timings

Milliseconds per tile; batch timing is divided by batch size. CUDA event intervals can include host-launch gaps and output-copy stalls, so output is **not a pure DMA bandwidth measurement**. CPU output wall time also waits for asynchronous inference: do not add it to CUDA inference time. Across workers, summed durations overlap and are not elapsed critical-path percentages.

| Chunk | Stage | Mean ms/tile | P50 | P95 |
|---|---|---:|---:|---:|
| 32³ | Input H2D | 2.82 | 2.72 | 3.95 |
| 32³ | Model CUDA interval | 191.79 | 157.30 | 339.28 |
| 32³ | Output processing / transfer interval | 95.02 | 59.49 | 238.25 |
| 32³ | Entire worker batch / tile | 291.03 | 246.28 | 517.54 |
| 64³ | Input H2D | 2.70 | 2.59 | 4.11 |
| 64³ | Model CUDA interval | 159.66 | 157.26 | 157.54 |
| 64³ | Output processing / transfer interval | 77.35 | 42.28 | 158.12 |
| 64³ | Entire worker batch / tile | 241.01 | 204.07 | 375.66 |
| 128³ | Input H2D | 3.95 | 3.74 | 6.09 |
| 128³ | Model CUDA interval | 160.97 | 157.42 | 158.78 |
| 128³ | Output processing / transfer interval | 75.07 | 55.79 | 138.85 |
| 128³ | Entire worker batch / tile | 241.44 | 219.73 | 313.54 |

For64³: model inference averages160ms/tile; output processing/transfer77ms; input transfer2.7ms. Model computation is the largest component, but output handling is a substantial optimization target. The existing copy is directly into shared file-backed arrays via torch.from_numpy(...).copy_(...), not via an extra temporary CPU tensor. A next experiment could compare bounded pinned staging/asynchronous output transfer against this direct pageable copy, without changing precision.

## Coordinator, accumulation and spilling

| Chunk | Result/event handling s | Background handling s | Full4-tile batches | Accumulation task mean ms | Queue+ack mean ms | Spill written GiB |
|---|---:|---:|---:|---:|---:|---:|
| 32³ | 82.04 | 83.60 | 46.7% | 5.17 | 362.42 | 66.2 |
| 64³ | 19.49 | 20.35 | 94.1% | 10.73 | 36.06 | 293.3 |
| 128³ | 8.61 | 9.74 | 90.3% | 22.68 | 27.30 | 782.3 |

Coordinator stage times are elapsed time within those calls, not CPU-only time. Accumulator queue+ack excludes backlog before successful enqueue and subtracts worker execution; it can include coordinator acknowledgement delay. Parallel task sums must not be interpreted as wall time.

32³ produced500,216accumulator batches versus295,184at64³. Queue+ack grew from36ms to362ms. Its ready-input median remained80tiles; six sampled points had all GPU work drained while accumulation still owned results, versus none for64³. This supports result/accumulator coordination and buffer recycling as the smaller-chunk bottleneck, not a shortage of loader threads. Consider better contribution/event batching and then test result-buffer headroom, rather than blindly increasing all worker counts.

Publication queue depth was zero throughout the32/64runs. This does not prove publication is free: in-flight publications are separately tracked. Publication duration was not individually instrumented in this sweep; counts/queues are available. The128tail includes finishing publications.

## Correctness

All182cases completed in each variant with identical reference geometry. Mean uses the established combined directed reference length/(counted restarts+1).

| Scroll |32³ restarts |64³ restarts |128³ restarts |
|---|---:|---:|---:|
| PHerc0139 | 9 | 9 | 7 |
| PHerc0332 | 0 | 0 | 0 |
| PHerc1299 | 6 | 6 | 6 |
| PHerc1451 | 0 | 0 | 0 |
| PHercParis4 | 10 | 9 | 10 |

Compared with the original unprofiled u31500, changed per-direction restart counts numbered chunk32=3, chunk64=2, chunk128=3. Even the64³repeat changed two cases while retaining the same24total. Therefore these are not bit-identical runs, and one fewer restart at128³ is not evidence of improved quality. Existing asynchronous accumulation/scheduling can alter values near decision boundaries; this sweep does not isolate the exact cause of each changed trace.

Trace geometry diagnostic: symmetric nearest-sampled-point distance between corresponding cases (not the acceptance metric). Mean of per-case means in base voxels: chunk32=2.081, chunk64=1.233, chunk128=1.416. Per-case changes and terminal-distance distributions are retained in the JSON report.

## Configuration / reproducibility

- Host: Intel Xeon Platinum8480+,224logical CPUs, Linux6.8.0-1062-oracle; six NVIDIA H10080GB HBM3.
- Native build: RelWithDebInfo; vc_fiber_trace_metric SHA256 `811e67a776e1efb28e81068c110bb6d1abdad9ed15324401e13db974b9af7fec`.
- Checkout HEAD `7ee60c0885128cd6d02efb121d60e5371710614c`, fiber-exp5, with profiling changes in the working tree.
- 192trace workers,128input/accumulator/publication workers,96input slots,32result slots; max GPU batch4.
- 504GiB hot accumulator payload fixed. Hot slots:917504/114688/14336 for32/64/128; spill segment slots16384/2048/256. Metadata overhead is not constant; no process-PSS/peak-RSS profile was captured.
- Independent plan/code review;17focused tests pass; diff whitespace check clean.
- Output root: `/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3/profile_chunks_u31500_v1/`.
- Each chunk directory retains workload, GPU CSV, monitor window, full summaries, per-case traces, queue history and pipeline_profile.json.
- No inference tile-size sweep was performed: this experiment isolates output chunks. That is a separate follow-up because tile size changes model context and potentially batch-memory limits.
- Mixed-model profile persistence is not supported here; borrowed-session profiles are labelled cumulative and rejected by the per-evaluation analyzer.

Run via existing approved command after setting scripts/training_failure_mining_control.json to `{"action":"reviewed_profile_sweep"}`:

```bash
bash /home/hendrik/villa3/scripts/run_training_failure_mining
/home/hendrik/.venv_las/bin/python scripts/report_reviewed_profile.py
/home/hendrik/.venv_las/bin/python -m pytest -q scripts/test_reviewed_profile_sweep.py scripts/test_reviewed_fiber_evaluation.py
```

Completed results are reused; incomplete profiling runs are rejected rather than resumed into warmed inference caches. For a genuinely fresh repetition use a new output/version and matching inference namespace.
