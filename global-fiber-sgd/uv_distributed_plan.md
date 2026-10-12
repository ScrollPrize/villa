# Larger batches and one multi-GPU joint solve

User authorizes larger-batch measurements after coordinate reformulation, then
one joint solve on available GPUs0–5. Use combined coordinates with standard
division, compiled full quadratic objective, no hypothesis pruning, fresh seed0.

## Part1: larger single-GPU batches
Exclusive same-GPU timing B16384/32768/65536/131072,105updates,5warmup; mean/p50/
p95,peakallocated. Matched ~66.06M sample quality trials B32768/65536/131072 with
state LR=.03sqrt(B/4096),widthLR2. This LR rule is a heuristic, not equivalence.
Report L1/within2/retainedlength/sign/ranges against B16384LR.06. Avoid OOM using
72GiB reservation per H100; do not change precision. Independent runs may use
matrix lanes under one exclusive lease, but that is not the distributed solver.

## Part2: distributed integration
Use one existing solver implementation, no duplicated loss/optimizer loop.
Replicate parameters/evidence on each GPU and shard each global microbatch.
All ranks generate the same seeded global sample IDs and slice disjoint equal
partitions; config batch_size is GLOBAL, divisible by world size. Nodes similarly
partitioned; tiny continuation sample and width priors replicated. Mean local
gradients across ranks therefore reproduce the global objective. Keep SparseAdam
semantics: coalesce local sparse gradients, stage into a reusable dense buffer,
all-reduce/average; separately union touched-row masks (including zero-gradient
rows), then reconstruct sparse gradient on those rows only. Never turn absent
rows into optimizer updates. All-reduce width gradients. All replicas use same
optimizer states/updates. Rank0 alone audits/writes, others barrier safely.
Record effective global/local batch,worldsize,communication latency/memory.

Launch via existing managed runner, one exclusive lease, multiprocessing spawn
inside leased process; set rank-local CUDA devices. NCCL on GPU, Gloo CPU tests.
Use a temporary local rendezvous file and cleanup/child failure propagation.
Initially support sampled minibatch mode only; reject exhaustive/full accumulated
mode explicitly rather than silently changing its semantics. Preserve singleGPU
path. Dataset/model cache and inference unchanged.

## Validation/performance
Independent plan and code review. CPU multi-process gradient and two-step
SparseAdam comparison (repeated/touched-zero/absent rows) vs concatenatedbatch;
include width averaging. CUDA distributed preflight vs singleGPU concatenated
gradient on small fixture. Exclusive1/2/3/6GPU production timings with matched
globalbatch and localbatch scaling; report communications and memory. Matched
full6GPU quality trial vs sameglobalbatch singleGPU when memory allows. No
bitwise promise across reduction trees, but objective/gradients must agree.

## Spec update
User follow-up selects fixed32768 constraints/GPU, so globalbatch scales with
worldsize. Benchmark1/2/3/6GPUs and run6GPU336updates (~10sampledpasses), not a
same-globalbatch singleGPU196608 control. Report this quality/scaling distinction.
Profile diagnostics separately (synchronization/copy, audit, reference matching,
logging), including per-call cProfile on6GPU run. Propose fixes without dropping
metrics or modifying diagnostic semantics. Timing instrumentation is read-only.

New opt-in distributed_world_size; batch_size consistently global. Prior stages,
losses,broken updates,width optimizer untouched. Only ordinary coordinate divide
was explicitly authorized to use the ordinary Triton lowering. No TF32/FP16.

## Docs updates
Archive coordinate tasklog. Document configuration, manager lease/rank mapping,
globalbatch/LR,supportedmodes,exact commands,unit tests,performance andquality in
uv_distributed_results.md; update README/task/status/changelog.
