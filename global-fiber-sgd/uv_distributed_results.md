# Larger batches and distributed solve

## Larger single-GPU batches (completed)

Combined coordinates, standard division, full compiled quadratic, fresh seed0.
Timing105updates,5warmup excluded,100CUDA-event samples, exclusive sameGPU5
(H10080GB). Quality runs use ~66,060,288samples and LR=.03sqrt(B/4096),widthLR2.
These are sampled pass-equivalents, not exhaustive epochs. No precision changes.

| Global batch | LR | Mean ms | p50 ms | p95 ms | kconstraints/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|
|16384|0.060000|22.431|22.171|24.257|730.4|7.392|
|32768|0.084853|37.266|37.260|37.332|879.3|11.954|
|65536|0.120000|69.496|69.491|69.551|943.0|21.069|
|131072|0.169706|131.754|131.753|131.812|994.8|39.304|

| Batch | Updates | Ref L1 | Within2 % | Retained length % | Sign errors % |
|---|---:|---:|---:|---:|---:|
|16384|4032|11.659781|13.287325|65.490114|3.195105|
|32768|2016|11.635647|13.269249|65.343782|3.316426|
|65536|1008|11.602768|13.210003|65.106543|3.524100|
|131072|504|11.559364|13.197674|65.032820|3.534599|

Larger batches improve throughput, with diminishing returns after32k–64k.
Square-root LR is not optimization equivalence: sign errors worsen modestly,
retention declines, and lower L1 uses a changed retained population. No quality
improvement claimed. Fitq10/q90 and reference ranges should be read from saved
full histories. Only one seed. Manager leaseea824a75-1012-4591-b9fd-1eb627d84608
completed all7jobs; quality lanesGPUs0/1/2,timinglaneGPU5.

Reproduction (approved runner):
```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_large_matrix.json
```

## Distributed implementation (validated)

One solver loop, globalbatch split equally over ranks. Each rank draws identical
seeded global event/node/continuation samples using an explicit CUDA generator;
only event/node batches are sliced. Continuation/widthprior terms are replicated
and therefore counted once after averaging. Parameter replicas have identical
SparseAdam/Adam updates. Dense gradient allreduce plus touched-row mask union
preserves absent-row optimizer semantics, including touchedzero/cancelled rows.

Rank0 alone audits and publishes; barriers prevent other ranks crossing those
checkpoints. Sampled/minibatch mode only; reject active oversampling,earlystop,
fullpass/globalaccumulation rather than silently changing behavior. No external
GPU requests perrank: spawn inside one manager lease. NCCL production/Gloo tests.

CPU2rank tests compare gradients,parameters and SparseAdam moments through2steps,
including duplicates,cancellation,exactzero and previously-touched now-absent
rows.28targetedtests pass. Production runs execute CUDA actual pair/continuation
loss-gradient equivalence preflight against the concatenatedbatch before loading
full data. Default coordinate backend remains reference; experiments opt in.

## Fixed32768/GPU scaling

One exclusive lease GPUs0–5, standard combined coordinates, explicit shared
sampling generator.1/2/3GPU runs105updates (5warmup excluded);6GPU336updates
(5warmup excluded). CUDA events exclude diagnostic barriers, initial widths and
logging. Globalbatch/LR scale with GPUcount: this is weak scaling, not identical
optimizer trajectories. User selected32k/GPU after larger-batch tradeoff.

| GPUs | Global batch | Mean ms | p50 ms | p95 ms | Mconstraints/s | Peak GiB/GPU | Reduction ms |
|---|---:|---:|---:|---:|---:|---:|---:|
|1|32768|37.261|37.256|37.295|0.879|11.954|—|
|2|65536|43.824|43.798|43.890|1.495|12.789|4.773|
|3|98304|46.194|46.186|46.258|2.128|12.790|5.793|
|6|196608|49.725|49.689|49.811|3.954|12.792|6.233|

Reduction includes coalescing, staging, communication and sparse reconstruction.
SixGPU throughput is4.496times singleGPU. Actual pair/continuation gradient
preflight passed on2,3,6GPUs; CPU2rank optimizer-state regression also passes.
SixGPU full trial uses336updates/66,060,288events;L1=11.521724,within2=13.068242%,
retainedlength65.077317%,signerrors3.474067%. Different batch/update count means
these are not expected to exactly reproduce16k/32k singleGPU quality. No claim
that lower L1 alone is better given changed retained population.

Run:run_uv_distributed_32k_suite.json via the same approved entrypoint above.
Diagnostics profile and full-runtime comparison:uv_diagnostic_profile.md.
