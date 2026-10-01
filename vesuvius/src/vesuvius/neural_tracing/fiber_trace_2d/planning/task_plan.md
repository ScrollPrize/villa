# Plan
1. Remove the experimental pinned-copy/memfd/shared-worker modifications entirely. Restore original result storage/copy and all queue/worker settings.
2. TTA internally reuses one GPU input for13transforms, inverse-reframes double-angle directions, and produces exactly2seven-channel products:mean and median. Aggregate onGPU in bounded slabs. No per-transform transfers/publications. Version schema and cache separately from rejected seven-output experiment.
3. Run ordinary single-output matched baseline separately (strict float32, batch1, float32 accumulation), then mean and median using one shared two-product cache. Explicit allow_tf32=False propagated to worker and cache config; original default unchanged. GPUs0–5 and same91fibers/182directions. Preserve hot RAM payload budget by adjusting record counts to actual32/60bytes-per-voxel. No scheduling changes.
4. Independent review, all-transform vector and spatial-gradient oracles, adapter13calls/14channels/2products/6persistedchannels test, normal pipeline/cache regressions. Run approved runner and report after each completed evaluation. Save exact configs and global/per-scroll results.
## Spec update
Opt-in experiment only, one branch/stride1/no embeddings. Median in canonical double-angle coordinates, projected to unit disk, not geometric spherical median. No production default changes.
## Docs updates
Record rejected expansion in changelog; replace current task log. Document freshv2 two-product experiment, matching baseline, throughput and restart results, limitations.
