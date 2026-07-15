# 3D Streaming Prefetcher Plan

## Goal

Make `fiber_trace_3d --prefetch` follow the 2D streaming prefetcher design
instead of serially generating every sample's chunk dependencies before any
download progress appears.

The 3D implementation should be structurally equivalent to the 2D prefetcher
except where the data model is inherently different.

## Current Problem

`FiberTrace3DLoader.prefetch` currently:

1. Prints a single "generating chunk requests" line.
2. Loops over every sample index serially.
3. Builds and de-duplicates a full in-memory request dictionary.
4. Starts a download thread pool only after all dependency generation is done.
5. Prints progress only during the download phase.

For large runs, this looks like a hang and does not overlap dependency
generation with cache classification or downloads.

## Target Architecture

Mirror the 2D prefetcher:

1. Keep a bounded producer pool for dependency generation.
   - Use `prefetch_sampler_workers`.
   - Submit at most `producer_count` sample jobs at a time.
   - Each producer builds dependency requests for one raw deterministic sample
     index.

2. Keep a bounded download pool.
   - Use `prefetch_workers`.
   - Only active download futures count against this worker limit.
   - Downloads use the existing `_download_prefetch_request` helper so atomic
     temp-file writes and `.empty` handling stay centralized.

3. Consume producer results in raw deterministic sample-index order.
   - Producer futures may finish out of order.
   - Store completed producer results in a small `producer_results` map.
   - Classify/enqueue only when `next_producer_result` is available.
   - This preserves the same deterministic request order as serial sampling and
     as the 2D prefetcher.

4. Classify each chunk request immediately.
   - De-duplicate globally by `(store_identity, key)`.
   - Existing cache data path: count as cache hit and mark complete.
   - Existing `.empty` marker: count as known missing and mark complete.
   - New chunk: enqueue for download.
   - Already-seen queued chunk: register the sample as waiting for that chunk,
     and update its priority if the current raw sample index is earlier.

5. Prioritize downloads by earliest raw deterministic sample index.
   - Use a heap of `(raw_sample_index, sequence, identity)`.
   - Keep `queued_downloads` as the current authoritative priority/request map.
   - Ignore stale heap entries.
   - Do not cancel active transfers if a lower-priority duplicate appears later.

6. Track per-sample completion and safe-prefix `idx`.
   - Maintain `sample_pending`, `chunk_waiters`, `sample_requests_closed`,
     `complete_raw_samples`, and `next_safe_raw_sample`.
   - Advance `idx` only when every chunk for every preceding raw sample is
     either cache-hit, missing-known, downloaded, newly-missing, or otherwise
     completed.

7. Print live progress during both phases.
   - Initial line should include samples, worker counts, sample mode, and
     `mode=dependency_chunks`.
   - While producer generation is incomplete, print sample progress and sample
     ETA.
   - Always print download progress, download ETA, `idx`, unique chunks,
     cache hits, queued downloads, configured transfer workers, configured
     sampler workers, skipped samples, missing chunks, downloaded chunks,
     errors, MiB, and MiB/s.
   - Once all producers are done, omit the sample progress prefix just like the
     2D prefetcher.
   - Use periodic progress updates while waiting for futures so a slow producer
     or slow download does not look silent.

8. Handle skips and errors like 2D.
   - `ValueError` from a sample producer should skip that sample, increment
     skipped count, close the sample's request set, and continue.
   - Download errors should increment download error count and preserve the
     first error string.
   - Fatal unexpected exceptions should cancel queued producer/download futures
     and avoid waiting on a stale backlog.

9. Temporarily pin PyTorch CPU intra-op threads to one while producers run.
   - Save the previous value.
   - Set `torch.set_num_threads(1)` during prefetch if needed.
   - Restore it in `finally`.

## 3D Build Request Function

Add a 3D-specific `build_requests(sample_index)` equivalent to 2D's producer
job:

1. Resolve the deterministic raw sample to the configured sample mode.
2. Build the 3D augmentation-envelope coordinate volume for that CP-centered
   patch.
3. Convert selected-scale coordinates to base coordinates exactly as current
   `chunk_requests_for_sample_index` does.
4. Build the finite/in-bounds valid mask.
5. Call `record.sampler.chunk_requests_for_coords(coords_base, valid)`.
   - This uses the already-fixed rank adapter that flattens `[Z,Y,X,3]` into
     the VC3D dependency API shape.
6. Return `(raw_sample_index, valid_voxel_count, requests)`.

No 2D strip handling is added to the 3D path.

## Planned Differences From The 2D Prefetcher

These are the only intentional differences.

1. Patch unit:
   - 2D: one sample expands into `len(strip_z_offsets)` strip image patches.
   - 3D: one sample is one CP-centered 3D patch.

2. Patch counter naming:
   - 2D reports `patches` because a sample has multiple strip-z image patches.
   - 3D should either omit `patches` or report `patches=samples_done`; it must
     not imply strip offsets exist.

3. Top-view branch:
   - 2D can include an optional top-view prefetch branch.
   - 3D has no top-view branch.

4. Dependency coordinates:
   - 2D producers use conservative source-strip envelope coordinates per strip
     offset.
   - 3D producers use one conservative 3D augmentation-envelope coordinate
     volume per CP-centered patch.

5. Valid-count label:
   - 2D counts valid pixels across strips/top views.
   - 3D counts valid voxels for the 3D patch.

6. Config inputs:
   - 2D uses `strip_z_offset_count`, `strip_z_offset_step`, and optional
     `include_top_view`.
   - 3D does not use those keys; it uses `patch_shape_zyx` and 3D augmentation
     envelope settings.

7. Function names/types:
   - The implementation lives in `fiber_trace_3d.loader.FiberTrace3DLoader`.
   - It should not import the 2D loader to share private state machinery unless
     that machinery is first extracted into a small neutral helper. For this
     task, duplicating the 2D prefetch state machine locally is acceptable to
     keep the diff direct and reviewable.

Everything else should match the 2D behavior:

- producer/download split;
- bounded producer futures;
- bounded active download futures;
- global chunk de-duplication;
- cache-hit and `.empty` classification before download;
- chunk waiters and per-sample pending sets;
- deterministic ordered producer consumption;
- download priority by earliest raw deterministic sample index;
- safe-prefix `idx`;
- progress style and fields;
- skip/error semantics;
- fatal cancellation behavior;
- temporary PyTorch CPU thread pinning;
- VC3D metadata preservation;
- Python atomic download helper usage.

## Implementation Steps

1. Refactor `FiberTrace3DLoader.prefetch`.
   - Replace the current full serial request-generation loop with the streaming
     producer/download state machine.
   - Keep `_download_prefetch_request` as the download path.
   - Reuse `_existing_data_path`, `_empty_marker_path`, and the same request
     metadata semantics already shared through `ZarrChunkRequest`.

2. Add 3D prefetch counters.
   - Either reuse/adapt the existing 2D `_PrefetchCounters` if it is in a
     shareable location, or add a small 3D-local dataclass with the same fields
     actually needed by 3D.
   - Include `samples_done`, `samples_skipped`, `valid_voxels`,
     `unique_chunks_seen`, `cache_hits`, `queued_for_download`,
     `download_done`, `downloaded`, `known_missing`, `newly_missing`,
     `download_errors`, `bytes_downloaded`, `queued_download_futures`,
     `max_exclusive_sample_index`, `first_error`, and `first_sample_skip`.

3. Add ordered producer result handling.
   - Producers submit raw sample indices.
   - Completion can be out of order.
   - Classification remains in raw sample order.

4. Add live progress output.
   - Match the 2D two-bar style and field semantics.
   - Use `samples[...]` while generation is incomplete.
   - Use `downloads[...]` throughout.
   - Include `samplers=<producer_count>` and `transfers=<worker_count>`.

5. Add robust cancellation.
   - On unexpected `BaseException`, cancel pending futures and shut down
     executors with `wait=False, cancel_futures=True`.
   - Restore PyTorch thread count in `finally`.

6. Keep `chunk_requests_for_sample_index` as the single-sample dependency API.
   - The streaming producer should call it rather than duplicating the geometry
     code.
   - This preserves the current fixed shape/rank behavior in
     `Vc3dCoordinateSampler.chunk_requests_for_coords`.

## Tests

1. Unit test streaming request generation.
   - Use a fake sampler returning deterministic overlapping chunk requests.
   - Verify the 3D prefetcher globally de-duplicates chunk identities.
   - Verify repeated chunk requests are downloaded/classified once.

2. Unit test ordered producer consumption.
   - Make fake producers complete out of order if feasible by monkeypatching the
     dependency generation helper or using controlled futures.
   - Verify `idx`/safe-prefix behavior only advances after all earlier samples
     are complete.

3. Unit test cache-hit and `.empty` classification.
   - Existing data file counts as cache hit and is not downloaded.
   - Existing `.empty` marker counts as known missing and is not downloaded.

4. Unit test sample skip behavior.
   - A `ValueError` from one sample increments skipped count, closes that sample,
     and allows later samples to continue.

5. Focused test command:
   - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

6. Optional local smoke command if network/cache access is available:
   - `PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_3d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_all_64_sd2.json --prefetch --prefetch-steps 1`

## Spec Update

Update `planning/specs.md` to say 3D prefetch follows the same streaming
dependency/download architecture as the 2D prefetcher, with only the inherent
3D differences listed above.

Also state that the 3D prefetch progress must be live during dependency
generation, not only after all chunk requests have been generated.

## Docs Updates

Update `docs/code_structure.md`:

- 3D prefetch no longer serially generates all requests before downloading.
- 3D prefetch uses producer workers and download workers like 2D.
- 3D dependency coordinates are a CP-centered augmentation-envelope volume
  instead of strip-z image surfaces.

Update `planning/local_development.md` only if the prefetch command or expected
progress output changes materially.

## Changelog

Add a 2026-07-15 entry:

- 3D prefetch now streams dependency generation and downloads like the 2D
  prefetcher, with live progress and deterministic safe-prefix reporting.

## Non-Goals

- Do not change 3D training sample order.
- Do not change 3D model architecture or target semantics.
- Do not change VC3D cache path construction or reconstruct cache paths in
  Python.
- Do not add 2D strip-z or top-view behavior to 3D.
- Do not add shear/ringing support.
