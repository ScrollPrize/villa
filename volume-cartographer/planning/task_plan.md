# Task plan

## Problem

The first streamed-byte implementation starts its active-time denominator at
the first response byte. High-latency requests therefore omit connection and
TTFB time, can fail to accumulate a five-second epoch, and overstate bandwidth.
It also retains the old four-completions-per-worker fallback and benchmark
display, allowing local work to affect the shared remote adaptive model.

## Implementation

1. Add an explicit fetcher capability identifying remote HTTP payload
   transfers. Remote Zarr construction sets it; local arrays and compatibility
   fetchers default to false.
2. Start a scheduler transfer measurement immediately before invoking a
   remote fetcher. The active interval begins at request issue, while the curl
   observer continues to provide exact response-body byte increments.
3. Measure aggregate bytes over the union of intervals with at least one
   remote request in flight. This includes DNS/connection/request/TTFB time and
   excludes only true idle gaps.
4. Make streamed bytes and remote-active time the only bandwidth inputs.
   Remove the completion-rate fallback, its four-samples-per-worker option,
   maximum-epoch compatibility option, sample-count statistics, and benchmark
   output.
5. Retain completion records only inside the current adaptive epoch for p90
   request latency, success/failure handling, and the one-completion-per-current
   admission gate. Initial 4x and refinement 2x concurrency probes remain.
   Missing or failed sparse-array requests end their own measurements without
   discarding successful observations from other requests in the epoch.
   Once a probe has selected a higher admission target, every terminal request
   may pace the next permit; only successful payloads contribute measurements.
6. Do not create transfer measurements for local/custom fetchers. Adaptive
   benchmark mode fails clearly if its fetcher lacks remote-byte measurement;
   fixed local benchmark tests remain valid but report no network bandwidth.
7. Keep one shared measurement implementation for VC3D and
   `vc_zarr_download_bench`.

## Tests

- Verify active-time measurement includes TTFB before the first byte.
- Verify concurrent transfers use aggregate bytes over union wall time.
- Verify idle gaps do not enter the denominator.
- Verify a five-second remote epoch scales admission after one completion per
  admitted worker, without an `admission * 4` gate.
- Verify local/custom fetches cannot change adaptive state or network stats.
- Verify interleaved missing chunks cannot prevent a clean-start initial probe.
- Verify adaptive benchmark mode rejects a non-remote fetcher.
- Update deterministic adaptive-controller fixtures to use measured remote
  transfers rather than the removed completion-rate API.
- Build focused scheduler/Zarr/HTTP tests and VC3D, then run the complete
  `vc-core` test label.

## Spec update

Update `planning/spec.md` to define request-issue through completion as the
remote active interval, remove the non-streaming fallback, and state that local
and custom fetches never affect remote adaptation or persisted state.

## Documentation updates

Update `docs/remote_file_cache.md` and the download benchmark documentation to
describe request-inclusive measurement and remove obsolete sample-window
options.

## Changelog

Record the corrected request-inclusive HTTP bandwidth/adaptation model and
local-transfer isolation.

## Independent review

- The remote capability is known before the fetch begins, which is necessary
  to include TTFB and impossible to infer safely from a later byte callback.
- Defaulting the capability to false preserves custom fetcher compatibility
  without misclassifying local work as network traffic.
- Aggregate bytes divided by the union of remote in-flight intervals already
  includes concurrency; applying a parallelism multiplier would double count.
- Completion latency remains useful for comparing probe candidates but does
  not estimate bandwidth.
- Keeping the 4x initial probe is intentional and distinct from the removed
  `admission * 4` completion window.
- No rendering, cache-content, queue-priority, or numeric sampling behavior is
  changed.
