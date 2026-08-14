# Zarr download benchmark

`vc_zarr_download_bench` measures encoded chunk downloads through the same
`ZarrChunkFetcher` and adaptive `ChunkRequestScheduler` used by remote VC3D
rendering. It excludes chunk decoding and the persistent cache so network
throughput and download-admission behavior can be tuned independently.

Build and run the production-default automatic policy:

```bash
cmake --build volume-cartographer/build --target vc_zarr_download_bench -j
volume-cartographer/build/bin/vc_zarr_download_bench \
  s3://bucket/path/to/volume.zarr --level 0 --chunks 256
```

The automatic defaults are the rendering defaults: 2 initial workers, 64
maximum workers, a 2--5 second measurement epoch, and a rolling displayed
bandwidth window of 4 successful samples per admitted worker. Initial discovery
probes `4C` and `C/4` to cover the range quickly. After the first overshoot,
direction reversal, or retained-center bracket, refinement compares `2C` and
`C/2`. Search remains continuous until five direction reversals or retained
centers confirm a local operating point. Increasing concurrency admits one
additional worker per completed transfer instead of starting a burst. Each
setting can be overridden for experiments:

```bash
volume-cartographer/build/bin/vc_zarr_download_bench \
  s3://bucket/path/to/volume.zarr --chunks 512 \
  --min-workers 4 --workers 32 --samples-per-worker 8 \
  --epoch-min-seconds 1.5 --epoch-max-seconds 4 \
  --initial-probe-multiplier 4 --search-turns 5
```

After concurrency settles, bandwidth stability controls exploration frequency.
Bandwidth cannot be classified as stable until at least 5 minutes of saturated
transfer time has been observed at that concurrency, so shorter runs retain the
1-minute exploration cadence. Once eligible, stable bandwidth is probed every
5 minutes. A bandwidth change of 2x or 0.5x relative to the long-term EMA
shortens that toward 1 minute, with intermediate changes interpolated in
logarithmic bandwidth space. Probe results use epoch-local goodput and p90
request latency; the long-term EMA is retained as the network-stability baseline
rather than used as the probe result itself.

An underfilled queue is not treated as reduced network capacity. Such samples
do not update the controller or stability EMA; the capacity estimate retains
the per-worker bandwidth measured at the greatest fully occupied concurrency.

Compare against fixed concurrency with `--mode fixed --workers N`. Payloads
are discarded after accounting by default. `--sink temp` writes each encoded
payload under a generated system temporary directory and removes it on exit;
use `--keep-temp` or `--temp-dir PATH` to retain the files. Use `--anonymous`
for unsigned requests to public S3 data.

The output reports one bandwidth metric: encoded bytes over the time covered by
up to the most recent `current admission x 4` successful, fully occupied chunk
downloads at the current admission. Before that window fills, every available
sample in that contiguous admission period is used. When the queue underfills,
the metric retains the capacity implied by the per-worker bandwidth at the
greatest fully occupied concurrency. While a benchmark is running, it prints
this bandwidth, queued chunks, active downloads, admission limit, and sample
count once per second and once more when the queue drains. Missing sparse chunks
are reported but are not counted as downloaded bytes.
