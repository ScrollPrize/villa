# vc_reader benchmarks

Compare `vc_reader.reader.ZarrReader` with zarr-python, and with TensorStore
when it is installed, on real Vesuvius Challenge data. Every backend's output
is hashed and compared with the first backend's; a mismatch aborts the run.
Only the reads are timed: hashing, eviction and opening are not. Backends are
interleaved inside each repetition, so they see the same disk and network
conditions. Results are written as JSON to `results/` and printed as a table.

Requirements: the `bench` extra (numpy, urllib3, numcodecs, zarr>=3, s3fs,
tensorstore). With the development venv from the package README:
`.venv/bin/python benchmarks/bench_local.py`. TensorStore is skipped if it is
not installed.

## Local reads: `bench_local.py`

```
python benchmarks/bench_local.py                    # default array below, 3 reps
python benchmarks/bench_local.py --array /path/to/array.zarr/0 --reps 5
python benchmarks/bench_local.py --only crops64,batch --backends zarr,vc_reader:slab
```

The default array is the local copy of PHercParis4 volume 20260411134726
(2.4 µm, 78 keV), level 2, under
`/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data/`. It is 382 GB
of uncompressed 128³ uint8 chunks on an md RAID of NVMe drives, larger than
RAM. The array is only ever read.

Crops are placed uniformly at random over the whole array (keeping those whose
centre chunk is stored). The scenarios are:

| scenario | operation |
|---|---|
| `crops128`, `crops64` | 200 × 128³ / 500 × 64³ crops, one after another (latency) |
| `batch128` | 256 × 128³ crops in one request (`read_many`; a 32-thread pool for the others) |
| `block512` | 4 × 512³ blocks (inference tiles) |
| `crops64-hot` | 500 × 64³ crops inside one 512³ neighbourhood, with vc_reader's memory cache |

Before each repetition, `-cold` scenarios drop every touched chunk file from
the page cache with `posix_fadvise(DONTNEED)`, which needs no root (Linux
only; skipped on tmpfs). `-warm` scenarios run one untimed repetition first.

Local backends: `vc_reader:slab` is the Python engine, `vc_reader:native` the
C++ engine (memory maps), and `vc_reader:native-slab` the C++ engine with
`pread`. The `+memory` variants add each engine's memory cache.

## Remote reads: `bench_remote.py`

```
python benchmarks/bench_remote.py
python benchmarks/bench_remote.py --arrays ct --reps 5
```

The reads go to `s3://vesuvius-challenge-open-data` inside one fully populated
1024³ region of the same CT level (uncompressed) and of its level-2 surface
prediction (blosc-zstd, 192³ chunks). The scenarios are a 256×512×512 block,
12 sequential 128³ crops, the same 12 crops as one batch, and the batch through
vc_reader's disk cache: first filling it, then served warm from disk.
zarr-python runs with its default async concurrency (10) and with concurrency
32. Remote numbers are dominated by the network path to us-east-1 and vary a
lot between runs.
