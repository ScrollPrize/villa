# Label-reader acceptance measurements

Measurements use Python 3.14.4, NumPy 2.5.4, Pillow 12.3.0, tifffile
2026.10.15, Linux amd64, and Release native builds. Each method uses six fixed
crops, two warmups, and ten measured repetitions (60 crops). Correctness checks
compare every crop with the direct reader/decoder. Timing excludes validation
and separate allocation/profiling passes. OS caches are not cleared. New disk
caches and new readers distinguish cold source reads from warm region reads.

## Published ink labels

The real input is PHerc0009B segment `20250919125754`, label set `20260918`,
target volume `20250820154339`. Catalog snapshot SHA-256:
`de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d`.
Source ID:
`catalog:PHerc0009B/segment/20250919125754/snapshot/de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d/data/11`.

The [published array metadata](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0009B/segments/20250919125754-auto_grown_20250919055754487_inp_hr/ink-labels/2.401um-volume-20250820154339/20260918/inklabels.zarr/0/zarr.json)
describes a `(25360, 20520)` uint8 array, a `(25600, 20992)` shard, and
`(512, 512)` inner chunks using Blosc/Zstd level 3 with bitshuffle. The index
uses little-endian uint64 offsets/lengths and CRC32C. The supervision and
validation companions have the same spatial grid and were read separately.

Six `(128, 128)` crop starts are `(18576, 7984)`, `(19344, 14192)`,
`(19856, 15696)`, `(20496, 13232)`, `(22128, 11472)`, `(23152, 7984)`.
They were chosen from nonzero samples in the small level-5 pyramid and then
fixed for measurement. Their nonzero pixel counts were 9926, 16073, 12902,
6245, 16384, and 5546.

| CPU Release method | Mean ms/crop | p50 | p95 |
| --- | ---: | ---: | ---: |
| Direct shared Zarr reader | 0.00848 | 0.00857 | 0.00930 |
| Source reader | 0.01048 | 0.01063 | 0.01177 |

This compares wrapper overhead on the same native implementation. Opening the
label and its two companions took 1910 ms, including remote metadata. The six
cold label reads took 279.55, 169.00, 68.49, 69.18, 62.97, and 66.72 ms. They
fetched 37,827 payload bytes in ten range requests: one shard index plus nine
inner chunks. Retained native label chunks totaled 2,359,296 bytes, with
262,144 decoded bytes per inner chunk. Supervision and validation fetched
33,899 and 33,658 payload bytes respectively. The separate traced allocation
pass peaked at 7,856 Python bytes; process peak RSS was 642,236,416 bytes and
cannot be attributed to reader allocations alone. Metadata requests are excluded
from native payload counters. No complete shard was downloaded.

The profile of a warm 60-crop batch is dominated by native `read_many`, output
allocation, and NumPy shape/product handling. The source wrapper preserves the
native batch planner. Tracemalloc measures Python allocations, not native codec
scratch; process peak RSS includes imports/native runtime initialization and
is not a decoded-cache measurement. Image-cache estimates, native cache bytes,
traced allocations, and process RSS are reported separately.

Reproduce using a local snapshot and the configured CPU build:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from vc_reader import Catalog
catalog = Catalog.from_file('/home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin')
source = catalog.samples['PHerc0009B'].on_volume('20250820154339').segments['20250919125754'].labels.ink.one(label_set='20260918').source
Path('/tmp/plan02-label-source.json').write_text(json.dumps(source.to_dict()))
starts = [[18576,7984], [19344,14192], [19856,15696], [20496,13232], [22128,11472], [23152,7984]]
Path('/tmp/plan02-label-requests.json').write_text(json.dumps([[p,[128,128]] for p in starts]))
PY
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_source_storage.py \
  --source /tmp/plan02-label-source.json --requests /tmp/plan02-label-requests.json \
  --native-dir build/cp314-cp314-linux_x86_64 \
  --output /tmp/plan02-public-zarr-v3-ink-cpu.json
```

## Image formats

The bundled PHercParis4 geometry mask `benchmarks/fixtures/fiber_132487/mask.tif`
is a real `(25, 4800)` uint8 image. SHA-256:
`bcc69634665c8bcde3e0736232eea41e00c039f83ce52ce60244d041cc16166f`.
It is a geometry mask, not an ink-label dataset. PNG, JPEG quality 95, and tiled
Deflate TIFF inputs below are explicitly re-encoded derivatives of this file.
The preparation script records each derivative's hash and size.

| Format | Direct mean / p50 / p95 ms | Source mean / p50 / p95 ms |
| --- | --- | --- |
| Uncompressed TIFF | 0.09670 / 0.09614 / 0.10249 | 0.11983 / 0.11826 / 0.12961 |
| Deflate TIFF, 16×128 tiles | 0.35515 / 0.31761 / 0.33627 | 0.12569 / 0.12466 / 0.13319 |
| PNG | 0.12350 / 0.12281 / 0.12716 | 0.01715 / 0.01706 / 0.01835 |
| JPEG | 0.09000 / 0.08946 / 0.09381 | 0.01755 / 0.01724 / 0.01914 |

The baseline decodes a full compressed image/page on each crop, or maps an
uncompressed TIFF page. Source readers reuse decoded images/tiles. PNG/JPEG
still require an initial whole-image decode; warm cache speed does not imply
cheap random access to their encoded files. JPEG comparisons use its own decoded
values, without treating lossy samples as categorical labels.

| Format | First crop ms | Cold payload/logical bytes, six crops | Retained decode bytes | Peak estimated decode/scratch bytes |
| --- | ---: | ---: | ---: | ---: |
| Uncompressed TIFF | 0.347 | 19,200 mapped logical bytes | 0 | 0 |
| Deflate TIFF | 0.345 | 587 compressed tile bytes | 45,056 | 85,155 |
| PNG | 0.411 | 270 encoded bytes | 120,000 | 2,008,846 |
| JPEG | 0.392 | 2,308 encoded bytes | 120,000 | 2,010,884 |

These counters exclude header reads and do not measure physical disk traffic.
Zero mapped decode bytes means the OS mapping/page cache owns the source pages.
A separate regression reads a 2×3 crop from a 2048×3072 compressed uint16 TIFF
under a 100,000-byte working limit, fetching fewer than 1024 tile bytes.

```bash
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/prepare_source_formats.py --output-dir /tmp/plan02-format-fixtures
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_source_storage.py --input benchmarks/fixtures/fiber_132487/mask.tif --format tiff --output /tmp/plan02-mask-tiff.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_source_storage.py --input /tmp/plan02-format-fixtures/mask-deflate.tif --format tiff --output /tmp/plan02-mask-deflate.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_source_storage.py --input /tmp/plan02-format-fixtures/mask.png --format png --output /tmp/plan02-mask-png.json
AGENTS_AGENT_MODE=1 .venv/bin/python benchmarks/bench_source_storage.py --input /tmp/plan02-format-fixtures/mask.jpg --format jpeg --output /tmp/plan02-mask-jpeg.json
```

Reports include source IDs, content/crop hashes, exact requests, compression,
versions, cold timings, warm distributions, allocation counters, and profiles.
Local copies are retained in `docs/benchmarks/source-reading/`, which follows
the repository's ignored benchmark-output convention.
