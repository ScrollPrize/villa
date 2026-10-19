# Training workflow acceptance and measurements

Validated on 2026-10-18 with CPython 3.14.4, Linux amd64, Intel Core Ultra 7
270K Plus and RTX 5090. Existing Release builds were used; no dependencies were
installed. Native kernels are unchanged; browsing now exposes `Asset.summary()`. The complete CPU suite
passed **130 tests and 1,376 subtests**, with 22 CUDA skips. The complete CUDA
suite passed **152 tests and 2,910 subtests**. Seven warnings concern existing
fork-based tests; the DataLoader benchmark uses spawn.

## Acceptance by workflow

The catalog file was `/home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin`,
SHA256 `55aa3d217732a3c75401765f7d425738a35fecab75baad6e4ef5cb8d8b9844e3`.
Its canonical source-ID fingerprint is
`de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d`.
Input IDs, levels and six exact requests are encoded in
[`training_inputs.py`](../benchmarks/training_inputs.py), used by the benchmark
and its validation tests, and included in every generated JSON report.

| Workflow | Recorded evidence | Remaining acceptance |
| --- | --- | --- |
| Published flattened pair | Six PHerc0009B CT/ink/supervision/validation reads equal independently opened Source arrays; first ink crop still has **9,926 nonzero pixels**; CPU/CUDA augmented outputs agree | None for this pinned pair |
| Custom pair | Explicit image/label overlays, pixel affine, exact categorical masks, JSON/pickle round trips and repeated spawned-worker epochs | None for this analytic fixture |
| CT-only | PHercParis4 `20230206171837`, level 0; candidate header verification; six crops equal independent volume reads; first crop has 64 valid voxels | None for this selected scan; the other candidate's contradictory shape still fails |
| Registered target | Real PHercParis4 `20260411134726`, CT level 2; surface-prediction model `20260413222639`, stored level 0, source ID ending `/data/2`; all six CT and prediction arrays equal independent reads; first crop has 512 valid voxels in each | Real categorical annotations are exercised through explicit surface projection below; geometric vector frames use analytic fixtures |
| Cross-volume transfer | Real PHerc0009B label set `20260918`, segment `20250919125754`, native volume `20250820154339` → `20250521125136`; six B CT and resampled A CT arrays independently checked; zero-thickness labels/coverage/conflicts checked with raw TIFF geometry and a scalar oracle; actual CPU/CUDA 9.6 µm projection and augmented outputs agree | Pixel-to-UV canvas convention is explicitly declared |
| Generated surface layers | Real PHerc0009B geometry and ink; offsets `(-9.6,0,9.6)` µm; every zero-offset CT tile checked using independent raw TIFF geometry and SciPy interpolation; actual CPU/CUDA readers and augmented targets agree | Declared canvas registration, not an anatomical alignment measurement |
| Projected surface labels | Same real geometry/label, thickness **9.6 µm**; all six CT boxes equal independent volume reads; zero-thickness labels/coverage/conflicts match the independent scalar oracle; 9.6 µm categorical values, coverage, conflicts and companion masks agree between actual CPU/CUDA readers and through augmentation | Positive-thickness categorical correctness additionally relies on the surface oracle regression suite |

The first generated tile has 768 valid CT samples (three 16×16 layers) and
256 valid label pixels. The first projected crop has 5,544 valid CT voxels and
2,003 covered label voxels. These geometry/data outputs are preserved; the
example applies no loss, conflict resolution policy, supervision threshold or
sample rejection. The declared `pixel_to_uv` convention and geometry source
are detailed in the [surface validation report](surface-sampling-benchmarks.md).

Independent references live in
[`training_reference.py`](../benchmarks/training_reference.py). Its geometric
oracle intentionally uses float64 to detect errors in production float32
arithmetic. The full surface tests retain independent categorical projection
oracles. The new tests cover image lifetime across reads/close, per-layer
validity with different holes at different depths, exact categorical mask
values, all field declarations, default collation, uint8 promotion, shared
worker caches, JSON/pickle overlays, persistent spawned workers, repeated epochs,
partial/no coverage, native byte order, writable broadcast masks, integer seeds,
registration errors, missing masks, early iterator closure
and retained prefetched outputs. Existing ambiguity, missing dependency and
resolution-failure regressions remain in the full suite.

The snapshot contains **33 active label assets and 171 mappings to other
volumes**. All 33 labels have a destination mapping. These are surface labels:
the registration chain is label → surface → native volume → catalog volume
affine → destination. Unresolved pixel-to-UV fields do not imply a missing
volume registration. The earlier missing-fixture conclusion was incorrect.
`label.summary()` now presents these independent facts together. Summarizing
all 33 labels took 1.06 ms in one metadata-only observation, importing no NumPy,
Torch or SciPy. The checked-in [catalog subset](../benchmarks/fixtures/training/catalog.json)
and [provenance](../benchmarks/fixtures/training/provenance.json) make the real
relationship regression reproducible without a local metadata cache.

For the recorded transfer, A→B uses derivation
`20250820154339-20250521125136⁻¹`; segment overlap is 1.0. Both scans use stored
level 0. The first projected B crop has origin ZYX `(6739,4323,5388)`, shape
`(7,4,6)`, 168 valid CT voxels in each scan and 85 covered label voxels at
9.6 µm thickness. Segment overlap does not guarantee arbitrary crop coverage.
The independent scan comparison uses the inverse catalog affine and SciPy
interpolation of raw A chunks (absolute tolerance 0.08 intensity units,
relative tolerance 2e-5). Zero-thickness categorical checks are exact for all
six requests. The scalar oracle retains deterministic clipped-polygon vertex
order because subdivision affects nearest-label selection at patch boundaries;
float64 half-pixel roundoff is removed before nearest-even sampling.

Real registered annotations and scan-to-scan transfer are now exercised.
macOS and arm64 runtime validation remain unperformed.

## Protocol and stage timings

Every workload uses six fixed requests, two warmup passes, and ten measured
passes. Per-batch stages process those six requests in three batches of two;
projection uses six batches of one because geometry determines crop shapes.
Seeds are explicitly 100–105. Workers persist across all measured epochs.
Reader threads and main-process PyTorch threads are both one; there are two
spawned workers. Every worker has its own shared 64 MiB chunk cache.

Timings exclude correctness comparisons, downloads of CUDA outputs, and source
fixture construction except where discovery/setup is explicitly reported.
Uploads and resident CUDA augmentation synchronize separately. Disk caches were
populated and OS caches were not cleared. These are not cold-network throughput
measurements. Initial population runs are kept separately. Main-process peak
RSS includes Torch/CUDA runtime and validation buffers and excludes workers;
it is not a cache budget or total GPU-memory measurement.

Published pair, milliseconds **mean / p50 / p95**:

| Stage | Unit | Mean / p50 / p95 ms |
| --- | --- | --- |
| Independent raw arrays | Crop | 0.070 / 0.070 / 0.077 |
| Paired reader, including validity/provenance | Crop | 0.123 / 0.123 / 0.132 |
| Dataset read plus selected dictionary | Crop | 0.132 / 0.130 / 0.141 |
| Dictionary/layout selection alone | Crop | 0.001 / 0.001 / 0.002 |
| Default collation | Two-crop batch | 0.124 / 0.104 / 0.264 |
| Pin host targets | Two-crop batch | 0.078 / 0.078 / 0.081 |
| Host-to-device upload | Two-crop batch | 0.069 / 0.052 / 0.120 |
| Resident CUDA normalization/flip | Two-crop batch | 0.737 / 0.733 / 0.758 |
| CPU normalization/flip | Single crop | 3.550 / 3.547 / 3.612 |
| Persistent DataLoader only | Six-crop epoch | 7.641 / 7.526 / 9.110 |
| Persistent DataLoader + CudaLoader | Six-crop epoch | 9.218 / 9.034 / 10.126 |

The Dataset adds approximately **8 µs per crop** relative to direct paired
reading. Independent raw reads omit the paired reader's geometric validity and
provenance, so their difference is not solely dictionary overhead. The
published-pair CPU augmentation takes about 7.09 ms for two samples, versus
0.73 ms on CUDA excluding upload. Stage medians are isolated measurements and
should not be summed to predict an overlapped epoch.

A first exploratory run with the environment's default PyTorch thread count
spent 25.2 ms per small collation. Explicitly setting one CPU thread reduced its
median to 0.10 ms. The example exposes this setting instead of creating a thread
pool implicitly inside each Dataset item.

Other workflows, medians in milliseconds (reader/Dataset per crop, CUDA per
batch, end-to-end per six-crop epoch):

| Workflow | Reader | Dataset | Resident CUDA | Loader + CUDA epoch |
| --- | ---: | ---: | ---: | ---: |
| CT-only, 4³ crops | 0.020 | 0.022 | 0.274 | 3.221 |
| Custom pair | 0.075 | 0.089 | 0.423 | 6.010 |
| Local registered targets | 0.101 | 0.110 | 0.422 | 5.148 |
| Local transfer | 0.121 | 0.130 | 0.498 | 7.034 |
| Real registered scalar target | 0.037 | 0.042 | 0.342 | 3.795 |
| Real generated layers | 0.492 | 0.499 | 0.294 | 7.804 |
| Real projection, batch size one | 7.891 | 7.864 | 0.199 | 30.385 |
| Real cross-volume projection, batch size one | 0.726 | 0.728 | 0.249 | 11.983 |

Small analytic crops and the 8³ real registered-target acceptance crop are
**slower on CUDA** than CPU even excluding upload: the real target takes about
0.026 ms for two CPU samples versus 0.342 ms for resident CUDA. This flags a
fixed-overhead regime of the existing batch executor; these tiny cases do not
establish a CUDA speedup. No new sampling or augmentation implementation was
introduced here. Use the representative 25×128×128 paired workload for the
training-stage comparison; larger 3D throughput is covered by the existing
[augmentation benchmarks](../benchmarks/README.md).

The real transfer uses small B-grid crops: its CPU augmentation takes 0.018 ms
per sample versus 0.249 ms resident CUDA, another fixed-overhead case rather
than a GPU speedup. Discovery and readers perform no work per whole volume.

## Opening, I/O and memory

Catalog loading/discovery/setup took approximately 475 ms for real workflows.
The cached CT-only reader opened in 7.52 ms and read its first crop in 0.80 ms.
Its initial online population instead took 675.90 ms to open and 141.58 ms
for the first crop, fetching one 2,097,152-byte chunk.
Single observed reader open / first request times were 8.72 / 2.70 ms for the
pair, 7.56 / 4.95 ms for the registered scalar target, 19.06 / 3.10 ms for layers,
and 18.97 / 11.27 ms for projection. These single observations are distinct
from the 60-request warm distributions. First DataLoader epochs took about
1.63–1.71 seconds, including spawn, catalog-view deserialization and reader
opening. Local fixtures needed about 0.65 seconds.

Warm reader passes fetched no additional remote data or disk chunks. Counters
before and after warm passes are recorded separately. Retained shared chunks
were 48,628,032 bytes for the pair, 9,175,040 for the registered target,
4,718,656 for layers and 2,621,504 for projection, counted once per reader set.
Layer sampling retained 55,117 bytes of tiles; projection retained 16,024 tile
bytes and 15,360 plan bytes. Peak sampling work was 73,728 bytes for layers and
233,820 for projection. These are separate from decoded TIFF caches and outputs.

Projection metrics separately report 4.36 ms cumulative geometry preparation,
1.46 ms candidate-index construction and 542.01 ms raster calls over first and
warm reads. These internal wall counters are not standalone GPU kernel timings;
all Dataset geometry is performed by CPU readers before upload. Existing surface
benchmarks compare explicit CPU/CUDA geometry execution in the device-owning
process.

Uploads per six requests were 5,308,416 bytes for the pair (including per-layer
validity), 12,288 for the registered scalar target, 29,184 for layers and
208,152 for projection. Main-process RSS peaks were roughly 0.95–1.09 GB with
the CUDA runtime loaded; raw reports include the precise values. Worker memory
and GPU allocator peaks are not measured by that RSS counter.

The cached real transfer opened in 20.06 ms; its first crop took 4.74 ms.
Warm reads reused all chunks with no new remote fetches. Shared retained chunks
were 6,815,808 bytes; retained tiles and plans were 16,024 and 19,584 bytes.
The six batches uploaded 11,592 bytes. The first worker epoch took 1.68 seconds;
subsequent six-crop epochs took a median 11.19 ms without augmentation and
11.98 ms with CudaLoader. The report includes surface preparation/index/raster
counters separately and full provenance for both scans and annotations.

## Reproduce

The [benchmark guide](../benchmarks/training.md) lists source selections and optional
dependencies. The linked plan overview was absent from this checkout; the
six/two/ten protocol was taken from the plan text and existing reader reports.
Use the existing builds without installation:

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks \
  VC_READER_NATIVE_DIR=build/cp314-cp314-linux_x86_64 \
  .venv/bin/python -m pytest tests -q
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks \
  VC_READER_NATIVE_DIR=/tmp/vc-reader-cuda \
  .venv/bin/python -m pytest tests -q

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_training.py paired \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --native-dir /tmp/vc-reader-cuda --cache-dir /tmp/vc-reader-paired-cache \
  --offline --cuda --output /tmp/training-paired.json

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_training.py published-target \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --native-dir /tmp/vc-reader-cuda --cache-dir /tmp/vc-reader-training-cache \
  --offline --cuda --output /tmp/training-published-target.json

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_training.py render \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --native-dir /tmp/vc-reader-cuda --cache-dir /tmp/vc-reader-surface-cache \
  --offline --cuda --output /tmp/training-render.json

AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_training.py project \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --native-dir /tmp/vc-reader-cuda --cache-dir /tmp/vc-reader-surface-cache \
  --offline --cuda --batch-size 1 --output /tmp/training-project.json
```

Run the real volume transfer using the same full snapshot to reproduce the
recorded source IDs, or use the checked-in catalog subset for a portable recipe:

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks .venv/bin/python benchmarks/bench_training.py published-transfer \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --native-dir /tmp/vc-reader-cuda --cache-dir /tmp/vc-reader-surface-cache \
  --offline --cuda --batch-size 1 --output /tmp/training-published-transfer.json
```

For a portable catalog selection, use
`--metadata benchmarks/fixtures/training/catalog.json`. Remove `--offline` for
initial cache population.

For `ct`, use the `published-target` command with `ct` and a distinct output
filename. For `custom`, `registered`, `transfer`, `local-render` and
`local-project`, change the workflow argument; metadata and remote caches are
not used. Omit `--cuda` and select the CPU build for CPU-only measurements.
The reports retain raw timing samples, array hashes, coverage, request metadata,
I/O counters, cache limits, commands and catalog SHA256. Copies are saved under
`docs/benchmarks/training/` following the repository's ignored-report convention.
