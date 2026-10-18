# Scroll discovery and paired-reading validation

Measured during 2026-10-17–18 on Linux amd64, Intel Core Ultra 7 270K Plus,
CPython 3.14.4. Existing Release builds were used: CPU in
`build/cp314-cp314-linux_x86_64`, CUDA in `/tmp/vc-reader-cuda` (architecture 120).
No dependencies were installed and no native kernels changed.

## Real acceptance

The local catalog snapshot was
`/home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin`, SHA-256
`55aa3d217732a3c75401765f7d425738a35fecab75baad6e4ef5cb8d8b9844e3`.
The canonical metadata fingerprint in catalog source IDs is
`de38ba456eedfa4e0c4c6d6b4e093f4664d635eae551bfbe923bbdf6dc4ec58d`.

`ScrollSource('PHerc0009B', '20250820154339')` discovers segment
`20250919125754`, ink label set `20260918`, its render, and supervision/validation
masks. The crop at YX `(18576,7984)`, size `(128,128)`, layers `40:65`, returned:

- CT shape `(1,25,128,128)` and ink shape `(1,128,128)`.
- Exactly **9,926 nonzero ink pixels**.
- Byte-identical CT, ink, supervision, and validation compared with independently
  opened Zarr arrays on both CPU and CUDA.

The label package is sharded Zarr v3; the render is Zarr v2 with 109 sampled
layers per chunk. The published metadata explicitly links the label to that
render. Base pixel spacing and YX shapes agree in storage metadata. A source
transfer's canvas-offset history is not applied again to the published arrays.
The public artifact array list includes supervision and validation. A creation
report's mapping-validity output is absent from that list and from storage;
it is not added as a published companion.

PHerc0009B has published type `fragment`. `catalog.samples` and `ScrollSource`
retain it; `catalog.scrolls` filters to the 35 published scrolls.

Ambiguity was exercised on PHerc0139, volume `20260413113053`, segment
`20260302000001`: two Zarr renders remain discoverable and `.one()` raises.

The CT-only search `catalog.scrolls['PHercParis4'].volumes.at_resolution(8,
rel_tol=.02)` returns both 7.91 µm native candidates:

| Volume | Header verification |
| --- | --- |
| `20230206171837` | Confirmed spacing `(7.91,7.91,7.91)` µm and shape `(10529,7812,8316)`; a `4³` crop at `(5264,3906,4158)` read successfully, all 64 voxels nonzero. |
| `20230205180739` | Explicit failure: catalog shape `(14369,7888,8096)` contradicts storage shape `(14258,7888,8096)`. |

The README selects the confirmed volume. The contradictory candidate remains
inspectable; it is neither silently corrected nor silently dropped.

## Protocol and measurements

Paired-reader inputs are six fixed YX starts:
`(18576,7984)`, `(19344,14192)`, `(19856,15696)`, `(20496,13232)`,
`(22128,11472)`, `(23152,7984)`. Each reads 25 layers, 128×128 CT pixels,
ink, supervision, and validation. All six output hashes match direct readers.

There are two warmup passes and ten measured passes over all six requests.
Opening and first-read measurements also use two warmups and ten repetitions
with fresh readers. The table uses an already populated disk cache and
`--offline`; OS caches are not cleared. Each reader set shares a 64 MiB native
chunk-cache budget. CPU output is NumPy; CUDA output stays on device. Output
validation/download is outside timing. The CUDA direct baseline includes
uploads. Direct readers return array values; paired readers additionally
validate grids and return geometric validity and contextual metadata.

Values below are **mean / p50 / p95 milliseconds**. First-read rows summarize
60 requests through freshly opened readers; warm rows summarize 60 requests
through an already warmed reader.

| Backend and operation | Direct-reader baseline | Paired reader |
| --- | ---: | ---: |
| CPU opening | 1.372 / 1.366 / 1.416 | 2.955 / 2.952 / 3.052 |
| CPU first crop, disk warm | 1.139 / 1.129 / 1.286 | 1.457 / 1.314 / 2.305 |
| CPU warm crop | 0.054 / 0.054 / 0.058 | 0.124 / 0.122 / 0.134 |
| CUDA opening | 1.401 / 1.403 / 1.457 | 2.957 / 2.969 / 2.984 |
| CUDA first crop, disk warm | 1.285 / 1.261 / 1.592 | 1.623 / 1.516 / 2.029 |
| CUDA warm crop | 0.112 / 0.112 / 0.119 | 0.215 / 0.215 / 0.227 |

A separate initial online CPU pass opened in 2,742 ms and averaged 1,153 ms
for first requests, including network reads and cache population. Its direct
baseline then benefited from the populated disk cache, so those initial online
first-read times are not used for the comparison table.

I/O and memory are reported separately:

- Warm passes: zero remote bytes, zero new disk fetches; 24 image chunk hits and
  nine hits per annotation array per six requests.
- Retained shared chunk memory: **48,628,032 bytes (46.4 MiB)**, counted once.
  No decoded-image cache, affine tile cache, or plan cache is needed on these
  matching pixel grids. Transformed-grid tests exercise their configured bounds.
- CUDA transfers: **2,752,512 bytes per six requests**. Isolated host-to-device
  uploads average **0.048 / 0.047 / 0.052 ms** per request. Timed runs exclude
  device-to-host copies. CUDA output also requires device validity buffers;
  these direct reads do not exercise a GPU resampling speedup.

Metadata indexing adapts 4,495 sources without array requests. With catalog
parsing included, the pre-change index at `74aa6d007` measured
**375.2 / 370.3 / 407.0 ms**; the relationship index measured
**414.0 / 409.9 / 478.1 ms**, each with two warmups and ten repetitions.
The new index additionally builds forward/reverse render links and direct owner
indexes. Peak Python allocation including records was 36,035,796 bytes.

Repeated navigation across the six fixed PHercParis4 volumes averaged **7.16 µs**
per request, versus **12.01 µs** for a direct metadata scan. Five of these
contexts have no direct ink labels; one has eight. Raw sample distributions
and IDs are in the generated report. This comparison measures navigation,
not opening or an I/O speedup.

Separate cProfile passes found canonical JSON fingerprinting and immutable
source adaptation to dominate indexing (about 1.25 s and 0.88 s of 2.53 s
profiled time). In 60 profiled warm CPU requests, shared `_sample_many` accounts
for about 13 ms of 14 ms cumulative time, including about 6 ms in source reads
and 3 ms canonicalizing array axes. CUDA adds upload/allocation and device
validity work. Profiling overhead is excluded from the timing table.

## Reproduce with existing environments

From `vc_reader/`, run the independently verified pair example:

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src .venv/bin/python scripts/paired_reading.py \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --native-dir build/cp314-cp314-linux_x86_64
```

Repeat with `--native-dir /tmp/vc-reader-cuda --backend cuda`. The example's
cache defaults to `/tmp/vc-reader-paired-cache` and can be selected explicitly.
A network connection is needed to populate missing public data.

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src .venv/bin/python benchmarks/bench_discovery.py \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --output /tmp/scroll-discovery-after.json --profile /tmp/scroll-discovery-after.prof

AGENTS_AGENT_MODE=1 PYTHONPATH=src .venv/bin/python benchmarks/bench_paired.py \
  --metadata /home/sean/.cache/vc_reader/metadata-cff196b7c8a57b2d.bin \
  --native-dir build/cp314-cp314-linux_x86_64 \
  --cache-dir /tmp/vc-reader-paired-cache \
  --output /tmp/scroll-paired-cpu.json --profile /tmp/scroll-paired-cpu.prof
```

After the online pass populates all six requests, add `--offline` to reproduce
the comparable disk-warm table. Use `--backend cuda --native-dir
/tmp/vc-reader-cuda` for CUDA. Full generated reports are stored locally under
`docs/benchmarks/scroll-reading/`, following the repository's ignored-report
convention. The metadata baseline was run against an isolated archive of
`74aa6d007`, using `SourceCollection(Catalog(raw))` with the same two/ten protocol.

Use the existing extension loader to run the full source tree without installing:

```bash
AGENTS_AGENT_MODE=1 PYTHONPATH=src:benchmarks \
VC_READER_NATIVE_DIR=build/cp314-cp314-linux_x86_64 .venv/bin/python -c \
'from pathlib import Path; from bench_pipeline import load_extensions; load_extensions(Path("build/cp314-cp314-linux_x86_64")); import pytest; raise SystemExit(pytest.main(["tests", "-q"]))'
```

For CUDA, replace both build paths with `/tmp/vc-reader-cuda`. Tests need access
to localhost for their HTTP storage fixtures and to the GPU for CUDA coverage.
No rebuild is needed for this Python-only change. An already configured native
build can be refreshed with `cmake --build build/cp314-cp314-linux_x86_64 -j2`
when needed; no setup or install command was run for this work.

## Coverage and limits

The complete CPU suite passed 105 tests with 21 CUDA skips; the complete CUDA
suite passed 126 tests and 2,904 subtests. A subsequent large-canvas affine
regression is covered by the paired-reader suite on both backends.

The suite covers direct associations and reverse IDs, explicit mappings,
missing/ambiguous relations, supersession, fragments, unknown artifacts,
metadata-only imports, resolution tolerances and contradictory headers,
overlay serialization, local Zarr/TIFF/PNG data, registered companions,
independent pyramid layouts, integer/channel/layer indexing, clipping, empty
requests, exact 64-bit categorical values, CPU/CUDA agreement, bounded caches,
retained output ownership, and CPU worker reopening.

macOS and arm64 validation was unavailable. No platform-specific code or new
dependency was introduced. Generating new surface layers and projecting labels
into 3D remain future geometry operations; their requirements are visible in
discovery. Cold remote timing depends on network/storage conditions and the
render's full-depth chunk layout, so the disk-warm results should not be used
as download throughput estimates.
