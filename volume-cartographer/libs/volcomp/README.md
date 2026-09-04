# volcomp (vendored)

Lossy `uint8` volumetric codec for micro-CT: 128³ chunks, 3-D DCT-16 with a
single quantiser step `q`, ~40× at q = 8 on scroll data (≈ 40 dB PSNR, P99
error ≈ 2.5 q). Upstream: <https://github.com/SuperOptimizer/volume-compressor>
(`volcomp.h` is copied verbatim; `spec/format.md` there is normative).

Files:

- `volcomp.h` — upstream single header (all functions `static`).
- `volcomp_lib.{h,c}` — compiles the header once and exports a plain C surface
  (`volcomp_lib_encode`, `volcomp_lib_decode`, `volcomp_lib_decode_block`,
  `volcomp_lib_is_chunk`, `volcomp_lib_chunk_q`, `volcomp_lib_kernels`).
- `utils/volcomp_codec.hpp` (in `utils/`) — the C++ shim used by `VcDataset`,
  and `ZarrChunkFetcher`, mirroring `c3d_codec.hpp`.

## Portability

No arch flags: on x86-64 the header compiles its AVX2+FMA kernels with a
target attribute and selects them at runtime when the CPU has them; arm64,
Windows/MinGW and x86-64 CPUs without AVX2 use its plain C kernels (about half
the AVX2 speed). `utils::volcomp_kernels()` reports `"avx2"` or `"c"`. Both
kernel sets decode to within ±1 LSB of each other (identical bytes with
clang/GCC on x86-64), so cached `.volcomp` chunks are portable across hosts.

## Where it plugs in

- Zarr v2: `.zarray` `"compressor": {"id": "volcomp", "q": Q}`.
- Zarr v3: codec `{"name": "volcomp", "configuration": {"q": Q}}`, usually as
  the inner codec of `sharding_indexed` with 128³ inner chunks (the published
  volcomp exports use 1024³ shards, `index_location: "end"`, crc32c index).
- `VcDataset` / `Volume` read and write it like any other compressor
  (`createZarrDataset(..., "volcomp", ..., compressionLevel = q)` requires
  uint8 and 128³ chunks).
- The streaming fetcher persists volcomp chunks verbatim in the disk cache
  (`.volcomp`), like `.c3d`.

## Updating

Copy the new `volcomp.h` over this one; `volcomp_lib.c` only needs changes if
the public API or the header magic/version changes. Run
`test_vcdataset_volcomp` (and, with `VC_VOLCOMP_LIVE_URL` / `VC_RECOMPRESS_BIN`
set, its opt-in live and end-to-end cases).
