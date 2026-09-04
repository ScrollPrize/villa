# volcomp (vendored)

Lossy `uint8` volumetric codec for micro-CT: 128³ chunks, 3-D DCT-16 with a
single quantiser step `q`, ~40× at q = 8 on scroll data (≈ 40 dB PSNR, P99
error ≈ 2.5 q). Upstream: <https://github.com/SuperOptimizer/volume-compressor>
(`volcomp.h` is copied verbatim; `spec/format.md` there is normative).

Files:

- `volcomp.h` — upstream single header (all functions `static`).
- `volcomp_lib.{h,c}` — compiles the header once and exports a plain C surface
  (`volcomp_lib_encode`, `volcomp_lib_decode`, `volcomp_lib_decode_block`,
  `volcomp_lib_is_chunk`, `volcomp_lib_chunk_q`, `volcomp_lib_available`).
- `utils/volcomp_codec.hpp` (in `utils/`) — the C++ shim used by `VcDataset`,
  `ZarrChunkFetcher` and `vc_zarr_recompress`, mirroring `c3d_codec.hpp`.

## Platform gate

The upstream kernels are explicit AVX2+FMA, so the codec is built only on
x86-64 with GCC/Clang (the TU gets `-mavx2 -mfma`; the rest of VC keeps its
own arch flags). Elsewhere the `volcomp` target still exists and every entry
point returns `VOLCOMP_LIB_UNSUPPORTED`; `utils::volcomp_available()` reports
false, encode/decode throw, and `vc_zarr_recompress --codec volcomp` refuses
to start. The check also covers x86-64 CPUs without AVX2 at runtime. A portable
scalar path is an upstream item, not a VC one.

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
- `vc_zarr_recompress in out --codec volcomp --q 8` writes 1024³/128³ shards.

## Updating

Copy the new `volcomp.h` over this one; `volcomp_lib.c` only needs changes if
the public API or the header magic/version changes. Run
`test_vcdataset_volcomp` (and, with `VC_VOLCOMP_LIVE_URL` / `VC_RECOMPRESS_BIN`
set, its opt-in live and end-to-end cases).
