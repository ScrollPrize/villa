# volcomp (vendored)

Lossy `uint8` volumetric codec for micro-CT: 128³ chunks, 3-D DCT-16 with a
single quantiser step `q`, ~40× at q = 8 on scroll data (≈ 40 dB PSNR, P99
error ≈ 2.5 q). Upstream: <https://github.com/SuperOptimizer/volume-compressor>
(`volcomp.h` is copied verbatim; `spec/format.md` there is normative).

Vendored: volcomp 1.3.0, format revision 4 (upstream `f31b0e2`). Every chunk
mode of that revision decodes here: lossy DCT (mode 0), lossless (1–3), binary
masks (4, 5) and surface chunks (6: a DCT base plus a refinement that keeps
`(src >= thr) == (decoded >= thr)` exact, for thresholded probability maps).

Files:

- `volcomp.h` — upstream single header (all functions `static`).
- `volcomp_lib.{h,c}` — compiles the header once and exports a plain C surface
  (`volcomp_lib_encode`, `volcomp_lib_decode`, `volcomp_lib_decode_block`,
  `volcomp_lib_is_chunk`, `volcomp_lib_chunk_q`, `volcomp_lib_kernels`,
  `volcomp_lib_version`, `volcomp_lib_format_revision`,
  `volcomp_lib_surface_encode`, `volcomp_lib_surface_info`,
  `volcomp_lib_decode_smooth`).
- `utils/volcomp_codec.hpp` (in `utils/`) — the C++ shim used by `VcDataset`,
  and `ZarrChunkFetcher`, mirroring `c3d_codec.hpp`.

## Portability

No arch flags: on x86-64 the header compiles its AVX2+FMA kernels with a
target attribute and selects them at runtime when the CPU has them; arm64,
Windows/MinGW and x86-64 CPUs without AVX2 use its plain C kernels (about half
the AVX2 speed). `utils::volcomp_kernels()` reports `"avx2"` or `"c"`. Both
kernel sets decode to within ±1 LSB of each other (identical bytes with
clang/GCC on x86-64), so cached `.volcomp` chunks are portable across hosts.

## Decode-side smoothing is opt-in

Upstream's `volcomp_decode_smooth` (smooth the voxels next to interior block
faces, then project back onto the stream's quantisation cells) is exposed as
`utils::volcomp_decode_smooth_into`, but no VC reader calls it: plain
`volcomp_decode` stays the default everywhere. `volcomp_deblock` (not called by
VC either) now always leaves faces touching exact zeros alone.

## Where it plugs in

- Zarr v2: `.zarray` `"compressor": {"id": "volcomp", "q": Q}`.
- Zarr v3: codec `{"name": "volcomp", "configuration": {"q": Q}}`, usually as
  the inner codec of `sharding_indexed` with 128³ inner chunks (the published
  volcomp exports use 1024³ shards, `index_location: "end"`, crc32c index).
  `ZarrArray` reads and writes both index locations, with or without the
  crc32c (`test_zarr_shard_index_location`).
- `VcDataset` / `Volume` read and write it like any other compressor
  (`createZarrDataset(..., "volcomp", ..., compressionLevel = q)` requires
  uint8 and 128³ chunks).
- The streaming fetcher persists volcomp chunks verbatim in the disk cache
  (`.volcomp`), like `.c3d`.

## Updating

Copy the new upstream `volcomp.h` over this one verbatim (no amalgamation step;
it is already a single header) and update the version line above.
`volcomp_lib.c` only needs changes if the public API, the header magic/version
or the format revision changes; its `_Static_assert`s fail the build when the
revision moves past the one `volcomp_lib.h` documents. Run
`test_vcdataset_volcomp` (and, with `VC_VOLCOMP_LIVE_URL` set, its opt-in live
case) and `test_zarr_shard_index_location`.
