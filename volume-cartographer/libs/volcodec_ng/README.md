# volcodec_ng (vendored, lossless)

Lossless `uint8` volumetric codec for micro-CT: one 128³ chunk per stream,
meant to sit inside zarr v3 `sharding_indexed` like volcomp. Every voxel
decodes exactly as it was stored. It is its own format, not a volcomp
revision: a chunk starts with the magic `VCNG`, and arrays name the codec
`volcodec_ng`. volcomp (`libs/volcomp`) is unchanged.

How it codes a chunk: each voxel is predicted from 12 voxels of the two
previous rows and the two previous slices, with weights fitted per chunk by
least squares and stored in the stream as integers. The prediction error is
coded with static canonical Huffman codes, one per activity class of the
neighborhood, into 4 interleaved bit streams so the decoder can work on 4
voxels at once. Constant chunks take 13 bytes; a chunk that would not shrink
is stored raw. The decoder is integer only, so every platform and build decodes
the same voxels, and the encoder writes the same bytes on every build.

Files (MIT, see `LICENSE`):

- `volcodec_ng_lossless.h`: the single header (all functions `static`). It is
  generated from Chase Hendrick's volcodec-ng source and keeps only the
  lossless chunk modes (2 raw, 3 constant, 5 lossless); its function bodies are
  verbatim copies, and its streams are byte-identical to that source's.
  Chunks of any other `VCNG` mode are refused with "chunk mode not supported".
- `volcodec_ng_lib.{h,c}`: compiles the header once and exports a plain C
  surface (`volcodec_ng_lib_encode`, `volcodec_ng_lib_decode`,
  `volcodec_ng_lib_is_chunk`, `volcodec_ng_lib_chunk_mode`,
  `volcodec_ng_lib_kernels`, `volcodec_ng_lib_format_revision`).
- `utils/volcodec_ng_codec.hpp` (in `utils/`): the C++ shim used by
  `VcDataset`, mirroring `volcomp_codec.hpp`.

## Portability

No arch flags: on x86-64 the header compiles AVX2 kernels for the predictor
with a target attribute and selects them at runtime; on x86-64 Linux the bit
coders are also built for Haswell and picked at load time. arm64 and other
targets use the plain C kernels. All of them give the same bytes.
`utils::volcodec_ng_kernels()` reports `"avx2"` or `"c"`.

## Where it plugs in

- Zarr v2: `.zarray` `"compressor": {"id": "volcodec_ng"}`.
- Zarr v3: codec `{"name": "volcodec_ng"}` (no configuration), usually as the
  inner codec of `sharding_indexed` with 128³ inner chunks, as for volcomp.
- `VcDataset` / `Volume` read and write it like any other compressor
  (`createZarrDataset(..., "volcodec_ng", ...)` requires uint8 and 128³
  chunks; the compression level is ignored).
- The streaming fetcher persists its chunks verbatim in the disk cache
  (`.volcodec_ng`), like `.volcomp`.

## Updating

Regenerate `volcodec_ng_lossless.h` from the codec source with its
`lossless_header.py` and replace this copy. `volcodec_ng_lib.c` only needs
changes if the API or the format revision changes; its `_Static_assert`s fail
the build when the revision moves. Run `test_vcdataset_volcodec_ng`: it pins
the encoded bytes of five synthetic chunks, so a format change fails it.
