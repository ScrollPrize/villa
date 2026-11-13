# Delta3D volume and cache integration

VC3D uses the native reader's Delta3D integration, pinned to
[vc-delta3d v0.3.2](https://github.com/ScrollPrize/vc-delta3d/releases/tag/v0.3.2).
Installed C++ packages must be version 0.3.2 or newer so builds can decode the
current preset frames. The embedding regression uses the same release.

The default is `fast`. Lossless uint8 chunks with dimensions divisible by 32 use
the new framing. The codec automatically selects legacy Delta3D for 16-bit,
non-aligned, or quantized chunks. Cache losslessness is checked through the codec
API, not by inspecting D3D1 header bytes. Zarr compressor metadata and cache marker
files remain unchanged.

D3D1 Delta and VCZ1 data remain readable. The 0.3 releases reject retired LPC
frames and experimental preset formats from before 0.3.0; re-encode those with
their original codec before upgrading. Older readers that only support D3D1
cannot read new preset frames: upgrade consumers before sharing newly encoded
volumes or caches with them.

## Tests

With the normal project build dependencies available:

```sh
cmake -S vc_reader -B <reader-build> -DVC_READER_BUILD_PYTHON=OFF \
  -DVC_READER_ENABLE_CUDA=OFF -DVC_READER_BUILD_NATIVE_TESTS=ON
cmake --build <reader-build> --target test_delta3d test_native_codecs \
  test_native_zarr test_native_zarr_writer
ctest --test-dir <reader-build> \
  -R '^(test_delta3d|test_native_codecs|test_native_zarr|test_native_zarr_writer)$' \
  --output-on-failure

# Minimal embedding regression: Qt autogen must not leak into codec C kernels.
cmake -S vc_reader/tests/cmake/delta3d_embedding -B <embedding-build> \
  -DCMAKE_BUILD_TYPE=Release
cmake --build <embedding-build> --parallel
ctest --test-dir <embedding-build> --output-on-failure
```
