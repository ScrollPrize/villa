# Delta3D cache integration

VC3D and the native reader use the preset-enabled Delta3D codec from
[vc-delta3d#7](https://github.com/ScrollPrize/vc-delta3d/pull/7), pinned to
`45339743b450ff707d21fb742dc1b3b2bf99306d`. Both CMake fetches and the VC Python
dependency use that revision. Installed C++ packages must be version 0.3 or newer;
an installed 0.2 package must not silently override this integration.

The default is `fast`. Lossless uint8 chunks with dimensions divisible by 32 use
the new framing. The codec automatically selects legacy Delta3D for 16-bit,
non-aligned, or quantized chunks. Cache losslessness is checked through the codec
API, not by inspecting D3D1 header bytes. Zarr compressor metadata and cache marker
files remain unchanged.

Existing Delta3D and VCZ1 data remain readable. The explicit Python VCZ1 adapter
continues writing VCZ1, selecting the Delta predictor deliberately. Older readers
that only support D3D1 cannot read new preset frames: upgrade consumers before
sharing a newly written cache with them. Unrelated Python projects in the
monorepo are not upgraded by this change.

## Tests

With the normal project build dependencies available:

```sh
cmake --build <vc-build> --target test_cache_compression
ctest --test-dir <vc-build> -R '^test_cache_compression$' --output-on-failure

cmake -S vc_reader -B <reader-build> -DVC_READER_BUILD_PYTHON=OFF \
  -DVC_READER_ENABLE_CUDA=OFF -DVC_READER_BUILD_NATIVE_TESTS=ON
cmake --build <reader-build> --target test_delta3d
ctest --test-dir <reader-build> -R '^test_delta3d$' --output-on-failure

python -m pytest volume-cartographer/python/tests/test_vcz1_codec.py
```

`test_cache_compression` additionally accepts `VC_DELTA3D_TEST_CHUNK`, a path to
a raw 128³ uint8 chunk. It checks exact roundtrips through the VC cache facade for
both encoders and prints complete encoded sizes. This test has no Qt dependency
and can also be compiled directly with the in-tree test harness and Delta3D.

### Integration validation

GCC 13.3, `-O2`, C++20; six real 128³ uint8 Paris4 inference chunks from
`PHercParis4-20260411134726-las-sd1-9ef89ac3`, level 2, chunk coordinate `(5,33,15)`.
Each input is 2,097,152 bytes. Both encoders reconstruct identical input bytes.
These are integration checks, not a new throughput benchmark.

| Channel | Delta bytes | Default fast bytes |
|---|---:|---:|
| nx | 532,044 | 412,772 |
| ny | 635,660 | 473,303 |
| presence | 577,780 | 380,894 |
| sheet_nx | 589,344 | 428,684 |
| sheet_ny | 564,196 | 411,821 |
| width | 487,384 | 366,605 |

Native cache-facade tests and all eight Python VCZ1 tests passed. Full VC3D/reader
build validation was unavailable on the test host: reader CMake configuration
stopped at missing `nlohmann_json`; Blosc development headers were also absent.
The new reader codec regression test is registered for builds with those dependencies.
