# Task log

## Findings

- The active denominator begins on the first body callback instead of at HTTP
  request issue, excluding connection and TTFB latency.
- The benchmark still exposes and prints an `admission * 4` completion window.
- The scheduler retains a completion-rate fallback, and production chunk-cache
  code starts a measurement for local fetchers as well as remote fetchers.
- Initial 4x concurrency probing is a separate requested behavior and remains.
- Missing or failed remote chunks reset the complete adaptive epoch. A masked
  Zarr can therefore keep a clean-start benchmark at its minimum admission
  indefinitely even while successful payload downloads remain saturated.
- Admission growth was also paced only by successful payload completions. On
  sparse arrays this delayed an already-selected 4x probe in proportion to the
  fill-chunk miss rate.

## Deviations

- None.

## Validation

- `cmake --build volume-cartographer/build --target vc_zarr_download_bench -j2`
  passed.
- `cmake --build volume-cartographer/build --target test_chunk_cache
  test_zarr_chunk_fetcher VC3D -j2` passed.
- `volume-cartographer/build/bin/test_chunk_cache` passed 75 test cases,
  including mixed successful/missing clean-start probing.
- `volume-cartographer/build/core/test/test_zarr_chunk_fetcher` passed 17 test
  cases.
- `volume-cartographer/build/core/test/test_http_fetch_errors` passed 6 test
  cases.
- `cmake --build volume-cartographer/build -j2` passed.
- `ctest --test-dir volume-cartographer/build --output-on-failure -L vc-core
  -j2` passed all 132 tests.
- The production-default 256-candidate Paris4 level-0 benchmark reached
  admission 8 after 7 seconds and admission 32 after 27 seconds despite 193
  sparse misses among 256 candidates. It completed in 29.93 seconds with 63
  two-MiB payload chunks and 6.04 MiB/s final rolling bandwidth. The same run
  before miss-paced ramping completed in 33.65 seconds and reached only
  admission 14.
- The first complete test run exposed two stale test executables after the
  `IChunkFetcher` vtable change. A complete rebuild relinked them; both tests
  and the full label then passed.
- `git diff --check` passed.
