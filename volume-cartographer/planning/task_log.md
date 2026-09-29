# Sharded Zarr Double Decode

- Created fix/sharded-zarr-double-decode from fetched origin/main.
- extract_inner_chunk() already decompresses and converts byte order;
  decode_chunk_from_storage_object() repeats both via decode_chunk_payload().
- No changes to PR #1704, trailing-index support or shard writers are needed.
- The AGENTS-mentioned scripts/build_dependencies.sh no longer exists; inspected
  build_from_src_debian.sh and CMake instead. No installs/bootstrap performed.
- Independent plan review confirmed the fix and clarified explicit writer codec
  selection and the limited existing outer-codec endian contract. Incorporated
  both clarifications; nested-endian support is not changed.
- Before the fix, zstd/gzip storage-object checks and the endian check failed.
  A test-fixture reopen initially reused a process-cache identity after source
  invalidation; switched to a fresh identity, matching the existing reopen test.
- Real data: Paris4_2um_ps256.zarr level 3, chunks (18,8,8) and (18,8,9),
  each 256^3 uint8 with range 0..255. Repacked unchanged into one zstd shard
  in /tmp. Both direct reads matched; both storage-object decodes threw
  "ZSTD_decompress failed: Unknown frame descriptor" before the fix.
- After the fix, both real-data storage-object decodes matched the original
  16,777,216-byte chunks exactly, as did direct reads.
- New reopen coverage exposed unrelated Auto layout detection of legacy files
  inside a mirror. The test explicitly selects ZarrMirror to cover that path;
  changing Auto layout/cache write policy is outside this decode-only task.
- Validation passed: five CTest suites (test_zarr_chunk_fetcher, test_vcdataset,
  test_zarr, test_zarr_more, test_chunk_cache), five repeat fetcher-suite runs,
  and git diff --check. Independent code review found no actionable issues.
- Linux amd64 QuickBuild, existing system dependencies, no GUI or macOS run.
  Actual scroll voxels were repacked into a temporary shard; this is not a claim
  that the original Paris4 source was sharded. No performance claim is made.
- PR publication awaits approval of its preview. No personal-verification
  checkbox or GUI validation claim will be included.

## Commands

```sh
env AGENTS_AGENT_MODE=1 cmake --build volume-cartographer/build --target test_zarr_chunk_fetcher test_vcdataset test_zarr test_zarr_more test_chunk_cache -j32
env AGENTS_AGENT_MODE=1 ctest --test-dir volume-cartographer/build --output-on-failure -R '^(test_zarr_chunk_fetcher|test_vcdataset|test_zarr|test_zarr_more|test_chunk_cache)$'
env AGENTS_AGENT_MODE=1 ctest --test-dir volume-cartographer/build --output-on-failure --repeat until-fail:5 -R '^test_zarr_chunk_fetcher$'
git diff --check
```

Local real-data harness: /tmp/zarr-shard-double-decode-repro.cpp, compiled with
C++23/-O1 against the same build's vc_core, utils, codec and system libraries.
The source argument was
/home/hendrik/business/aiconsulting/vesuviuschallenge/data/s1/PHercParis4.volpkg/volumes/Paris4_2um_ps256.zarr/3;
outputs /tmp/paris4-shard-double-decode-before and -after contain the same two
source chunks in a 256x256x512 shard with 256^3 inner chunks. The before binary
is /tmp/zarr-shard-double-decode-repro and the after binary has suffix -fixed.
