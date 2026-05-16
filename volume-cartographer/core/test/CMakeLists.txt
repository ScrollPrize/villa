include(FetchContent)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "" FORCE)

find_package(doctest CONFIG QUIET)
if(NOT TARGET doctest::doctest)
    FetchContent_Declare(
        doctest
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        URL https://github.com/doctest/doctest/archive/v2.4.12.tar.gz
    )
    FetchContent_MakeAvailable(doctest)
endif()

add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE doctest::doctest)
add_test(NAME test_smoke COMMAND test_smoke)

add_executable(test_appendmask_render test_appendmask_render.cpp)
target_link_libraries(test_appendmask_render PRIVATE doctest::doctest vc_core)
add_test(NAME test_appendmask_render COMMAND test_appendmask_render)

add_executable(test_chunked_plane_sampler_fallback test_chunked_plane_sampler_fallback.cpp)
target_link_libraries(test_chunked_plane_sampler_fallback PRIVATE doctest::doctest vc_core)
add_test(NAME test_chunked_plane_sampler_fallback COMMAND test_chunked_plane_sampler_fallback)

add_executable(test_quadsurface_save_roundtrip test_quadsurface_save_roundtrip.cpp)
target_link_libraries(test_quadsurface_save_roundtrip PRIVATE doctest::doctest vc_core)
add_test(NAME test_quadsurface_save_roundtrip COMMAND test_quadsurface_save_roundtrip)

add_executable(test_atomic_imwrite_multi test_atomic_imwrite_multi.cpp)
target_link_libraries(test_atomic_imwrite_multi PRIVATE doctest::doctest vc_core)
add_test(NAME test_atomic_imwrite_multi COMMAND test_atomic_imwrite_multi)

add_executable(test_post_process test_post_process.cpp)
target_link_libraries(test_post_process PRIVATE doctest::doctest vc_core)
add_test(NAME test_post_process COMMAND test_post_process)

# Pin the on-disk wire format: chunks must remain Blosc1-readable because
# VcDataset writes Zarr v2 metadata advertising compressor_id = "blosc".
# This catches any future migration to Blosc2 that would silently produce
# chunks numcodecs / zarr-python cannot decode.
#
# Links vc_core directly so it can call vc::testing::bloscCompressForTest,
# which exercises the *real* compress path used by VcDataset.
add_executable(test_blosc1_compat test_blosc1_compat.cpp)
target_link_libraries(test_blosc1_compat PRIVATE doctest::doctest vc_core Blosc::blosc)
add_test(NAME test_blosc1_compat COMMAND test_blosc1_compat)

# Parser-only regression tests for vc_merge_tifxyz: link the small static
# lib that holds gmResolveGrid + gmCheckConnected so the failure-message
# assertions are exact, without spawning the binary.
add_executable(test_merge_grid_resolve test_merge_grid_resolve.cpp)
target_link_libraries(test_merge_grid_resolve PRIVATE doctest::doctest vc_merge_grid)
add_test(NAME test_merge_grid_resolve COMMAND test_merge_grid_resolve)

# End-to-end test that spawns the built vc_merge_tifxyz binary on two
# synthetic 16x16 tifxyz dirs. Opt-in via VC_RUN_E2E=1; the test skips
# itself if unset, so the default ctest run stays fast.
add_executable(test_merge_e2e_small test_merge_e2e_small.cpp)
target_link_libraries(test_merge_e2e_small PRIVATE doctest::doctest vc_core nlohmann_json::nlohmann_json)
if(VC_RUN_E2E)
    add_test(NAME test_merge_e2e_small COMMAND test_merge_e2e_small)
    # The binary path probe in the test prefers VC_MERGE_TIFXYZ_BIN, so
    # CTest runs reliably regardless of the working directory.
    set_tests_properties(test_merge_e2e_small PROPERTIES
        ENVIRONMENT "VC_RUN_E2E=1;VC_MERGE_TIFXYZ_BIN=$<TARGET_FILE:vc_merge_tifxyz>")
endif()

add_executable(bench_volume_sample bench_volume_sample.cpp)
target_link_libraries(bench_volume_sample PRIVATE doctest::doctest vc_core)
# Bench has a generous wall-clock assertion; not part of default ctest
# unless the user opts in via VC_RUN_BENCHES. Expose it as a target so
# it's always built and runnable manually.
if(VC_RUN_BENCHES)
    add_test(NAME bench_volume_sample COMMAND bench_volume_sample)
endif()

# Sanitizer canaries: one exe, one subcommand per sanitizer. Each canary
# triggers a violation that the matching sanitizer must report; ctest
# expects the abort and matches the report text via PASS_REGULAR_EXPRESSION.
# A miss (sanitizer not actually instrumenting) makes the canary exit 0,
# which fails the test — that's the whole point.
set(_canary_srcs sanitizer_canary.cpp)
if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN OR VC_ENABLE_LSAN
   OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN)
    list(APPEND _canary_srcs "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.c")
endif()
add_executable(sanitizer_canary ${_canary_srcs})
set_target_properties(sanitizer_canary PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

# ctest: PASS_REGULAR_EXPRESSION must match in stdout/stderr for the test
# to pass. The canary aborts via the sanitizer runtime on success, or
# `exit(1)`s on the fall-through (sanitizer not actually instrumenting), so
# both the regex and a non-zero exit are required for a meaningful signal.
# Regexes are tight — generic substrings like "runtime error" can be emitted
# by unrelated failures and would mask a regression.
function(_add_canary name arg regex)
    add_test(NAME ${name} COMMAND sanitizer_canary ${arg})
    set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
endfunction()

if(VC_ENABLE_ASAN)
    _add_canary(sanitizer_canary_asan  asan  "AddressSanitizer: heap-buffer-overflow")
endif()
if(VC_ENABLE_UBSAN)
    _add_canary(sanitizer_canary_ubsan ubsan "runtime error: signed integer overflow")
endif()
if(VC_ENABLE_TSAN)
    _add_canary(sanitizer_canary_tsan  tsan  "ThreadSanitizer: data race")
endif()
if(VC_ENABLE_TYSAN)
    _add_canary(sanitizer_canary_tysan tysan "TypeSanitizer: type-aliasing-violation")
endif()
if(VC_ENABLE_NSAN)
    _add_canary(sanitizer_canary_nsan  nsan  "NumericalStabilitySanitizer: inconsistent")
endif()

add_executable(test_gridstore test_gridstore.cpp)
target_link_libraries(test_gridstore PRIVATE doctest::doctest vc_core)
add_test(NAME test_gridstore COMMAND test_gridstore)

add_executable(test_gridstore_branches test_gridstore_branches.cpp)
target_link_libraries(test_gridstore_branches PRIVATE doctest::doctest vc_core)
add_test(NAME test_gridstore_branches COMMAND test_gridstore_branches)

# Live network test: anonymous read of the PHerc 0172 zarr on AWS S3.
# Soft-skips on network failure so offline runs still pass; set
# VC_TEST_REQUIRE_NETWORK=1 to flip those skips into hard failures.
add_executable(test_pherc0172_live test_pherc0172_live.cpp)
target_link_libraries(test_pherc0172_live PRIVATE doctest::doctest vc_core)
add_test(NAME test_pherc0172_live COMMAND test_pherc0172_live)

add_executable(test_tiff test_tiff.cpp)
target_link_libraries(test_tiff PRIVATE doctest::doctest vc_core)
add_test(NAME test_tiff COMMAND test_tiff)

add_executable(test_normalgridtools test_normalgridtools.cpp)
target_link_libraries(test_normalgridtools PRIVATE doctest::doctest vc_core)
add_test(NAME test_normalgridtools COMMAND test_normalgridtools)

# Live S3: opens the PHerc 0172 volume and reads small chunks.
add_executable(test_volume_live_s3 test_volume_live_s3.cpp)
target_link_libraries(test_volume_live_s3 PRIVATE doctest::doctest vc_core)
add_test(NAME test_volume_live_s3 COMMAND test_volume_live_s3)

add_executable(test_normal_grid_volume test_normal_grid_volume.cpp)
target_link_libraries(test_normal_grid_volume PRIVATE doctest::doctest vc_core)
add_test(NAME test_normal_grid_volume COMMAND test_normal_grid_volume)

add_executable(test_surface_patch_index test_surface_patch_index.cpp)
target_link_libraries(test_surface_patch_index PRIVATE doctest::doctest vc_core)
add_test(NAME test_surface_patch_index COMMAND test_surface_patch_index)

add_executable(test_tiff_merge test_tiff_merge.cpp)
target_link_libraries(test_tiff_merge PRIVATE doctest::doctest vc_core)
add_test(NAME test_tiff_merge COMMAND test_tiff_merge)

add_executable(test_http_fetch_errors test_http_fetch_errors.cpp)
target_link_libraries(test_http_fetch_errors PRIVATE doctest::doctest vc_core)
add_test(NAME test_http_fetch_errors COMMAND test_http_fetch_errors)

# Live S3: stages PHerc 0172 level 5 to a local zarr dir, then exercises
# the NormalGridGenerate library API against it. Soft-skips on network.
add_executable(test_normal_grid_live test_normal_grid_live.cpp)
target_link_libraries(test_normal_grid_live PRIVATE doctest::doctest vc_core)
add_test(NAME test_normal_grid_live COMMAND test_normal_grid_live)

add_executable(test_normalgridtools_real test_normalgridtools_real.cpp)
target_link_libraries(test_normalgridtools_real PRIVATE doctest::doctest vc_core)
add_test(NAME test_normalgridtools_real COMMAND test_normalgridtools_real)

add_executable(test_tiff_more test_tiff_more.cpp)
target_link_libraries(test_tiff_more PRIVATE doctest::doctest vc_core)
add_test(NAME test_tiff_more COMMAND test_tiff_more)

add_executable(test_geometry test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE doctest::doctest vc_core)
add_test(NAME test_geometry COMMAND test_geometry)

add_executable(test_affine_transform test_affine_transform.cpp)
target_link_libraries(test_affine_transform PRIVATE doctest::doctest vc_core)
add_test(NAME test_affine_transform COMMAND test_affine_transform)

add_executable(test_point_index test_point_index.cpp)
target_link_libraries(test_point_index PRIVATE doctest::doctest vc_core)
add_test(NAME test_point_index COMMAND test_point_index)

add_executable(test_plane_surface test_plane_surface.cpp)
target_link_libraries(test_plane_surface PRIVATE doctest::doctest vc_core)
add_test(NAME test_plane_surface COMMAND test_plane_surface)

add_executable(test_compositing test_compositing.cpp)
target_link_libraries(test_compositing PRIVATE doctest::doctest vc_core)
add_test(NAME test_compositing COMMAND test_compositing)

add_executable(test_quadsurface_basics test_quadsurface_basics.cpp)
target_link_libraries(test_quadsurface_basics PRIVATE doctest::doctest vc_core)
add_test(NAME test_quadsurface_basics COMMAND test_quadsurface_basics)

add_executable(test_load_json test_load_json.cpp)
target_link_libraries(test_load_json PRIVATE doctest::doctest vc_core)
add_test(NAME test_load_json COMMAND test_load_json)

add_executable(test_remote_url test_remote_url.cpp)
target_link_libraries(test_remote_url PRIVATE doctest::doctest vc_core)
add_test(NAME test_remote_url COMMAND test_remote_url)

add_executable(test_logging test_logging.cpp)
target_link_libraries(test_logging PRIVATE doctest::doctest vc_core)
add_test(NAME test_logging COMMAND test_logging)

add_executable(test_plane_surface_branches test_plane_surface_branches.cpp)
target_link_libraries(test_plane_surface_branches PRIVATE doctest::doctest vc_core)
add_test(NAME test_plane_surface_branches COMMAND test_plane_surface_branches)

add_executable(test_affine_transform_surface test_affine_transform_surface.cpp)
target_link_libraries(test_affine_transform_surface PRIVATE doctest::doctest vc_core)
add_test(NAME test_affine_transform_surface COMMAND test_affine_transform_surface)

add_executable(test_quadsurface_more test_quadsurface_more.cpp)
target_link_libraries(test_quadsurface_more PRIVATE doctest::doctest vc_core)
add_test(NAME test_quadsurface_more COMMAND test_quadsurface_more)

add_executable(test_thinning test_thinning.cpp)
target_link_libraries(test_thinning PRIVATE doctest::doctest vc_core)
add_test(NAME test_thinning COMMAND test_thinning)

add_executable(test_umbilicus test_umbilicus.cpp)
target_link_libraries(test_umbilicus PRIVATE doctest::doctest vc_core)
add_test(NAME test_umbilicus COMMAND test_umbilicus)

add_executable(test_colormaps test_colormaps.cpp)
target_link_libraries(test_colormaps PRIVATE doctest::doctest vc_core)
add_test(NAME test_colormaps COMMAND test_colormaps)

add_executable(test_quadsurface_io test_quadsurface_io.cpp)
target_link_libraries(test_quadsurface_io PRIVATE doctest::doctest vc_core)
add_test(NAME test_quadsurface_io COMMAND test_quadsurface_io)

add_executable(test_quadsurface_pointto test_quadsurface_pointto.cpp)
target_link_libraries(test_quadsurface_pointto PRIVATE doctest::doctest vc_core)
add_test(NAME test_quadsurface_pointto COMMAND test_quadsurface_pointto)

add_executable(test_quadsurface_snapshot test_quadsurface_snapshot.cpp)
target_link_libraries(test_quadsurface_snapshot PRIVATE doctest::doctest vc_core)
add_test(NAME test_quadsurface_snapshot COMMAND test_quadsurface_snapshot)

add_executable(test_quadsurface_save_paths test_quadsurface_save_paths.cpp)
target_link_libraries(test_quadsurface_save_paths PRIVATE doctest::doctest vc_core)
add_test(NAME test_quadsurface_save_paths COMMAND test_quadsurface_save_paths)

add_executable(test_volume_local test_volume_local.cpp)
target_link_libraries(test_volume_local PRIVATE doctest::doctest vc_core)
add_test(NAME test_volume_local COMMAND test_volume_local)

add_executable(test_zarr_helpers test_zarr_helpers.cpp)
target_link_libraries(test_zarr_helpers PRIVATE doctest::doctest vc_core)
add_test(NAME test_zarr_helpers COMMAND test_zarr_helpers)

add_executable(test_chunked_plane_sampler test_chunked_plane_sampler.cpp)
target_link_libraries(test_chunked_plane_sampler PRIVATE doctest::doctest vc_core)
add_test(NAME test_chunked_plane_sampler COMMAND test_chunked_plane_sampler)

add_executable(test_slicing test_slicing.cpp)
target_link_libraries(test_slicing PRIVATE doctest::doctest vc_core)
add_test(NAME test_slicing COMMAND test_slicing)

add_executable(test_volume_pkg test_volume_pkg.cpp)
target_link_libraries(test_volume_pkg PRIVATE doctest::doctest vc_core)
add_test(NAME test_volume_pkg COMMAND test_volume_pkg)

add_executable(test_volpkg_convert test_volpkg_convert.cpp)
target_link_libraries(test_volpkg_convert PRIVATE doctest::doctest vc_core)
add_test(NAME test_volpkg_convert COMMAND test_volpkg_convert)

add_executable(test_vcdataset test_vcdataset.cpp)
target_link_libraries(test_vcdataset PRIVATE doctest::doctest vc_core)
add_test(NAME test_vcdataset COMMAND test_vcdataset)

add_executable(test_zarr test_zarr.cpp)
target_link_libraries(test_zarr PRIVATE doctest::doctest vc_core)
add_test(NAME test_zarr COMMAND test_zarr)

add_executable(test_chunk_cache test_chunk_cache.cpp)
target_link_libraries(test_chunk_cache PRIVATE doctest::doctest vc_core)
add_test(NAME test_chunk_cache COMMAND test_chunk_cache)

add_executable(test_volume_extras test_volume_extras.cpp)
target_link_libraries(test_volume_extras PRIVATE doctest::doctest vc_core)
add_test(NAME test_volume_extras COMMAND test_volume_extras)

add_executable(test_zarr_more test_zarr_more.cpp)
target_link_libraries(test_zarr_more PRIVATE doctest::doctest vc_core)
add_test(NAME test_zarr_more COMMAND test_zarr_more)

add_executable(test_vcdataset_compressors test_vcdataset_compressors.cpp)
target_link_libraries(test_vcdataset_compressors PRIVATE doctest::doctest vc_core)
add_test(NAME test_vcdataset_compressors COMMAND test_vcdataset_compressors)

add_executable(test_volume_static_read test_volume_static_read.cpp)
target_link_libraries(test_volume_static_read PRIVATE doctest::doctest vc_core)
add_test(NAME test_volume_static_read COMMAND test_volume_static_read)

add_executable(test_volume_pyramid test_volume_pyramid.cpp)
target_link_libraries(test_volume_pyramid PRIVATE doctest::doctest vc_core)
add_test(NAME test_volume_pyramid COMMAND test_volume_pyramid)

add_executable(test_chunk_cache_persist test_chunk_cache_persist.cpp)
target_link_libraries(test_chunk_cache_persist PRIVATE doctest::doctest vc_core)
add_test(NAME test_chunk_cache_persist COMMAND test_chunk_cache_persist)

add_executable(test_zarr_chunk_fetcher test_zarr_chunk_fetcher.cpp)
target_link_libraries(test_zarr_chunk_fetcher PRIVATE doctest::doctest vc_core)
add_test(NAME test_zarr_chunk_fetcher COMMAND test_zarr_chunk_fetcher)

add_executable(test_chunked_plane_sampler_more test_chunked_plane_sampler_more.cpp)
target_link_libraries(test_chunked_plane_sampler_more PRIVATE doctest::doctest vc_core)
add_test(NAME test_chunked_plane_sampler_more COMMAND test_chunked_plane_sampler_more)

add_executable(test_slicing_more test_slicing_more.cpp)
target_link_libraries(test_slicing_more PRIVATE doctest::doctest vc_core)
add_test(NAME test_slicing_more COMMAND test_slicing_more)

add_executable(test_vcdataset_more test_vcdataset_more.cpp)
target_link_libraries(test_vcdataset_more PRIVATE doctest::doctest vc_core)
add_test(NAME test_vcdataset_more COMMAND test_vcdataset_more)
