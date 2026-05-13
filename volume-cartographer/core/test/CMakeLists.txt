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
