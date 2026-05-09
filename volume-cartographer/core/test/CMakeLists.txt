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

# Parser-only regression tests for vc_merge_tifxyz: link the small static
# lib that holds gmResolveGrid + gmCheckConnected so the failure-message
# assertions are exact, without spawning the binary.
add_executable(test_merge_grid_resolve test_merge_grid_resolve.cpp)
target_link_libraries(test_merge_grid_resolve PRIVATE doctest::doctest vc_merge_grid)
add_test(NAME test_merge_grid_resolve COMMAND test_merge_grid_resolve)

add_executable(bench_volume_sample bench_volume_sample.cpp)
target_link_libraries(bench_volume_sample PRIVATE doctest::doctest vc_core)
# Bench has a generous wall-clock assertion; not part of default ctest
# unless the user opts in via VC_RUN_BENCHES. Expose it as a target so
# it's always built and runnable manually.
if(VC_RUN_BENCHES)
    add_test(NAME bench_volume_sample COMMAND bench_volume_sample)
endif()
