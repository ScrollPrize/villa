include(FetchContent)

set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "" FORCE)

FetchContent_Declare(
    doctest
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://github.com/doctest/doctest/archive/v2.4.12.tar.gz
)
FetchContent_MakeAvailable(doctest)

add_executable(test_smoke test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE doctest::doctest)
add_test(NAME test_smoke COMMAND test_smoke)

add_executable(test_appendmask_render test_appendmask_render.cpp)
target_link_libraries(test_appendmask_render PRIVATE doctest::doctest vc_core)
add_test(NAME test_appendmask_render COMMAND test_appendmask_render)

add_executable(test_quadsurface_trim test_quadsurface_trim.cpp)
target_link_libraries(test_quadsurface_trim PRIVATE doctest::doctest vc_core)
add_test(NAME test_quadsurface_trim COMMAND test_quadsurface_trim)

add_executable(bench_volume_sample bench_volume_sample.cpp)
target_link_libraries(bench_volume_sample PRIVATE doctest::doctest vc_core)
# Bench has a generous wall-clock assertion; not part of default ctest
# unless the user opts in via VC_RUN_BENCHES. Expose it as a target so
# it's always built and runnable manually.
if(VC_RUN_BENCHES)
    add_test(NAME bench_volume_sample COMMAND bench_volume_sample)
endif()
