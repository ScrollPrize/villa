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

add_executable(test_render_pipeline test_render_pipeline.cpp)
target_link_libraries(test_render_pipeline PRIVATE doctest::doctest vc_core)
add_test(NAME test_render_pipeline COMMAND test_render_pipeline)

add_executable(test_vcdataset test_vcdataset.cpp)
target_link_libraries(test_vcdataset PRIVATE doctest::doctest vc_core)
add_test(NAME test_vcdataset COMMAND test_vcdataset)

add_executable(test_json test_json.cpp)
target_link_libraries(test_json PRIVATE doctest::doctest utils)
add_test(NAME test_json COMMAND test_json)

add_executable(test_edge_cases test_edge_cases.cpp)
target_link_libraries(test_edge_cases PRIVATE doctest::doctest vc_core)
add_test(NAME test_edge_cases COMMAND test_edge_cases)

add_executable(test_slicing test_slicing.cpp)
target_link_libraries(test_slicing PRIVATE doctest::doctest vc_core)
add_test(NAME test_slicing COMMAND test_slicing)

add_executable(test_pipeline_coverage test_pipeline_coverage.cpp)
target_link_libraries(test_pipeline_coverage PRIVATE doctest::doctest vc_core)
add_test(NAME test_pipeline_coverage COMMAND test_pipeline_coverage)

add_executable(test_volume test_volume.cpp)
target_link_libraries(test_volume PRIVATE doctest::doctest vc_core)
add_test(NAME test_volume COMMAND test_volume)

add_executable(test_render_post test_render_post.cpp)
target_link_libraries(test_render_post PRIVATE doctest::doctest vc_core)
add_test(NAME test_render_post COMMAND test_render_post)

add_executable(test_zarr_metadata test_zarr_metadata.cpp)
target_link_libraries(test_zarr_metadata PRIVATE doctest::doctest utils)
add_test(NAME test_zarr_metadata COMMAND test_zarr_metadata)

add_executable(test_remote_zarr test_remote_zarr.cpp)
target_link_libraries(test_remote_zarr PRIVATE doctest::doctest vc_core)
add_test(NAME test_remote_zarr COMMAND test_remote_zarr)
