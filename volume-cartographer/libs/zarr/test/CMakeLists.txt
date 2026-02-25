add_executable(test_zarr test_zarr.cpp)
target_link_libraries(test_zarr PRIVATE vc_zarr)
target_include_directories(test_zarr PRIVATE ${PROJECT_SOURCE_DIR}/core/test)
add_test(NAME test_zarr COMMAND test_zarr)
