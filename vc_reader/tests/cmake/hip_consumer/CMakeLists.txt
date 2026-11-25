cmake_minimum_required(VERSION 3.22)
project(vc_reader_hip_consumer LANGUAGES CXX)

# Point vc_reader_DIR at the installed HIP package's CMake config, then build.
find_package(CUDAToolkit REQUIRED)
find_package(vc_reader CONFIG REQUIRED)
add_executable(hip_consumer main.cpp)
target_link_libraries(hip_consumer PRIVATE vc_reader::transforms_cuda)
