cmake_minimum_required(VERSION 3.22)
project(pre_reader_cache_benchmark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(PRE_READER_SOURCE "" CACHE PATH "Historical villa checkout containing ChunkCache")
find_package(Threads REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_path(DELTA3D_INCLUDE_DIR vc_delta3d/codec.hpp REQUIRED)
find_library(DELTA3D_LIBRARY vc_delta3d REQUIRED)
find_library(ZSTD_LIBRARY zstd REQUIRED)
set(core "${PRE_READER_SOURCE}/volume-cartographer/core")
set(utils "${PRE_READER_SOURCE}/volume-cartographer/utils")
# Expose the real storage routines to a benchmark-only decoded-input adapter.
# No historical implementation or data structure is copied or modified.
file(READ "${core}/include/vc/core/render/ChunkCache.hpp" header)
string(REPLACE "friend class ChunkCacheService;" "friend class ChunkCacheService;\n    friend class CacheBenchmarkAccess;" header "${header}")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/vc/core/render")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/vc/core/render/ChunkCache.hpp" "${header}")
add_executable(bench_cache_pre_reader ../cache_scale.cpp
    "${core}/src/render/ChunkCache.cpp"
    "${core}/src/render/DecodedChunkCacheBudget.cpp"
    "${core}/src/render/ChunkRequestScheduler.cpp"
    "${core}/src/render/PersistentZarrCacheBudget.cpp"
    "${core}/src/Logging.cpp" "${utils}/src/Json.cpp")
target_compile_definitions(bench_cache_pre_reader PRIVATE VCR_BENCH_PRE_READER)
target_include_directories(bench_cache_pre_reader PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include"
    "${core}/include" "${utils}/include" "${DELTA3D_INCLUDE_DIR}")
target_link_libraries(bench_cache_pre_reader PRIVATE Threads::Threads nlohmann_json::nlohmann_json
    "${DELTA3D_LIBRARY}" "${ZSTD_LIBRARY}")
enable_testing()
add_test(NAME pre_reader_smoke COMMAND bench_cache_pre_reader 4096 8 1 0 .01 0 0 1 full)
