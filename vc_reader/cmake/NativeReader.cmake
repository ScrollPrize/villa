find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(BLOSC REQUIRED IMPORTED_TARGET blosc)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
pkg_check_modules(LZ4 REQUIRED IMPORTED_TARGET liblz4)
find_package(SQLite3 3.24 REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(TIFF REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
if(NOT TARGET vc_delta3d::vc_delta3d)
    find_package(vc_delta3d CONFIG QUIET)
endif()
if(NOT TARGET vc_delta3d::vc_delta3d)
    include(FetchContent)
    set(VC_DELTA3D_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(VC_DELTA3D_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(VC_DELTA3D_INSTALL_CPP ON CACHE BOOL "" FORCE)
    FetchContent_Declare(vc_delta3d
        GIT_REPOSITORY https://github.com/ScrollPrize/vc-delta3d.git
        GIT_TAG 103b502ad784df14c876100d87f601c0b6c46888) # v0.2.0
    FetchContent_MakeAvailable(vc_delta3d)
endif()
if(TARGET vc_delta3d)
    set_target_properties(vc_delta3d PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
# Embedding builds (the object store's Python module) link the reader statically so
# the module carries no shared library of its own to install.
option(VC_READER_SHARED_READER "Build the native reader as a shared library" ON)
if(VC_READER_SHARED_READER)
    add_library(vc_reader_native SHARED)
else()
    add_library(vc_reader_native STATIC)
endif()
target_sources(vc_reader_native PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/display/display.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/publication.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/native_reader.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/pathfinding/pathfinding.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/composite/reduce.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/computed.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/grid/grid.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/sources/aligned_reader.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/sources/caches.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/sources/image_file.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/sources/image_write.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/sources/resolution.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/sources/source_reader.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/sources/surface_reader.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/tifxyz/tifxyz.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/zarr/meta.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/zarr/access.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/zarr/layers.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/zarr/open.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/zarr/pyramid.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/zarr/store.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/zarr/store_metadata.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/zarr/writer.cpp")
# CPU and CUDA composites evaluate the same unfused float32 formulas (the kernels
# build with --fmad=false).
set_source_files_properties("${CMAKE_CURRENT_LIST_DIR}/../src/cpp/display/display.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/composite/reduce.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/grid/grid.cpp" PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>")
add_library(vc_reader::reader ALIAS vc_reader_native)
set_target_properties(vc_reader_native PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME reader)
target_compile_features(vc_reader_native PUBLIC cxx_std_20)
target_include_directories(vc_reader_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(vc_reader_native PUBLIC vc_reader::headers CURL::libcurl PkgConfig::BLOSC PkgConfig::ZSTD PkgConfig::LZ4
    SQLite::SQLite3 vc_delta3d::vc_delta3d ZLIB::ZLIB vc_reader::volcomp vc_reader::sftp Threads::Threads nlohmann_json::nlohmann_json
    PRIVATE vc_reader::catalog TIFF::TIFF PNG::PNG JPEG::JPEG)


if(VC_READER_ENABLE_CUDA)
    add_library(vc_reader_reader_cuda STATIC
        src/cpp/reader_cuda/kernels.cu src/cpp/aligned/kernels.cu src/cpp/projection/kernels.cu
        src/cpp/sheet_normal/kernels.cu src/cpp/geometry/kernels.cu src/cpp/composite/kernels.cu
        src/cpp/grid/kernels.cu src/cpp/pathfinding/kernels.cu src/cpp/display/kernels.cu)
    add_library(vc_reader::reader_cuda ALIAS vc_reader_reader_cuda)
    set_target_properties(vc_reader_reader_cuda PROPERTIES EXPORT_NAME reader_cuda
        POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    # Separate float32 rounding preserves geometric ties and filter agreement.
    set_source_files_properties(src/cpp/projection/kernels.cu src/cpp/sheet_normal/kernels.cu
        src/cpp/geometry/kernels.cu src/cpp/composite/kernels.cu src/cpp/grid/kernels.cu src/cpp/display/kernels.cu
        PROPERTIES COMPILE_OPTIONS "--fmad=false")
    target_compile_definitions(vc_reader_reader_cuda PUBLIC VC_READER_CUDA=1)
    target_link_libraries(vc_reader_reader_cuda PUBLIC vc_reader::headers CUDA::cudart)
    target_link_libraries(vc_reader_native PUBLIC vc_reader::reader_cuda)
endif()
