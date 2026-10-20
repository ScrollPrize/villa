find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(BLOSC REQUIRED IMPORTED_TARGET blosc)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
find_package(SQLite3 3.24 REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
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
        GIT_TAG 457f612aa0379a73fd750d8b8e61eb8b11bf45fd)
    FetchContent_MakeAvailable(vc_delta3d)
endif()
if(TARGET vc_delta3d)
    set_target_properties(vc_delta3d PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_library(vc_reader_native SHARED "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/native_reader.cpp")
add_library(vc_reader::reader ALIAS vc_reader_native)
set_target_properties(vc_reader_native PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME reader)
target_compile_features(vc_reader_native PUBLIC cxx_std_20)
target_include_directories(vc_reader_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(vc_reader_native PUBLIC vc_reader::headers CURL::libcurl PkgConfig::BLOSC PkgConfig::ZSTD
    SQLite::SQLite3 vc_delta3d::vc_delta3d Threads::Threads nlohmann_json::nlohmann_json)


if(VC_READER_ENABLE_CUDA)
    add_library(vc_reader_reader_cuda STATIC
        src/cpp/reader_cuda/kernels.cu src/cpp/aligned/kernels.cu src/cpp/surface/kernels.cu
        src/cpp/sheet_normal/kernels.cu src/cpp/geometry/kernels.cu)
    add_library(vc_reader::reader_cuda ALIAS vc_reader_reader_cuda)
    set_target_properties(vc_reader_reader_cuda PROPERTIES EXPORT_NAME reader_cuda
        POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    # Separate float32 rounding preserves geometric ties and filter agreement.
    set_source_files_properties(src/cpp/surface/kernels.cu src/cpp/sheet_normal/kernels.cu
        src/cpp/geometry/kernels.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
    target_compile_definitions(vc_reader_reader_cuda PUBLIC VC_READER_CUDA=1)
    target_link_libraries(vc_reader_reader_cuda PUBLIC vc_reader::headers CUDA::cudart)
    target_link_libraries(vc_reader_native PUBLIC vc_reader::reader_cuda)
endif()
