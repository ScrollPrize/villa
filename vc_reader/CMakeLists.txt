cmake_minimum_required(VERSION 3.20)
project(vc_reader_native LANGUAGES CXX)

# Native read path for vc_reader (module vc_reader._native), built by
# scikit-build-core. Needs libcurl, c-blosc, zstd and SQLite development files:
#   Ubuntu: apt install libcurl4-openssl-dev libblosc-dev libzstd-dev libsqlite3-dev
#   macOS:  brew install c-blosc zstd sqlite   (libcurl comes with the SDK)
# VC-Delta3D is fetched at a pinned commit (the same codec VC3D links).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

# The augmentation module has no dependency on reader networking or codecs.
nanobind_add_module(_augment NB_STATIC NOMINSIZE cpp/augment/bindings.cpp)
install(TARGETS _augment LIBRARY DESTINATION vc_reader)

option(VC_READER_BUILD_READER "Build the native Zarr reader as well as augmentation" ON)
if(VC_READER_BUILD_READER)
    find_package(CURL REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(BLOSC REQUIRED IMPORTED_TARGET blosc)
    pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
    find_package(SQLite3 3.24 REQUIRED)

    include(FetchContent)
    set(VC_DELTA3D_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(VC_DELTA3D_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(VC_DELTA3D_INSTALL_CPP OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(vc_delta3d
        GIT_REPOSITORY https://github.com/ScrollPrize/vc-delta3d.git
        GIT_TAG 457f612aa0379a73fd750d8b8e61eb8b11bf45fd  # v0.1.1
        GIT_SHALLOW FALSE)
    FetchContent_MakeAvailable(vc_delta3d)
    set_target_properties(vc_delta3d PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # NOMINSIZE: nanobind otherwise appends -Os, which costs the point samplers ~8%.
    nanobind_add_module(_native NB_STATIC NOMINSIZE cpp/bindings.cpp)
    target_include_directories(_native PRIVATE cpp)
    target_link_libraries(_native PRIVATE CURL::libcurl PkgConfig::BLOSC PkgConfig::ZSTD SQLite::SQLite3 vc_delta3d::vc_delta3d Threads::Threads)
    install(TARGETS _native LIBRARY DESTINATION vc_reader)
endif()
