cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(VC3D VERSION 1.0.0)

# We don't use C++20/23 modules. CMake 3.28+ would otherwise invoke
# clang-scan-deps on every C++ source to discover module imports, which
# requires the `clang-tools-N` package that isn't part of Ubuntu 24.04's
# default `clang` install. Skip the scan globally.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VC_MARCH_NATIVE     "Tune for the host CPU (-march=native)"        OFF)
option(VC_WARNINGS        "Enable curated warning set"                   OFF)
option(VC_WERROR          "Treat compiler warnings as errors"            OFF)
option(VC_ENABLE_ASAN     "AddressSanitizer"                             OFF)
option(VC_ENABLE_UBSAN    "UndefinedBehaviorSanitizer"                   OFF)
option(VC_ENABLE_TSAN     "ThreadSanitizer"                              OFF)
option(VC_ENABLE_LSAN     "LeakSanitizer (standalone)"                   OFF)
option(VC_ENABLE_TYSAN    "TypeSanitizer (clang-only, LLVM 18+)"         OFF)
option(VC_ENABLE_NSAN     "NumericalStabilitySanitizer (clang-only, LLVM 19+)" OFF)
option(VC_ENABLE_COVERAGE "Code coverage instrumentation"                OFF)
option(VC_TESTING         "Build and run unit tests"                     OFF)
option(VC_DEAD_CODE_REPORT "Enable dead-code reporting (curated -Wunused* + linker --print-gc-sections; use a Debug build to avoid LTO)" OFF)
option(VC_PROFILE_COMPILE  "Emit per-TU compile-time profiles (-ftime-trace on clang, -ftime-report on gcc)" OFF)

# ---- Toolchain --------------------------------------------------------------
foreach(_path "$ENV{HOME}/miniconda3" "$ENV{HOME}/anaconda3")
    if(EXISTS "${_path}")
        list(APPEND CMAKE_IGNORE_PREFIX_PATH "${_path}")
    endif()
endforeach()

# ---- Git revision (used for ProjectInfo::VersionString) ---------------------
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${CMAKE_SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE GIT_SHA1 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${CMAKE_SOURCE_DIR}" log -1 --format=%cs HEAD
        OUTPUT_VARIABLE GIT_COMMIT_DATE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(GIT_SHA1)
    string(SUBSTRING ${GIT_SHA1} 0 7 GIT_SHA1_SHORT)
else()
    set(GIT_SHA1 Untracked)
    set(GIT_SHA1_SHORT Untracked)
    set(GIT_COMMIT_DATE Untracked)
endif()
set(VC_VERSION_STRING "${GIT_SHA1_SHORT}-${GIT_COMMIT_DATE}")

# ---- Base + build-type flags ------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(VC_DEAD_CODE_REPORT)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# Portable shippable defaults; VC_MARCH_NATIVE=ON (set by dev-* presets) opts
# into host tuning. armv8.2-a floor avoids SVE so Apple M1-M3 + Qualcomm Oryon
# work alongside Graviton.
if(VC_MARCH_NATIVE)
    add_compile_options(-pipe -march=native)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    add_compile_options(-pipe -march=x86-64-v3)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_compile_options(-pipe -march=armv8.2-a)
else()
    add_compile_options(-pipe)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_dbg -g3 -fno-omit-frame-pointer -fasynchronous-unwind-tables -fstandalone-debug)
    set(_lto -flto=thin)
    if(APPLE)
        set(_link "")
    else()
        set(_link -fuse-ld=lld -Wl,--gc-sections -Wl,--as-needed)
    endif()
else()
    set(_dbg -g3 -fno-omit-frame-pointer -fasynchronous-unwind-tables -fno-eliminate-unused-debug-types)
    set(_lto -flto=auto)
    if(APPLE)
        set(_link "")
    else()
        find_program(VC_MOLD_BINARY mold)
        if(VC_MOLD_BINARY)
            set(_link -fuse-ld=mold -Wl,--gc-sections -Wl,--as-needed)
        else()
            set(_link -Wl,--gc-sections -Wl,--as-needed)
        endif()
    endif()
endif()
add_compile_options(${_dbg} -ffunction-sections -fdata-sections)
add_link_options(${_link})

# ---- Sanitizers + coverage --------------------------------------------------
if(VC_ENABLE_ASAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "ASan + TSan are mutually exclusive")
endif()
if(VC_ENABLE_LSAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "LSan + TSan are mutually exclusive")
endif()
if(VC_ENABLE_TYSAN AND (VC_ENABLE_ASAN OR VC_ENABLE_TSAN OR VC_ENABLE_NSAN))
    message(FATAL_ERROR "TySan + (ASan/TSan/NSan) are mutually exclusive")
endif()
if(VC_ENABLE_NSAN AND (VC_ENABLE_ASAN OR VC_ENABLE_TSAN))
    message(FATAL_ERROR "NSan + (ASan or TSan) are mutually exclusive")
endif()
if((VC_ENABLE_TYSAN OR VC_ENABLE_NSAN) AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "TySan / NSan require clang")
endif()

if(VC_ENABLE_ASAN)
    list(APPEND _san -fsanitize=address,leak -DADDRESS_SANITIZER)
endif()
if(VC_ENABLE_UBSAN)
    list(APPEND _san -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND _san -fsanitize=nullability)
    endif()
endif()
if(VC_ENABLE_TSAN)
    list(APPEND _san -fsanitize=thread)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
if(VC_ENABLE_TYSAN)
    list(APPEND _san -fsanitize=type)
endif()
if(VC_ENABLE_NSAN)
    list(APPEND _san -fsanitize=numerical)
endif()
if(VC_ENABLE_LSAN AND NOT VC_ENABLE_ASAN)
    list(APPEND _san -fsanitize=leak)
endif()
# Sanitizers default to "recover and continue" on most checks, which lets
# CI silently pass on real UB / type confusion / numerical issues. Force
# abort on first detection across every enabled sanitizer.
if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN
   OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN OR VC_ENABLE_LSAN)
    list(APPEND _san -fno-sanitize-recover=all)
endif()
if(VC_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND _san -fprofile-instr-generate -fcoverage-mapping)
    else()
        list(APPEND _san --coverage)
    endif()
endif()
# Sanitizer/coverage flags applied below, after libs/ — see that block.

# ---- Optimization + LTO -----------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-O2)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_compile_options(-Os)
endif()
if(NOT _san AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(${_lto})
    add_link_options(${_lto})
endif()

# ---- Warnings ---------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-unknown-warning-option)
else()
    add_compile_options(-Wno-psabi -Wno-ignored-attributes -Wno-narrowing)
endif()

if(VC_WARNINGS AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
        -Weverything
        -Wno-padded
        -Wno-unsafe-buffer-usage -Wno-unsafe-buffer-usage-in-libc-call -Wno-unsafe-buffer-usage-in-container
        -Wno-ctad-maybe-unsupported -Wno-disabled-macro-expansion
        -Wno-exit-time-destructors -Wno-global-constructors
        -Wno-newline-eof -Wno-extra-semi -Wno-extra-semi-stmt
        -Wno-weak-vtables -Wno-packed -Wno-nrvo
        -Wno-missing-prototypes -Wno-missing-variable-declarations
        -Wno-reserved-identifier -Wno-reserved-macro-identifier
        -Wno-covered-switch-default -Wno-switch-enum -Wno-switch-default
        -Wno-source-uses-openmp -Wno-old-style-cast
        -Wno-zero-as-null-pointer-constant -Wno-format-nonliteral
        -Wno-comment -Wno-documentation -Wno-documentation-unknown-command -Wno-comma
        -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat-unnamed-type-template-args
        -Wno-c++98-compat-extra-semi
        -Wno-c++11-compat -Wno-c++14-compat -Wno-c++17-compat -Wno-c++20-compat
        -Wno-pre-c++14-compat -Wno-pre-c++17-compat
        -Wno-pre-c++20-compat -Wno-pre-c++23-compat -Wno-pre-c++26-compat
        -Wno-nested-anon-types -Wno-gnu-anonymous-struct
    )
elseif(VC_WARNINGS)
    add_compile_options(
        -Wall -Wextra -pedantic
        -Wattributes -Wcast-align -Wcast-qual -Wchar-subscripts -Wcomment
        -Wconversion -Wdelete-incomplete -Wdelete-non-virtual-dtor
        -Wenum-compare -Wmain -Wmissing-field-initializers -Wmissing-noreturn
        -Wold-style-cast -Woverloaded-virtual -Wpointer-arith
        -Wtautological-compare -Wundef -Wuninitialized -Wunreachable-code
        -Wunused -Wvla -Wunused-parameter
        -Wlogical-op -Wduplicated-cond -Wduplicated-branches
        -Wnull-dereference -Wuseless-cast -Wsuggest-override
    )
endif()
if(VC_WERROR)
    add_compile_options(-Werror)
endif()

if(VC_PROFILE_COMPILE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-ftime-trace -ftime-trace-granularity=200)
    else()
        add_compile_options(-ftime-report)
    endif()
endif()

if(VC_DEAD_CODE_REPORT)
    add_compile_options(
        -Wunused -Wunused-function -Wunused-variable -Wunused-parameter
        -Wunused-value -Wunused-result
        -Wunused-but-set-variable -Wunused-but-set-parameter
        -Wunused-local-typedef
        -Wunused-macros
        -Wunreachable-code
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(
            -Wunused-template
            -Wunused-member-function
            -Wunused-private-field
            -Wunreachable-code-aggressive
        )
    endif()
    add_link_options(-Wl,--print-gc-sections)
endif()

# ---- Dependencies -----------------------------------------------------------
find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network Concurrent OpenGLWidgets)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)
find_package(TIFF REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
find_package(PkgConfig REQUIRED)
# Blosc1 (libblosc-dev on Debian/Ubuntu; c-blosc on Homebrew). No upstream
# CMake config is shipped, so locate by header+lib and wrap in an imported
# target consumers link as Blosc::blosc.
#
# Note on package collision: libblosc-dev installs <blosc.h>; libblosc2-dev
# installs <blosc2.h> (a *different* header, not a compat shim). Searching
# `NAMES blosc.h` therefore only matches Blosc1. The matching libraries
# also have distinct SONAMEs (libblosc.so vs libblosc2.so), so a system
# with both packages installed cannot accidentally hybridize.
find_path(BLOSC_INCLUDE_DIR NAMES blosc.h
    HINTS $ENV{HOMEBREW_PREFIX}/opt/c-blosc/include
          $ENV{HOMEBREW_PREFIX}/include)
find_library(BLOSC_LIBRARY NAMES blosc
    HINTS $ENV{HOMEBREW_PREFIX}/opt/c-blosc/lib
          $ENV{HOMEBREW_PREFIX}/lib)
if(NOT BLOSC_INCLUDE_DIR OR NOT BLOSC_LIBRARY)
    message(FATAL_ERROR "Blosc1 (libblosc-dev / c-blosc) not found")
endif()
if(NOT TARGET Blosc::blosc)
    add_library(Blosc::blosc UNKNOWN IMPORTED)
    set_target_properties(Blosc::blosc PROPERTIES
        IMPORTED_LOCATION "${BLOSC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIR}")
endif()
pkg_check_modules(zstd REQUIRED IMPORTED_TARGET libzstd)
pkg_check_modules(lz4 REQUIRED IMPORTED_TARGET liblz4)
if(NOT TARGET zstd::libzstd)
    add_library(zstd::libzstd ALIAS PkgConfig::zstd)
endif()
if(NOT TARGET LZ4::lz4)
    add_library(LZ4::lz4 ALIAS PkgConfig::lz4)
endif()
find_package(CGAL REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C  ALIAS openmp_stub)
else()
    find_package(OpenMP REQUIRED)
endif()

include(ExternalProject)
include(ProcessorCount)
ProcessorCount(_nproc)
if(_nproc EQUAL 0)
    set(_nproc 1)
endif()
set(_libbt_install ${CMAKE_BINARY_DIR}/_deps/libbacktrace-install)
set(_libbt_lib ${_libbt_install}/lib/libbacktrace.a)
file(MAKE_DIRECTORY ${_libbt_install}/include)
ExternalProject_Add(libbacktrace_build
    GIT_REPOSITORY https://github.com/ianlancetaylor/libbacktrace.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    PREFIX ${CMAKE_BINARY_DIR}/_deps/libbacktrace
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${_libbt_install} --enable-static --disable-shared --with-pic
    BUILD_COMMAND make -j${_nproc}
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${_libbt_lib})
add_library(libbacktrace_static STATIC IMPORTED GLOBAL)
set_target_properties(libbacktrace_static PROPERTIES
    IMPORTED_LOCATION ${_libbt_lib}
    INTERFACE_INCLUDE_DIRECTORIES ${_libbt_install}/include)
add_dependencies(libbacktrace_static libbacktrace_build)
set(VC_LIBBACKTRACE libbacktrace_static)

# ---- Subdirectories ---------------------------------------------------------
include_directories(core/include)

add_subdirectory(libs/OpenABF)
add_subdirectory(libs/c3d)
add_subdirectory(libs/flatboi)

# Sanitizers/coverage on our code only — libs/ (PaStiX, libigl, ...) builds
# clean to keep sanitizer build times reasonable.
if(_san)
    add_compile_options(${_san})
    add_link_options(${_san})
endif()

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(apps)

if(VC_TESTING)
    enable_testing()
    add_subdirectory(core/test)
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        USES_TERMINAL)
endif()
