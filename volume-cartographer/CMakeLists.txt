cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(VC3D)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VC_USE_CCACHE      "Use ccache for compilation"                   ON)
option(VC_WARNINGS        "Enable curated warning set"                   OFF)
option(VC_WERROR          "Treat compiler warnings as errors"            OFF)
option(VC_ENABLE_ASAN     "AddressSanitizer"                             OFF)
option(VC_ENABLE_UBSAN    "UndefinedBehaviorSanitizer"                   OFF)
option(VC_ENABLE_TSAN     "ThreadSanitizer"                              OFF)
option(VC_ENABLE_LSAN     "LeakSanitizer (standalone)"                   OFF)
option(VC_ENABLE_COVERAGE "Code coverage instrumentation"                OFF)
option(VC_TESTING         "Build and run unit tests"                     OFF)

# ---- Toolchain --------------------------------------------------------------
foreach(_path "$ENV{HOME}/miniconda3" "$ENV{HOME}/anaconda3")
    if(EXISTS "${_path}")
        list(APPEND CMAKE_IGNORE_PREFIX_PATH "${_path}")
    endif()
endforeach()

if(VC_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER   "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

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

add_compile_options(-pipe -march=native -mcpu=native)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_dbg -g3 -fno-omit-frame-pointer -fasynchronous-unwind-tables -fstandalone-debug)
    set(_lto -flto=thin)
    set(_link -fuse-ld=lld -Wl,--gc-sections -Wl,--as-needed)
else()
    set(_dbg -g3 -fno-omit-frame-pointer -fasynchronous-unwind-tables -fno-eliminate-unused-debug-types)
    set(_lto -flto=auto)
    set(_link -Wl,--gc-sections -Wl,--as-needed)
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
if(VC_ENABLE_LSAN AND NOT VC_ENABLE_ASAN)
    list(APPEND _san -fsanitize=leak)
endif()
if(VC_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND _san -fprofile-instr-generate -fcoverage-mapping)
    else()
        list(APPEND _san --coverage)
    endif()
endif()
if(_san)
    add_compile_options(${_san} -Og)
    add_link_options(${_san})
endif()

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
find_package(Blosc2 REQUIRED)
find_package(zstd REQUIRED)
find_package(lz4 REQUIRED)
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
set(_libbt_install ${CMAKE_BINARY_DIR}/_deps/libbacktrace-install)
set(_libbt_lib ${_libbt_install}/lib/libbacktrace.a)
file(MAKE_DIRECTORY ${_libbt_install}/include)
ExternalProject_Add(libbacktrace_build
    GIT_REPOSITORY https://github.com/ianlancetaylor/libbacktrace.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    PREFIX ${CMAKE_BINARY_DIR}/_deps/libbacktrace
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${_libbt_install} --enable-static --disable-shared --with-pic
    BUILD_COMMAND make -j
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

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(apps)

if(VC_TESTING)
    include(CTest)
    add_subdirectory(core/test)
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        USES_TERMINAL)
endif()
