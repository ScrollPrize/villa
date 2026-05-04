cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

set(VC_VERSION 3.0.2)

list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/vc-dependencies")

option(VC_VERSION_DATESTAMP "Append date stamp to version number" off)
if(VC_VERSION_DATESTAMP)
string(TIMESTAMP VC_VERSION_TWEAK "%s")
set(VC_VERSION ${VC_VERSION}.${VC_VERSION_TWEAK})
endif()

project(VC3D VERSION ${VC_VERSION})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# ---- Options (centralized) ---------------------------------------------------
option(VC_USE_CCACHE           "Use ccache for compilation"                                  ON)
option(VC_USE_OPENMP           "Enable OpenMP support"                                       OFF)
option(VC_USE_VALGRIND         "Enable Valgrind support"                                     OFF)
option(VC_USE_PCH              "Enable precompiled headers (faster builds)"                  OFF)
option(VC_NATIVE_ARCH          "Build with -march=native (non-portable artifacts)"           ON)
option(VC_STRIP_BINARIES       "Strip release binaries (disables file:line in crash dumps)"  OFF)
option(VC_VECTORIZATION_REPORT "Emit clang vectorization remarks during compile"             OFF)
option(VC_WITH_CUDA_SPARSE     "Use cuDSS for sparse linear solves"                          ON)
option(VC_WITH_PASTIX          "Build with PaStiX support"                                   OFF)
option(VC_WARNINGS             "Enable curated warning set for the VC build"                 OFF)
option(VC_WERROR               "Treat compiler warnings as errors"                           OFF)
option(VC_ENABLE_ASAN          "Enable AddressSanitizer for memory error detection"          OFF)
option(VC_ENABLE_UBSAN         "Enable UndefinedBehaviorSanitizer"                            OFF)
option(VC_ENABLE_TSAN          "Enable ThreadSanitizer for data race detection"              OFF)
option(VC_ENABLE_LSAN          "Enable LeakSanitizer for memory leak detection (standalone)" OFF)
option(VC_ENABLE_COVERAGE      "Enable code coverage instrumentation (use with Debug build)" OFF)
option(VC_PGO_GENERATE         "Build instrumented binaries that emit PGO profile data"      OFF)
option(VC_PGO_USE              "Build using previously-collected PGO profile data"           OFF)
option(VC_BUILD_TESTS          "Build unit tests"                                            OFF)
set(VC_PGO_PROFILE_DIR "${CMAKE_BINARY_DIR}/pgo-data" CACHE PATH
    "Directory holding PGO profile data (.gcda for gcc, .profraw for clang)")
set(CHUNK_CACHE_SIZE_GB "10" CACHE STRING "Size of the chunk cache in gigabytes")

# ---- Modules -----------------------------------------------------------------
include(VCToolchain)
include(VCConstants)
include(VCFindDependencies)
include(VCCompilerFlags)
include(VCWarnings)
include(VCSanitizers)
include(VCPgo)
include(CTest)

include_directories(core/include)

add_subdirectory(misc)
add_subdirectory(libs/OpenABF)
add_subdirectory(libs/c3d)

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(apps)

if(VC_BUILD_TESTS AND BUILD_TESTING)
    add_subdirectory(core/test)
endif()

