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

# ---- Options -----------------------------------------------------------------
option(VC_USE_OPENMP        "Enable OpenMP support"                              ON)
option(VC_USE_VALGRIND      "Enable Valgrind support"                            OFF)
option(VC_USE_PCH           "Enable precompiled headers (faster incremental builds, but disables ccache)" OFF)
option(VC_INSTALL_APPS      "Install VC core programs"                           ON)
option(VC_INSTALL_LIBS      "Install VC libraries"                               ON)
option(VC_INSTALL_UTILS     "Install VC utility programs"                        ON)
option(VC_WITH_PASTIX       "Build with PaStiX support"                          OFF)
option(VC_DEVELOPER_WARNINGS "For developers: Enable extensive compiler warnings" OFF)
option(VC_ENABLE_ASAN       "Enable AddressSanitizer for memory error detection" OFF)
option(VC_ENABLE_UBSAN      "Enable UndefinedBehaviorSanitizer"                  OFF)
option(VC_ENABLE_TSAN       "Enable ThreadSanitizer for data race detection"     OFF)
option(VC_ENABLE_LSAN       "Enable LeakSanitizer for memory leak detection (standalone)" OFF)
option(VC_BUILD_TESTS          "Build unit tests"                               OFF)
mark_as_advanced(VC_DEVELOPER_WARNINGS)

# ---- Modules -----------------------------------------------------------------
include(VCToolchain)
include(VCConstants)
include(VCFindDependencies)
include(VCCompilerFlags)
include(VCSanitizers)

include_directories(core/include)

add_subdirectory(misc)
add_subdirectory(libs/OpenABF)
add_subdirectory(core)
add_subdirectory(apps)

if(VC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(core/test)
endif()

include(VCPackageConfig)
include(VCInstall)
include(Packing)
