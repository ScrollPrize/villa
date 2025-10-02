# Detects whether this is a top-level project
get_directory_property(LIBIGL_PARENT_DIR PARENT_DIRECTORY)
if(NOT LIBIGL_PARENT_DIR)
    set(LIBIGL_TOPLEVEL_PROJECT ON)
else()
    set(LIBIGL_TOPLEVEL_PROJECT OFF)
endif()



# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.16.0")
if(LIBIGL_TOPLEVEL_PROJECT)
    cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
else()
    # Don't use cmake_minimum_required here to avoid implicitly overriding parent policies
    if(${CMAKE_VERSION} VERSION_LESS ${REQUIRED_CMAKE_VERSION})
        message(FATAL_ERROR "CMake required version to build Libigl is ${REQUIRED_CMAKE_VERSION}")
    endif()
endif()

# Include user-provided default options if available. We do that before the main
# `project()` so that we can define the C/C++ compilers from the option file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/LibiglOptions.cmake)
    message(STATUS "Using local options file: ${CMAKE_CURRENT_SOURCE_DIR}/LibiglOptions.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/LibiglOptions.cmake)
endif()

# If this option is enabled, this will setup the Hunter package manager.
option(HUNTER_ENABLED "Enable Hunter package manager support" OFF)
if(HUNTER_ENABLED)
    include("cmake/misc/HunterGate.cmake")
    HunterGate(
        URL "https://github.com/cpp-pm/hunter/archive/v0.24.8.tar.gz"
        SHA1 "ca7838dded9a1811b04ffd56175f629e0af82d3d"
    )
endif()

################################################################################
project(libigl VERSION 2.5.0)

# CMake module path
list(PREPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_LIST_DIR}/cmake/
    ${CMAKE_CURRENT_LIST_DIR}/cmake/igl
    ${CMAKE_CURRENT_LIST_DIR}/cmake/find
    ${CMAKE_CURRENT_LIST_DIR}/cmake/recipes/external
)

if(HUNTER_ENABLED)
    list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/recipes/hunter)
else()
    list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/recipes/external)
endif()

set_property(GLOBAL PROPERTY __igl_module_path ${CMAKE_MODULE_PATH})

set(LIBIGL_DEFAULT_CGAL ${LIBIGL_TOPLEVEL_PROJECT})
set(MATLAB_ADDITIONAL_VERSIONS
    "R2023b=10.4"
    "R2023a=10.4"
    "R2022b=10.3"
    "R2022a=10.2"
    "R2021b=10.1"
    "R2021a=10.0"
)
set(LIBIGL_DEFAULT_MATLAB ${LIBIGL_TOPLEVEL_PROJECT})
set(LIBIGL_DEFAULT_MOSEK ${LIBIGL_TOPLEVEL_PROJECT})
if(LIBIGL_TOPLEVEL_PROJECT)
    find_package(Matlab QUIET)
    if(NOT Matlab_FOUND)
        set(LIBIGL_DEFAULT_MATLAB OFF)
        message(WARNING "Matlab not found, disabling igl_restricted::matlab module.")
    endif()
    find_package(MOSEK QUIET)
    if(NOT MOSEK_FOUND)
        set(LIBIGL_DEFAULT_MOSEK OFF)
        message(WARNING "Mosek not found, disabling igl_restricted::mosek module.")
    endif()
endif()
set(LIBIGL_DEFAULT_COMISO ${LIBIGL_TOPLEVEL_PROJECT})
if(LIBIGL_TOPLEVEL_PROJECT AND (NOT APPLE) AND UNIX)
  find_package(BLAS QUIET)
    if(NOT BLAS_FOUND)
        set(LIBIGL_DEFAULT_COMISO OFF)
        message(WARNING "BLAS not found, disabling igl_copyleft::comiso module.")
    endif()
endif()

# ====================== CHOLMOD (SuiteSparse) support =======================
# Turn on CHOLMOD (multithreaded via OpenMP + threaded BLAS) for SLIM, etc.
option(LIBIGL_WITH_CHOLMOD "Enable CHOLMOD (SuiteSparse) support" ON)
if(LIBIGL_WITH_CHOLMOD)
  # Ensure all compilation units see CHOLMOD (slim.cpp uses #ifdef CHOLMOD)
  add_compile_definitions(CHOLMOD)
  # Try pkg-config first (some distros ship cholmod.pc), then fall back.
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(CHOLMOD_PC QUIET cholmod)
  endif()

  add_library(igl_cholmod INTERFACE)
  add_library(igl::cholmod ALIAS igl_cholmod)
  # compile def also added globally above; keep it here for consumers outside this tree
  target_compile_definitions(igl_cholmod INTERFACE CHOLMOD)

  if(CHOLMOD_PC_FOUND)
    target_include_directories(igl_cholmod INTERFACE ${CHOLMOD_PC_INCLUDE_DIRS})
    target_link_libraries(igl_cholmod INTERFACE ${CHOLMOD_PC_LIBRARIES})
    message(STATUS "CHOLMOD enabled (via pkg-config)")
    # expose include dirs to all later targets in this directory (e.g., tutorials)
    set(_CHOLMOD_INCS ${CHOLMOD_PC_INCLUDE_DIRS})
  else()
    # Fallback for Ubuntu/Debian: find headers and libs manually
    find_path(CHOLMOD_INCLUDE_DIR NAMES cholmod.h PATH_SUFFIXES suitesparse)
    find_library(CHOLMOD_LIB            NAMES cholmod)
    find_library(SUITESPARSECONFIG_LIB  NAMES suitesparseconfig)
    find_library(AMD_LIB                NAMES amd)
    find_library(COLAMD_LIB             NAMES colamd)
    find_library(CAMD_LIB               NAMES camd)
    find_library(CCOLAMD_LIB            NAMES ccolamd)
    set(CHOLMOD_FALLBACK_LIBS
      ${CHOLMON_LIB} ${CHOLMOD_LIB}      # keep ${CHOLMON_LIB} typo-safe
      ${SUITESPARSECONFIG_LIB}
      ${AMD_LIB} ${COLAMD_LIB} ${CAMD_LIB} ${CCOLAMD_LIB})

    if(NOT CHOLMOD_INCLUDE_DIR OR NOT CHOLMOD_LIB)
      message(FATAL_ERROR
        "CHOLMOD not found. Install 'libsuitesparse-dev' or provide paths via CMAKE_PREFIX_PATH.")
    endif()
    target_include_directories(igl_cholmod INTERFACE ${CHOLMOD_INCLUDE_DIR})
    target_link_libraries(igl_cholmod INTERFACE ${CHOLMOD_FALLBACK_LIBS})

    # Also link BLAS/LAPACK if available
    find_package(BLAS)
    find_package(LAPACK)
    if (TARGET LAPACK::LAPACK)
      target_link_libraries(igl_cholmod INTERFACE LAPACK::LAPACK)
    endif()
    if (TARGET BLAS::BLAS)
      target_link_libraries(igl_cholmod INTERFACE BLAS::BLAS)
    endif()
    message(STATUS "CHOLMOD enabled (manual find):")
    message(STATUS "  include: ${CHOLMOD_INCLUDE_DIR}")
    message(STATUS "  libs   : ${CHOLMOD_FALLBACK_LIBS}")
  endif()
  
  # Make sure every target created afterwards inherits CHOLMOD includes & libs
  include_directories(SYSTEM ${_CHOLMOD_INCS})
  #link_libraries(igl::cholmod)

  # Try to enable OpenMP for CHOLMOD parallelism
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    target_link_libraries(igl_cholmod INTERFACE OpenMP::OpenMP_CXX)
  endif()
  # If we install, export the interface target too (harmless no-op otherwise)
  if(LIBIGL_INSTALL)
    install(TARGETS igl_cholmod EXPORT LibiglTargets)
  endif()
endif()
# ===========================================================================

# ============================ PaStiX (v5.x) ================================
option(LIBIGL_WITH_PASTIX "Enable PaStiX 5.x support (NOMPI+OpenMP)" OFF)
if(LIBIGL_WITH_PASTIX)
  add_compile_definitions(PASTIX)
  add_library(igl_pastix INTERFACE)
  add_library(igl::pastix ALIAS igl_pastix)
  target_compile_definitions(igl_pastix INTERFACE PASTIX)

  # If you installed PaStiX into a custom prefix, point CMake at it:
  # cmake .. -DCMAKE_PREFIX_PATH=$HOME/pastix
  # Headers live in include/pastix/, library name may vary:
  find_path(PASTIX_INCLUDE_DIR NAMES pastix_nompi.h pastix.h
            PATH_SUFFIXES pastix)
  # prefer NOMPI+double if present, else try generic
  find_library(PASTIX_LIB NAMES pastix_nompi_double pastix)

  # Common PaStiX 5 deps (adjust if you built without them)
  find_library(PASTIX_SCOTCH_LIB NAMES scotch)
  find_library(PASTIX_SCOTCHERR_LIB NAMES scotcherr scotcherrexit)
  #find_library(PASTIX_HWLOC_LIB NAMES hwloc)

  # BLAS/LAPACK (OpenBLAS recommended)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)

  if(NOT PASTIX_INCLUDE_DIR OR NOT PASTIX_LIB)
    message(FATAL_ERROR "PaStiX not found. Set CMAKE_PREFIX_PATH to your PaStiX 5.x install prefix.")
  endif()

  target_include_directories(igl_pastix INTERFACE ${PASTIX_INCLUDE_DIR})
  target_link_libraries(igl_pastix INTERFACE
    ${PASTIX_LIB}
    ${PASTIX_SCOTCH_LIB}
    ${PASTIX_SCOTCHERR_LIB}
    #${PASTIX_HWLOC_LIB}
    LAPACK::LAPACK BLAS::BLAS)

  # Ensure all compilation units in this directory see <pastix_nompi.h>
  include_directories(SYSTEM ${PASTIX_INCLUDE_DIR})
  
  # Make all later targets pull PaStiX in
  #link_libraries(igl::pastix)

  # Optional: OpenMP (PaStiX is OpenMP-parallel)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(igl_pastix INTERFACE OpenMP::OpenMP_CXX)
  endif()

  message(STATUS "PaStiX enabled:")
  message(STATUS "  include: ${PASTIX_INCLUDE_DIR}")
  message(STATUS "  lib    : ${PASTIX_LIB}")

  # If we install, export the interface target so exports are consistent
  if(LIBIGL_INSTALL)
    install(TARGETS igl_pastix EXPORT LibiglTargets)
  endif()
endif()
# ===========================================================================

# Build tests and tutorials
option(LIBIGL_BUILD_TESTS      "Build libigl unit test"                ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_BUILD_TUTORIALS  "Build libigl tutorial"                 ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_INSTALL          "Enable installation of libigl targets" ${LIBIGL_TOPLEVEL_PROJECT})

# USE_STATIC_LIBRARY speeds up the generation of multiple binaries,
# at the cost of a longer initial compilation time
# (by default, static build is off since libigl is a header-only library)
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" ${LIBIGL_TOPLEVEL_PROJECT})

# Permissive modules. These modules are available under MPL2 license, and their dependencies are available
# under a permissive or public domain license.
option(LIBIGL_EMBREE           "Build target igl::embree"           ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_GLFW             "Build target igl::glfw"             ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_IMGUI            "Build target igl::imgui"            ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_OPENGL           "Build target igl::opengl"           ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_STB              "Build target igl::stb"              ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_PREDICATES       "Build target igl::predicates"       ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_SPECTRA          "Build target igl::spectra"          ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_XML              "Build target igl::xml"              ${LIBIGL_TOPLEVEL_PROJECT})

# Copyleft modules. These modules are available under GPL license, and their dependencies are
# available under a copyleft license.
option(LIBIGL_COPYLEFT_CORE    "Build target igl_copyleft::core"    ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_COPYLEFT_CGAL    "Build target igl_copyleft::cgal"    ${LIBIGL_DEFAULT_CGAL})
option(LIBIGL_COPYLEFT_COMISO  "Build target igl_copyleft::comiso"  ${LIBIGL_TOPLEVEL_PROJECT})
option(LIBIGL_COPYLEFT_TETGEN  "Build target igl_copyleft::tetgen"  ${LIBIGL_TOPLEVEL_PROJECT})

# Restricted modules. These modules are available under MPL2 license, but their dependencies are
# only available under a non-commercial or proprietary license.
option(LIBIGL_RESTRICTED_MATLAB   "Build target igl_restricted::matlab"   ${LIBIGL_DEFAULT_MATLAB})
option(LIBIGL_RESTRICTED_MOSEK    "Build target igl_restricted::mosek"    ${LIBIGL_DEFAULT_MOSEK})
option(LIBIGL_RESTRICTED_TRIANGLE "Build target igl_restricted::triangle" ${LIBIGL_TOPLEVEL_PROJECT})

# GLFW doesn't run on headless CI machines so don't run (or build them).
# Unfortunately on headless mac machines glfw seems to hang rather than crash
# making it hard to catch at runtime.
option(LIBIGL_GLFW_TESTS       "Build igl::glfw tests"             ${LIBIGL_TOPLEVEL_PROJECT})

option(LIBIGL_WARNINGS_AS_ERRORS "Turn on many warnings and treat as errors" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


# Include CMake helper functions
include(igl_add_library)
include(igl_add_test)
include(igl_add_tutorial)
include(igl_copy_dll)
include(igl_include)
include(igl_install)
include(igl_target_sources)

# Enable unit testing at the root level
if(LIBIGL_BUILD_TESTS)
    include(CTest)
endif()

# Defines CMake targets for selected libigl modules
include(libigl)

# Attach solver interfaces to libigl’s core (use the real target, not the ALIAS)
if (TARGET igl_core)
  if(LIBIGL_WITH_CHOLMOD)
    target_link_libraries(igl_core INTERFACE igl::cholmod)
  endif()
  if(LIBIGL_WITH_PASTIX)
    target_link_libraries(igl_core INTERFACE igl::pastix)
  endif()
endif()

if(LIBIGL_BUILD_TUTORIALS)
    add_subdirectory(tutorial)
endif()

################################################################################
# Install CMake config files
################################################################################

if(LIBIGL_INSTALL)
    include(GNUInstallDirs)
    set(project_config_in "${PROJECT_SOURCE_DIR}/cmake/igl/libigl-config.cmake.in")
    set(project_config_out "${CMAKE_CURRENT_BINARY_DIR}/libigl-config.cmake")
    set(config_targets_base "LibiglConfigTargets")
    set(version_config_file "${CMAKE_CURRENT_BINARY_DIR}/LibiglConfigVersion.cmake")
    set(export_dest_dir "${CMAKE_INSTALL_LIBDIR}/cmake/igl")

    foreach(suffix IN ITEMS "") #"_restricted" "_copyleft")
        install(EXPORT LibiglTargets${suffix}
            DESTINATION ${export_dest_dir}
            NAMESPACE igl${suffix}::
            FILE ${config_targets_base}${suffix}.cmake
            COMPONENT LibiglDevelopment
        )
    endforeach()

    include(CMakePackageConfigHelpers)
    configure_package_config_file(
            "${project_config_in}"
            "${project_config_out}"
        INSTALL_DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/libigl/cmake
    )

    write_basic_package_version_file("${version_config_file}" COMPATIBILITY SameMajorVersion)
    install(FILES "${project_config_out}" "${version_config_file}" DESTINATION "${export_dest_dir}")
endif()

