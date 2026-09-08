cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

#### Initialize cache variables ####
get_property(multiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT multiConfig AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

#### Define project ####
project(pgs-deps)

#### CMake Module Path ####
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

### Include extra CMake Modules ###
include(ExternalProject)
include(CMakeDependentOption)

#### Setup output directory if not specified ####
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/installed" CACHE PATH "Default Install Path" FORCE )
endif()

### Global arguments ###
set(GLOBAL_CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}"
    "-DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}"
    "-DBUILD_SHARED_LIBS:BOOL=FALSE"
)

# AppleClang 11.0.0-11.0.2 Bug: https://developer.apple.com/forums/thread/121887
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang")
if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 11 AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 11.0.3)
    message(STATUS "Disabling stack check for AppleClang ${CMAKE_CXX_COMPILER_VERSION}")
    add_compile_options(-fno-stack-check)
    list(APPEND GLOBAL_CMAKE_ARGS "-DCMAKE_CXX_FLAGS:STRING=-fno-stack-check")
endif()
endif()
endif()

# Check for SSE support for OpenMVS on ARM
message(STATUS "Detecting SSE support")
cmake_host_system_information(RESULT HAS_SSE QUERY HAS_SSE)
if(HAS_SSE)
    message(STATUS "Detecting SSE support - done")
else()
    message(STATUS "Detecting SSE support - failed")
endif()

### Eigen ###
include(BuildEigen)

### jpeg ###
include(BuildJPEG)

### OpenCV ###
include(BuildOpenCV)

### CGAL ###
include(BuildCGAL)

### VCG ###
include(BuildVCG)

### openMVG ###
include(BuildOpenMVG)

### openMVS ###
include(BuildOpenMVS)

### pgs-recon-utilities
include(BuildPGSUtils)