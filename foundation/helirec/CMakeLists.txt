cmake_minimum_required(VERSION 3.24)
project(helirec VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------- options
option(HELIREC_PYTHON "Build the nanobind Python module" ON)
set(HELIREC_CUDA "AUTO" CACHE STRING "Build CUDA kernels: ON, OFF or AUTO")
set_property(CACHE HELIREC_CUDA PROPERTY STRINGS ON OFF AUTO)

set(CMAKE_CXX_STANDARD 20)  # vc-delta3d's C++ API uses std::span; CUDA code stays C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CPU code: enable the host's FMA/SIMD units but never contract a*b+c implicitly:
# kernels that must match across backends spell out their FMAs (src/common/hd_math.hpp).
set(HELIREC_CPU_ARCH "native" CACHE STRING "value for -march (empty: compiler default)")

set(HELIREC_WITH_CUDA OFF)
if(HELIREC_CUDA STREQUAL "ON" OR HELIREC_CUDA STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Default: native SASS for the GPU(s) of the build machine. PTX JIT is not
    # relied on (it fails when the driver is older than the toolkit).
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
    set(HELIREC_WITH_CUDA ON)
  elseif(HELIREC_CUDA STREQUAL "ON")
    message(FATAL_ERROR "HELIREC_CUDA=ON but no CUDA compiler was found")
  endif()
endif()
message(STATUS "helirec: CUDA kernels ${HELIREC_WITH_CUDA}")

# ---------------------------------------------------------------- dependencies
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)   # double precision: filter construction
find_library(FFTW3F_OMP_LIB NAMES fftw3f_omp HINTS ${FFTW3F_LIBRARY_DIRS} REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# vc-delta3d (lossless uint8 chunk codec of vc_reader / VC3D), pinned to the revision vc_reader uses
# so that everything helirec writes is readable there. Offline builds: -DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D=<src>
option(HELIREC_DELTA3D "Build the vc-delta3d chunk codec (uint8 OME-Zarr output)" ON)
if(HELIREC_DELTA3D)
  include(FetchContent)
  set(VC_DELTA3D_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
  set(VC_DELTA3D_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(VC_DELTA3D_INSTALL_CPP OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(vc_delta3d
    GIT_REPOSITORY https://github.com/ScrollPrize/vc-delta3d.git
    GIT_TAG 45339743b450ff707d21fb742dc1b3b2bf99306d)  # = vc_reader cmake/NativeReader.cmake pin
  set(CMAKE_AUTOMOC OFF)
  FetchContent_MakeAvailable(vc_delta3d)
  set_target_properties(vc_delta3d PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# ---------------------------------------------------------------- core library
file(GLOB HELIREC_CPU_SOURCES CONFIGURE_DEPENDS src/core/*.cpp src/cpu/*.cpp src/io/*.cpp)
add_library(helirec STATIC ${HELIREC_CPU_SOURCES})
target_include_directories(helirec PUBLIC include PRIVATE src)
target_compile_options(helirec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -ffp-contract=off>)
if(HELIREC_CPU_ARCH)
  target_compile_options(helirec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=${HELIREC_CPU_ARCH}>)
endif()
target_link_libraries(helirec
  PUBLIC OpenMP::OpenMP_CXX
  PRIVATE PkgConfig::FFTW3F PkgConfig::FFTW3 ${FFTW3F_OMP_LIB} HDF5::HDF5)
if(HELIREC_DELTA3D)
  target_link_libraries(helirec PRIVATE vc_delta3d::vc_delta3d)
  target_compile_definitions(helirec PRIVATE HELIREC_WITH_DELTA3D=1)
endif()

if(HELIREC_WITH_CUDA)
  file(GLOB HELIREC_CUDA_SOURCES CONFIGURE_DEPENDS src/cuda/*.cu)
  target_sources(helirec PRIVATE ${HELIREC_CUDA_SOURCES})
  target_compile_definitions(helirec PUBLIC HELIREC_WITH_CUDA=1)
  target_compile_options(helirec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --extended-lambda>)
  set_target_properties(helirec PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(helirec PUBLIC CUDA::cudart PRIVATE CUDA::cufft)
endif()

# ---------------------------------------------------------------- python module
if(HELIREC_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  file(GLOB HELIREC_BINDINGS CONFIGURE_DEPENDS bindings/*.cpp)
  nanobind_add_module(_core NB_STATIC ${HELIREC_BINDINGS})
  target_link_libraries(_core PRIVATE helirec)
  install(TARGETS _core LIBRARY DESTINATION helirec)
endif()
