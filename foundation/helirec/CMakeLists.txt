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
find_package(Ceres 2.2 REQUIRED COMPONENTS EigenSparse)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)   # double precision: filter construction
find_library(FFTW3F_OMP_LIB NAMES fftw3f_omp HINTS ${FFTW3F_LIBRARY_DIRS} REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(ZLIB REQUIRED)   # zlib chunks of Zarr arrays (zarr.hpp)

find_package(CURL REQUIRED)   # Zarr arrays at http(s):// and s3:// locations (reference volumes)

# OpenJPEG: frames stored as JPEG 2000 (HTJ2K) chunks, as ESRF BM18 writes projections (src/io/j2k_chunk.hpp).
# Optional: without it such scans are refused with a message that names the package.
option(HELIREC_OPENJPEG "Read frames stored as JPEG 2000 (needs libopenjp2 >= 2.5)" ON)
if(HELIREC_OPENJPEG)
  pkg_check_modules(OPENJP2 IMPORTED_TARGET libopenjp2>=2.5)
endif()
message(STATUS "helirec: JPEG 2000 frames (OpenJPEG) ${OPENJP2_FOUND}")

# HTJ2K with 32-bit samples and seekable reduced-resolution decoding. The pinned
# source is built into helirec; no SDK or separately installed library is needed.
# Offline builds: -DFETCHCONTENT_SOURCE_DIR_OPENJPH=<source directory>.
option(HELIREC_OPENJPH "Read float32 HTJ2K volumes with OpenJPH" ON)
if(HELIREC_OPENJPH)
  include(FetchContent)
  function(helirec_add_openjph)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(BUILD_SHARED_LIBS OFF)
    set(OJPH_BUILD_EXECUTABLES OFF)
    set(OJPH_BUILD_STREAM_EXPAND OFF)
    set(OJPH_ENABLE_TIFF_SUPPORT OFF)
    set(OJPH_BUILD_TESTS OFF)
    set(OJPH_BUILD_FUZZER OFF)
    FetchContent_Declare(openjph
      URL https://codeload.github.com/aous72/OpenJPH/tar.gz/refs/tags/0.32.0
      URL_HASH SHA256=5cb1ebe18e5ee1322d23ff2130b37562d447d1ad5586c3d16cf84be3ae30f719
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SOURCE_SUBDIR helirec-unused)
    # Add explicitly with EXCLUDE_FROM_ALL (also on CMake 3.24) to avoid
    # installing OpenJPH's headers, archive and package files into our wheel.
    FetchContent_MakeAvailable(openjph)
    add_subdirectory("${openjph_SOURCE_DIR}" "${openjph_BINARY_DIR}" EXCLUDE_FROM_ALL)
    if(HELIREC_PYTHON)
      install(FILES "${openjph_SOURCE_DIR}/LICENSE" DESTINATION helirec/licenses RENAME OpenJPH.txt)
    else()
      install(FILES "${openjph_SOURCE_DIR}/LICENSE" DESTINATION share/helirec/licenses RENAME OpenJPH.txt)
    endif()
  endfunction()
  helirec_add_openjph()
endif()

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
file(GLOB HELIREC_CPU_SOURCES CONFIGURE_DEPENDS src/core/*.cpp src/cpu/*.cpp src/io/*.cpp src/calibration/*.cpp)
add_library(helirec STATIC ${HELIREC_CPU_SOURCES})
target_include_directories(helirec PUBLIC include PRIVATE src)
target_compile_options(helirec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -ffp-contract=off>)
if(HELIREC_CPU_ARCH)
  target_compile_options(helirec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=${HELIREC_CPU_ARCH}>)
endif()
target_link_libraries(helirec
  PUBLIC OpenMP::OpenMP_CXX Ceres::ceres
  PRIVATE PkgConfig::FFTW3F PkgConfig::FFTW3 ${FFTW3F_OMP_LIB} HDF5::HDF5 ZLIB::ZLIB CURL::libcurl)
if(HELIREC_DELTA3D)
  target_link_libraries(helirec PRIVATE vc_delta3d::vc_delta3d)
  target_compile_definitions(helirec PRIVATE HELIREC_WITH_DELTA3D=1)
endif()
if(HELIREC_OPENJPEG AND OPENJP2_FOUND)
  target_link_libraries(helirec PRIVATE PkgConfig::OPENJP2)
  target_compile_definitions(helirec PRIVATE HELIREC_WITH_OPENJPEG=1)
endif()
if(HELIREC_OPENJPH AND TARGET openjph)
  target_link_libraries(helirec PRIVATE openjph)
  target_compile_definitions(helirec PRIVATE HELIREC_WITH_OPENJPH=1)
endif()

if(HELIREC_WITH_CUDA)
  file(GLOB HELIREC_CUDA_SOURCES CONFIGURE_DEPENDS src/cuda/*.cu)
  target_sources(helirec PRIVATE ${HELIREC_CUDA_SOURCES})
  target_compile_definitions(helirec PUBLIC HELIREC_WITH_CUDA=1)
  target_compile_options(helirec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --extended-lambda>)
  set_target_properties(helirec PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_link_libraries(helirec PUBLIC CUDA::cudart PRIVATE CUDA::cufft)
endif()

# ---------------------------------------------------------------- tools
# helirec-calibration: checks an acquisition record against itself and against the raw frames (src/calibration)
option(HELIREC_TOOLS "Build the command-line tools" ON)
if(HELIREC_TOOLS)
  add_executable(helirec-calibration tools/helirec_calibration.cpp)
  target_link_libraries(helirec-calibration PRIVATE helirec)
  target_compile_options(helirec-calibration PRIVATE -O2 -Wall -Wextra)
  install(TARGETS helirec-calibration RUNTIME DESTINATION bin)
  include(CTest)
  if(BUILD_TESTING)
    add_test(NAME calibration_selftest COMMAND helirec-calibration selftest)
    add_executable(helirec-fitting-test tests/cpp/test_fitting.cpp)
    target_include_directories(helirec-fitting-test PRIVATE src)
    target_link_libraries(helirec-fitting-test PRIVATE helirec)
    add_test(NAME fitting_derivatives COMMAND helirec-fitting-test)
  endif()
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
