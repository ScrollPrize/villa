# --- VC dependencies ----------------------------------------------------------
include(FetchContent)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(VC_BUILD_JSON "Build in-source JSON library" OFF)
option(VC_BUILD_Z5   "Build (vendor) z5 header-only library" ON)
option(VC_BUILD_BLOSC "Build (vendor) Blosc2 library" ON)
option(VC_BUILD_XSIMD "Build (vendor) xsimd" ON)
option(VC_BUILD_XTENSOR "Build (vendor) xtensor" ON)
option(VC_BUILD_XTL "Build (vendor) xtl" ON)
option(VC_BUILD_OPENCV "Build (vendor) OpenCV" ON)
option(VC_BUILD_JSON "Build (vendor) nlohmann/json" ON)

#find_package(CURL REQUIRED)
#find_package(OpenSSL REQUIRED)
#find_package(ZLIB REQUIRED)
#find_package(glog REQUIRED)

# --- Vendored Blosc2 ---------------------------------------------------------
if (VC_BUILD_BLOSC)
    set(BUILD_SHARED     OFF CACHE BOOL "Build shared Blosc2 library"      FORCE)
    set(BUILD_STATIC     ON  CACHE BOOL "Build static Blosc2 library"       FORCE)
    set(BUILD_TESTS      OFF CACHE BOOL "Disable Blosc2 tests"              FORCE)
    set(BUILD_FUZZERS    OFF CACHE BOOL "Disable Blosc2 fuzzers"            FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable Blosc2 benchmarks"         FORCE)
    set(BUILD_EXAMPLES   OFF CACHE BOOL "Disable Blosc2 examples"           FORCE)
    set(BUILD_PLUGINS    ON  CACHE BOOL "Enable Blosc2 plugins"             FORCE)
    # Enable install to create the blosc2 interface target (needed for z5 export)
    set(BLOSC_INSTALL    ON  CACHE BOOL "Enable Blosc2 install targets"     FORCE)

    # Find OpenH264 for inline H264 codec support
    find_path(OPENH264_INCLUDE_DIR wels/codec_api.h)
    find_library(OPENH264_LIBRARY NAMES openh264)
    if(OPENH264_INCLUDE_DIR AND OPENH264_LIBRARY)
        message(STATUS "Found OpenH264: ${OPENH264_LIBRARY}")
        message(STATUS "OpenH264 include: ${OPENH264_INCLUDE_DIR}")
        set(VC_HAVE_OPENH264 TRUE)
    else()
        message(STATUS "OpenH264 not found - H264 codec will not be available")
        set(VC_HAVE_OPENH264 FALSE)
    endif()

    add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/c-blosc2 EXCLUDE_FROM_ALL)

    # Add OpenH264 support to blosc2 after subdirectory is processed
    if(VC_HAVE_OPENH264 AND TARGET blosc2_static)
        target_compile_definitions(blosc2_static PRIVATE WITH_OPENH264)
        target_include_directories(blosc2_static PRIVATE
            ${OPENH264_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/thirdparty/c-blosc2/plugins/codecs
        )
        target_link_libraries(blosc2_static PRIVATE ${OPENH264_LIBRARY})
    endif()
endif()

# ---- Qt (apps / utils) -------------------------------------------------------
find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Guard old qt cmake helper on distros with Qt < 6.3
if(NOT DEFINED qt_generate_deploy_app_script)
    message(WARNING "WARNING qt_generate_deploy_app_script MISSING!")
    function(qt_generate_deploy_app_script)
    endfunction()
endif()

# ---- CUDA sparse toggle ------------------------------------------------------
option(VC_WITH_CUDA_SPARSE "use cudss" ON)
if (VC_WITH_CUDA_SPARSE)
    add_definitions(-DVC_USE_CUDA_SPARSE=1)
endif()

# ---- Ceres -------------------------------------------------------------------
find_package(Ceres REQUIRED)

# ---- Eigen -------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED)
if (CMAKE_GENERATOR MATCHES "Ninja|.*Makefiles.*" AND "${CMAKE_BUILD_TYPE}" MATCHES "^$|Debug")
    message(AUTHOR_WARNING
        "Configuring a Debug build. Eigen performance will be degraded. "
        "Consider RelWithDebInfo for symbols, or Release for max performance.")
endif()

# ---- OpenCV ------------------------------------------------------------------
if (VC_BUILD_OPENCV)
    # Temporarily disable Qt AUTOMOC to prevent it from scanning OpenCV sources
    # (OpenCV has generated headers that don't exist at configure time)
    set(_VC_SAVE_AUTOMOC ${CMAKE_AUTOMOC})
    set(_VC_SAVE_AUTORCC ${CMAKE_AUTORCC})
    set(_VC_SAVE_AUTOUIC ${CMAKE_AUTOUIC})
    set(CMAKE_AUTOMOC OFF)
    set(CMAKE_AUTORCC OFF)
    set(CMAKE_AUTOUIC OFF)

    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static OpenCV" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Disable OpenCV tests" FORCE)
    set(BUILD_PERF_TESTS OFF CACHE BOOL "Disable OpenCV perf tests" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable OpenCV examples" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "Disable OpenCV docs" FORCE)
    set(BUILD_opencv_apps OFF CACHE BOOL "Disable OpenCV apps" FORCE)
    set(BUILD_opencv_java OFF CACHE BOOL "Disable OpenCV Java" FORCE)
    set(BUILD_opencv_python OFF CACHE BOOL "Disable OpenCV Python" FORCE)
    set(BUILD_opencv_python2 OFF CACHE BOOL "Disable OpenCV Python2" FORCE)
    set(BUILD_opencv_python3 OFF CACHE BOOL "Disable OpenCV Python3" FORCE)
    set(BUILD_LIST "core;imgproc;imgcodecs;videoio;calib3d;video;photo;highgui;ximgproc" CACHE STRING "OpenCV modules to build" FORCE)
    set(OPENCV_EXTRA_MODULES_PATH "${CMAKE_SOURCE_DIR}/thirdparty/opencv_contrib/modules" CACHE PATH "OpenCV contrib modules" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/opencv EXCLUDE_FROM_ALL)

    # Restore Qt auto settings
    set(CMAKE_AUTOMOC ${_VC_SAVE_AUTOMOC})
    set(CMAKE_AUTORCC ${_VC_SAVE_AUTORCC})
    set(CMAKE_AUTOUIC ${_VC_SAVE_AUTOUIC})

    # Set up OpenCV include directories for downstream targets
    # (needed when building OpenCV as a subproject)
    # Note: OpenCV generates headers (opencv_modules.hpp, cvconfig.h, etc.) in CMAKE_BINARY_DIR
    set(OpenCV_INCLUDE_DIRS
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/core/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/imgproc/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/imgcodecs/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/videoio/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/calib3d/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/video/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/photo/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/highgui/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/features2d/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv/modules/flann/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/opencv_contrib/modules/ximgproc/include"
        "${CMAKE_BINARY_DIR}"
        CACHE PATH "OpenCV include directories" FORCE)
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    find_package(OpenCV 3 QUIET)
    if(NOT OpenCV_FOUND)
        find_package(OpenCV 4 QUIET REQUIRED)
    endif()
endif()

# ---- OpenMP ------------------------------------------------------------------
if (VC_USE_OPENMP)
    message(STATUS "OpenMP support enabled")
    find_package(OpenMP REQUIRED)
    set(XTENSOR_USE_OPENMP 1)
else()
    message(STATUS "OpenMP support disabled")
    set(XTENSOR_USE_OPENMP 0)
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C  ALIAS openmp_stub)
    # Add openmp_stub to the export set so install(EXPORT) works
    install(TARGETS openmp_stub EXPORT "${targets_export_name}")
endif()

# ---- xtensor/xsimd toggle used by your code ---------------------------------
set(XTENSOR_USE_XSIMD 1)
if (VC_BUILD_XTL)
    set(BUILD_TESTS OFF CACHE BOOL "Disable xtl tests" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtl EXCLUDE_FROM_ALL)
    set(xtl_DIR "${CMAKE_BINARY_DIR}/thirdparty/xtl" CACHE PATH "Vendored xtl config dir" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/thirdparty/xtl")
else()
    find_package(xtl REQUIRED)
endif()

if (VC_BUILD_XSIMD)
    set(BUILD_TESTS OFF CACHE BOOL "Disable xsimd tests" FORCE)
    set(BUILD_BENCHMARK OFF CACHE BOOL "Disable xsimd benchmarks" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "Disable xsimd examples" FORCE)
    set(ENABLE_XTL_COMPLEX ON CACHE BOOL "Enable xtl complex support in xsimd" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xsimd EXCLUDE_FROM_ALL)
    set(xsimd_DIR "${CMAKE_BINARY_DIR}/thirdparty/xsimd" CACHE PATH "Vendored xsimd config dir" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/thirdparty/xsimd")
else()
    find_package(xsimd REQUIRED)
endif()

if (VC_BUILD_XTENSOR)
    set(xsimd_REQUIRED_VERSION 14.0.0 CACHE STRING "xsimd version required by xtensor" FORCE)
    set(XTENSOR_USE_XSIMD ON CACHE BOOL "Enable xsimd for xtensor" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "Disable xtensor tests" FORCE)
    set(BUILD_BENCHMARK OFF CACHE BOOL "Disable xtensor benchmarks" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtensor EXCLUDE_FROM_ALL)
    if (TARGET xtensor)
        target_include_directories(xtensor INTERFACE
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/thirdparty/xtensor-compat/include>
        )
    endif()
    set(xtensor_DIR "${CMAKE_BINARY_DIR}/thirdparty/xtensor" CACHE PATH "Vendored xtensor config dir" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/thirdparty/xtensor")
else()
    find_package(xtensor REQUIRED)
    if (xtensor_INCLUDE_DIRS)
        include_directories(SYSTEM ${xtensor_INCLUDE_DIRS})
    endif()
endif()

# --- Vendored z5 -------------------------------------------------------------
if (VC_BUILD_Z5)
    # z5 defines options; set them in the cache *before* adding the subproject.
    set(BUILD_Z5PY OFF CACHE BOOL "Disable Python bits for z5" FORCE)
    set(WITH_BLOSC ON  CACHE BOOL "Enable Blosc in z5"        FORCE)

    # On CMake ≥4, compatibility with <3.5 was removed. Setting this floor
    # avoids errors if z5 asks for 3.1 in its CMakeLists.
    if (NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    endif()

    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/z5 EXCLUDE_FROM_ALL)
else()
    find_package(z5 CONFIG REQUIRED)
endif()
if (TARGET z5)
    if (TARGET xtensor)
        target_link_libraries(z5 INTERFACE xtensor)
        get_target_property(_vc_xtensor_includes xtensor INTERFACE_INCLUDE_DIRECTORIES)
        if (_vc_xtensor_includes)
            target_include_directories(z5 INTERFACE ${_vc_xtensor_includes})
        endif()
    elseif (TARGET xtensor::xtensor)
        target_link_libraries(z5 INTERFACE xtensor::xtensor)
        get_target_property(_vc_xtensor_includes xtensor::xtensor INTERFACE_INCLUDE_DIRECTORIES)
        if (_vc_xtensor_includes)
            target_include_directories(z5 INTERFACE ${_vc_xtensor_includes})
        endif()
    endif()
    target_include_directories(z5 INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/thirdparty/xtensor-compat/include>
    )
endif()

# ---- nlohmann/json -----------------------------------------------------------
if (VC_BUILD_JSON)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install   OFF CACHE INTERNAL "")
    add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/json EXCLUDE_FROM_ALL)
    set(nlohmann_json_DIR "${CMAKE_BINARY_DIR}/thirdparty/json" CACHE PATH "Vendored nlohmann_json config dir" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}/thirdparty/json")
else()
    find_package(nlohmann_json 3.9.1 REQUIRED)
endif()

# ---- Boost (apps/utils only) -------------------------------------------------
find_package(Boost 1.58 REQUIRED COMPONENTS program_options)

# ---- PaStiX ------------------------------------------------------------------
if (VC_WITH_PASTIX)
  find_package(PaStiX REQUIRED)
  message(STATUS "PaStiX found: ${PASTIX_LIBRARY}")
  if (NOT TARGET vc3d_pastix)
    add_library(vc3d_pastix INTERFACE)
    target_link_libraries(vc3d_pastix INTERFACE PaStiX::PaStiX)
    target_compile_definitions(vc3d_pastix INTERFACE VC_HAVE_PASTIX=1)
  endif()
endif()
