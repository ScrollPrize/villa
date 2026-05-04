# --- VC dependencies ----------------------------------------------------------
include(FetchContent)

# Recursively suppress all warnings for vendored subproject targets.
function(vc_suppress_warnings dir)
    get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(t IN LISTS targets)
        get_property(type TARGET "${t}" PROPERTY TYPE)
        if(type MATCHES "STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY|OBJECT_LIBRARY|EXECUTABLE")
            target_compile_options("${t}" PRIVATE -w)
        endif()
    endforeach()
    get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
    foreach(sd IN LISTS subdirs)
        vc_suppress_warnings("${sd}")
    endforeach()
endfunction()

# ---- utils is now vendored in utils/ and added via add_subdirectory() ------
# (see top-level CMakeLists.txt)

# ---- xtensor removed — replaced by core/include/vc/core/types/Array3D.hpp ----

# ---- Qt (apps / utils) -------------------------------------------------------
find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network Concurrent OpenGLWidgets)
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
if (VC_WITH_CUDA_SPARSE)
    add_definitions(-DVC_USE_CUDA_SPARSE=1)
endif()

# ---- Ceres -------------------------------------------------------------------
find_package(Ceres REQUIRED)

# ---- Eigen -------------------------------------------------------------------
# Accept Eigen 3.3+ or 5.x (Eigen jumped from 3.4 to 5.0)
find_package(Eigen3 REQUIRED)
if (CMAKE_GENERATOR MATCHES "Ninja|.*Makefiles.*" AND "${CMAKE_BUILD_TYPE}" MATCHES "^$|Debug")
    message(AUTHOR_WARNING
        "Configuring a Debug build. Eigen performance will be degraded. "
        "Consider RelWithDebInfo for symbols, or Release for max performance.")
endif()

# ---- OpenCV ------------------------------------------------------------------
find_package(OpenCV 3 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 4 QUIET REQUIRED)
endif()

# ---- CGAL --------------------------------------------------------------------
find_package(CGAL QUIET)
if (NOT CGAL_FOUND)
    message(FATAL_ERROR
        "CGAL is required but was not found.\n"
        "Please install it first (Ubuntu): sudo apt-get install -y libcgal-dev\n"
        "If installed in a non-standard prefix, set CGAL_DIR to the directory containing\n"
        "CGALConfig.cmake (or cgal-config.cmake), or add that directory to CMAKE_PREFIX_PATH.")
endif()

# ---- OpenMP ------------------------------------------------------------------
# Auto-disable OpenMP when using Clang (Clang's OpenMP support is often problematic)
if (VC_USE_OPENMP AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Clang detected — disabling OpenMP (use GCC for OpenMP support)")
    set(VC_USE_OPENMP OFF)
endif()

if (VC_USE_OPENMP)
    message(STATUS "OpenMP support enabled")
    if(APPLE)
        # On macOS, use standalone libomp package to match OpenBLAS/Ceres
        # (avoids duplicate libomp runtime error with LLVM's libomp)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
            message(STATUS "Using standalone libomp from: ${LIBOMP_PREFIX}")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
            find_package(OpenMP REQUIRED)
        else()
            message(STATUS "Standalone libomp not found, using default OpenMP")
            find_package(OpenMP REQUIRED)
        endif()
    else()
        find_package(OpenMP REQUIRED)
    endif()
else()
    message(STATUS "OpenMP support disabled")
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C  ALIAS openmp_stub)
endif()

# ---- nlohmann/json -----------------------------------------------------------
FetchContent_Declare(
    json
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    URL https://github.com/nlohmann/json/archive/v3.12.0.tar.gz
)
FetchContent_GetProperties(json)
if (NOT json_POPULATED)
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install   ON  CACHE INTERNAL "")
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
    vc_suppress_warnings("${json_SOURCE_DIR}")
endif()

# ---- c-blosc (compression for zarr chunks) -----------------------------------
# Use system-installed libblosc instead of building from source.
find_package(PkgConfig REQUIRED)
pkg_check_modules(BLOSC REQUIRED IMPORTED_TARGET blosc)
# Alias so existing target_link_libraries(... blosc_static) still works
if(NOT TARGET blosc_static)
    add_library(blosc_static ALIAS PkgConfig::BLOSC)
endif()

# ---- CURL (for HTTP chunk source / remote volumes) ---------------------------
find_package(CURL REQUIRED)

# ---- TIFF --------------------------------------------------------------------
find_package(TIFF REQUIRED)

# ---- Boost (apps/utils only) -------------------------------------------------
find_package(Boost REQUIRED COMPONENTS program_options)

# ---- libbacktrace (crash dumper, file/line resolution) -----------------------
# Only required on Linux; the crash handler is Linux-only and reduces to a
# no-op stub on other platforms (see core/src/CrashHandler.cpp).
# libbacktrace ships bundled with gcc as a static archive at
# /usr/lib/gcc/<triple>/<ver>/libbacktrace.a. It's also packaged standalone on
# some distros (Ubuntu 26.04+) as libbacktrace-dev. Try the standalone shared
# library first and fall back to gcc's bundled static archive — that path is
# universally available wherever gcc is installed (including Ubuntu 24.04 CI).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(VC_LIBBACKTRACE NAMES backtrace)
    if(NOT VC_LIBBACKTRACE)
        find_program(VC_BACKTRACE_GCC NAMES gcc cc REQUIRED)
        execute_process(
            COMMAND ${VC_BACKTRACE_GCC} -print-file-name=libbacktrace.a
            OUTPUT_VARIABLE VC_LIBBACKTRACE_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(VC_LIBBACKTRACE_PATH AND EXISTS "${VC_LIBBACKTRACE_PATH}")
            set(VC_LIBBACKTRACE "${VC_LIBBACKTRACE_PATH}"
                CACHE FILEPATH "libbacktrace static archive bundled with gcc" FORCE)
            message(STATUS "libbacktrace: using gcc bundled static ${VC_LIBBACKTRACE}")
        endif()
    endif()
    if(NOT VC_LIBBACKTRACE)
        message(FATAL_ERROR
            "libbacktrace not found. Install libbacktrace-dev (Ubuntu 26.04+) or "
            "ensure gcc is installed (gcc bundles libbacktrace.a).")
    endif()
endif()

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
