# --- VC dependencies ----------------------------------------------------------
include(FetchContent)


set(BUILD_Z5PY OFF CACHE BOOL "Disable Python bits for z5" FORCE)
set(WITH_BLOSC ON  CACHE BOOL "Enable Blosc in z5"        FORCE)

# ---- xtl / xsimd / xtensor from source (before z5, which needs them) --------
set(XTENSOR_USE_XSIMD 1)

FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        0.8.1
)
FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG        13.2.0
)
FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        0.27.1
)
FetchContent_MakeAvailable(xtl xsimd xtensor)

# xtensor sets cxx_std_20 INTERFACE which can downgrade our C++23; upgrade it
set_property(TARGET xtensor PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_23)

# Mark xtensor-stack headers as SYSTEM to suppress warnings from -Weverything
# This requires getting the interface include dirs and re-adding them as SYSTEM
foreach(_target xtl xsimd xtensor)
    get_target_property(_inc_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(_inc_dirs)
        set_target_properties(${_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
        target_include_directories(${_target} SYSTEM INTERFACE ${_inc_dirs})
    endif()
endforeach()

# Point z5's find_package(xtensor) (and transitive deps) at FetchContent builds
set(xtl_DIR     "${FETCHCONTENT_BASE_DIR}/xtl-build"     CACHE PATH "" FORCE)
set(xsimd_DIR   "${FETCHCONTENT_BASE_DIR}/xsimd-build"   CACHE PATH "" FORCE)
set(xtensor_DIR "${FETCHCONTENT_BASE_DIR}/xtensor-build" CACHE PATH "" FORCE)


FetchContent_Declare(
    z5
    GIT_REPOSITORY https://github.com/constantinpape/z5.git
    GIT_TAG        2.0.20
)
FetchContent_MakeAvailable(z5)

# z5's CMakeLists uses include_directories() which doesn't propagate;
# link xtensor onto the z5 INTERFACE target so consumers get the headers.
target_link_libraries(z5 INTERFACE xtensor)

# Mark z5 headers as SYSTEM to suppress warnings
get_target_property(_z5_inc_dirs z5 INTERFACE_INCLUDE_DIRECTORIES)
if(_z5_inc_dirs)
    set_target_properties(z5 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
    target_include_directories(z5 SYSTEM INTERFACE ${_z5_inc_dirs})
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

# ---- OpenMP ------------------------------------------------------------------
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

# ---- xtensor/xsimd (already fetched above, before z5) -----------------------

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
endif()

# ---- TIFF --------------------------------------------------------------------
find_package(TIFF REQUIRED)

# ---- Boost (apps/utils only) -------------------------------------------------
find_package(Boost REQUIRED COMPONENTS program_options)

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
