# --- VC dependencies ----------------------------------------------------------
include(FetchContent)

option(VC_BUILD_JSON "Build in-source JSON library" OFF)
option(VC_BUILD_Z5   "Build (vendor) z5 header-only library" ON)

#find_package(CURL REQUIRED)
#find_package(OpenSSL REQUIRED)
#find_package(ZLIB REQUIRED)
#find_package(glog REQUIRED)

set(BUILD_Z5PY OFF CACHE BOOL "Disable Python bits for z5" FORCE)
set(WITH_BLOSC ON  CACHE BOOL "Enable Blosc in z5"        FORCE)

# ---- xtl / xsimd / xtensor from source (before z5, which needs them) --------
set(XTENSOR_USE_XSIMD 1)

FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        master
)
FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG        13.2.0
)
FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(xtl xsimd xtensor)

# xtensor sets cxx_std_20 INTERFACE which can downgrade our C++23; upgrade it
set_property(TARGET xtensor PROPERTY INTERFACE_COMPILE_FEATURES cxx_std_23)

# Point z5's find_package(xtensor) (and transitive deps) at FetchContent builds
set(xtl_DIR     "${FETCHCONTENT_BASE_DIR}/xtl-build"     CACHE PATH "" FORCE)
set(xsimd_DIR   "${FETCHCONTENT_BASE_DIR}/xsimd-build"   CACHE PATH "" FORCE)
set(xtensor_DIR "${FETCHCONTENT_BASE_DIR}/xtensor-build" CACHE PATH "" FORCE)

FetchContent_Declare(
    z5
    GIT_REPOSITORY https://github.com/constantinpape/z5.git
)
FetchContent_MakeAvailable(z5)

# z5's CMakeLists uses include_directories() which doesn't propagate;
# link xtensor onto the z5 INTERFACE target so consumers get the headers.
target_link_libraries(z5 INTERFACE xtensor)

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
find_package(OpenCV 3 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 4 QUIET REQUIRED)
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

# ---- xtensor/xsimd (already fetched above, before z5) -----------------------

# ---- nlohmann/json -----------------------------------------------------------
if (VC_BUILD_JSON)
    FetchContent_Declare(
        json
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        URL https://github.com/nlohmann/json/archive/v3.11.3.tar.gz
    )
    FetchContent_GetProperties(json)
    if (NOT json_POPULATED)
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        set(JSON_Install   ON  CACHE INTERNAL "")
        FetchContent_Populate(json)
        add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
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
