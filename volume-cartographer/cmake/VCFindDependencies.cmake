option(VC_BUILD_JSON "Build in-source JSON library" off)
option(VC_BUILD_Z5 "Build in-source z5 header only library" on)

if(VC_BUILD_Z5)
    # Declare the project
    FetchContent_Declare(
            z5
            GIT_REPOSITORY https://github.com/constantinpape/z5.git
            GIT_TAG ee2081bb974fe0d0d702538400c31c38b09f1629
    )

    # Populate the project but exclude from all
    FetchContent_GetProperties(z5)
    if(NOT z5_POPULATED)
        FetchContent_Populate(z5)
    endif()
    option(BUILD_Z5PY "" OFF)
    option(WITH_BLOSC "" ON)
    add_subdirectory(${z5_SOURCE_DIR} ${z5_BINARY_DIR} EXCLUDE_FROM_ALL)
    # target_link_libraries(z5 INTERFACE blosc)
else()
    find_package(z5 REQUIRED)
endif()

if((VC_BUILD_APPS OR VC_BUILD_UTILS) AND VC_BUILD_GUI)
    find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network)
    # qt_standard_project_setup() #NOTE below settings for QT < 6.3, commented command for qt >= 6.3, ubuntu 22.04 has qt 6.2!
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
     
    if(NOT DEFINED qt_generate_deploy_app_script)
        message(WARNING "WARNING qt_generate_deploy_app_script MISSING!")
        function(qt_generate_deploy_app_script)
        endfunction()
    endif()
     
endif()

option(VC_WITH_CUDA_SPARSE "use cudss" ON)
if (VC_WITH_CUDA_SPARSE)
    add_definitions(-DVC_USE_CUDA_SPARSE=1)
endif()

### ceres-solver ###
find_package(Ceres REQUIRED)


### Eigen ###
find_package(Eigen3 3.3 REQUIRED)
if(CMAKE_GENERATOR MATCHES "Ninja|.*Makefiles.*" AND "${CMAKE_BUILD_TYPE}" MATCHES "^$|Debug")
    message(AUTHOR_WARNING "Configuring a Debug build. Eigen performance will be degraded. If you need debug symbols, \
    consider setting CMAKE_BUILD_TYPE to RelWithDebInfo. Otherwise, set to Release to maximize performance.")
endif()

### OpenCV ###
find_package(OpenCV 3 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 4 QUIET REQUIRED)
endif()


if(VC_USE_OPENMP)
    message(STATUS "OpenMP support enabled")

    find_package(OpenMP REQUIRED)
    set(XTENSOR_USE_OPENMP 1)
else()
    message(STATUS "OpenMP support disabled")

    set(XTENSOR_USE_OPENMP 0)
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C ALIAS openmp_stub)
endif()

set(XTENSOR_USE_XSIMD 1)
find_package(xtensor REQUIRED)


### spdlog ###
find_package(spdlog 1.4.2 CONFIG REQUIRED)

if(VC_BUILD_JSON)
    FetchContent_Declare(
            json
            DOWNLOAD_EXTRACT_TIMESTAMP ON
            URL https://github.com/nlohmann/json/archive/v3.11.3.tar.gz
    )

    FetchContent_GetProperties(json)
    if(NOT json_POPULATED)
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        set(JSON_Install ON CACHE INTERNAL "")
        FetchContent_Populate(json)
        add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
else()
    find_package(nlohmann_json 3.9.1 REQUIRED)
endif()


### Boost (for app use only) ###
if(VC_BUILD_APPS OR VC_BUILD_UTILS)
    find_package(Boost 1.58 REQUIRED COMPONENTS system program_options)
endif()
