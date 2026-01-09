list(PREPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(FetchContent)

find_package(Blosc2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV QUIET)


set(XTENSOR_USE_XSIMD 1)
find_package(xtl REQUIRED)
find_package(xsimd REQUIRED)
find_package(xtensor REQUIRED)

find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)


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


if (VC_WITH_PASTIX)
  find_package(PaStiX REQUIRED)
  message(STATUS "PaStiX found: ${PASTIX_LIBRARY}")
  if (NOT TARGET vc3d_pastix)
    add_library(vc3d_pastix INTERFACE)
    target_link_libraries(vc3d_pastix INTERFACE PaStiX::PaStiX)
    target_compile_definitions(vc3d_pastix INTERFACE VC_HAVE_PASTIX=1)
  endif()
endif()

# Try a preinstalled z5 first, unless the user explicitly forces vendoring.
if (VC_BUILD_Z5)
    find_package(z5 CONFIG QUIET)
    if (z5_FOUND)
        message(STATUS "Using preinstalled z5 at: ${z5_DIR} (set VC_BUILD_Z5=OFF to force this; keep ON to try vendoring).")
        set(VC_BUILD_Z5 OFF CACHE BOOL "" FORCE)
    endif()
endif()

if (NOT VC_BUILD_Z5)
    # Use a system / previously installed z5
    find_package(z5 CONFIG REQUIRED)
else()
    # Vendoring path: fetch z5 and add it as a subdir.
    # z5 defines options; set them in the cache *before* adding the subproject.
    set(BUILD_Z5PY OFF CACHE BOOL "Disable Python bits for z5" FORCE)
    set(WITH_BLOSC ON  CACHE BOOL "Enable Blosc in z5"        FORCE)

    # On CMake ≥4, compatibility with <3.5 was removed. Setting this floor
    # avoids errors if z5 asks for 3.1 in its CMakeLists.
    if (NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    endif()

    # FetchContent: manually populate and patch z5 to avoid xtensor conflict
    FetchContent_Declare(
            z5
            GIT_REPOSITORY https://github.com/constantinpape/z5.git
            GIT_TAG        ee2081bb974fe0d0d702538400c31c38b09f1629
    )
    FetchContent_GetProperties(z5)
    if(NOT z5_POPULATED)
        FetchContent_Populate(z5)
        # Replace the find_package(xtensor) line in z5's CMakeLists.txt with a comment
        file(READ "${z5_SOURCE_DIR}/CMakeLists.txt" z5_cmake_content)
        string(REPLACE "find_package(xtensor REQUIRED)" "# find_package(xtensor REQUIRED) - already found by main project" z5_cmake_content "${z5_cmake_content}")
        file(WRITE "${z5_SOURCE_DIR}/CMakeLists.txt" "${z5_cmake_content}")
        add_subdirectory(${z5_SOURCE_DIR} ${z5_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()

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
    find_package(nlohmann_json REQUIRED)
endif()