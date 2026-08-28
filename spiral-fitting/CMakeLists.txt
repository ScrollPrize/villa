cmake_minimum_required(VERSION 3.22)
project(spiral_fitting LANGUAGES CXX)

option(VC_SPIRAL_BUILD_PYTHON "Build the vc_spiral extension modules" ON)
option(VC_SPIRAL_BUILD_TOOLS "Build native command-line tools" ON)

find_package(TIFF REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenMP QUIET COMPONENTS CXX)
include(CTest)

add_library(vc_spiral_surface_core STATIC
    spiral_graph/src/mapped_tifxyz.cpp
)
add_library(vc_spiral::surface_core ALIAS vc_spiral_surface_core)
target_compile_features(vc_spiral_surface_core PUBLIC cxx_std_23)
target_include_directories(vc_spiral_surface_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/spiral_graph/include"
)
target_link_libraries(vc_spiral_surface_core PUBLIC TIFF::TIFF)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vc_spiral_surface_core PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(vc_spiral_surface_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_subdirectory(cpp)
add_subdirectory(spiral_graph)
