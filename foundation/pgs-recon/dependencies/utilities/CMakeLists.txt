cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(pgs-recon-utils VERSION 1.0.0)

# find dependencies
find_package(OpenMVG REQUIRED)
find_package(OpenCV 4.7 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Boost 1.70 CONFIG REQUIRED COMPONENTS system program_options)

# header-only dependencies
include(FetchContent)
FetchContent_Declare(
        libcore
        GIT_REPOSITORY https://github.com/educelab/libcore.git
        GIT_TAG c061afd
        EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(libcore)
FetchContent_Declare(
        indicators
        GIT_REPOSITORY https://github.com/p-ranav/indicators.git
        GIT_TAG ac6c93e
        EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(indicators)

# global scaler utility
add_executable(global_scaler src/RANSAC.hpp src/global_scaler.cpp)
target_include_directories(global_scaler
    PUBLIC
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${indicators_SOURCE_DIR}/include>
)
target_link_libraries(global_scaler
    Boost::program_options
    Eigen3::Eigen
    opencv_calib3d
    opencv_core
    opencv_imgcodecs
    opencv_objdetect
    OpenMVG::openMVG_camera
    OpenMVG::openMVG_geometry
    OpenMVG::openMVG_multiview
    OpenMVG::openMVG_sfm
)
target_compile_features(global_scaler PUBLIC cxx_std_17)
set_target_properties(global_scaler PROPERTIES OUTPUT_NAME "pgs-global-scaler")

# generate aruco markers utility
add_executable(gen_aruco_markers src/gen_aruco_markers.cpp)
target_include_directories(gen_aruco_markers
    PUBLIC
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
)
target_link_libraries(gen_aruco_markers
        Boost::program_options
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_objdetect
)
target_compile_features(gen_aruco_markers PUBLIC cxx_std_17)
set_target_properties(gen_aruco_markers PROPERTIES OUTPUT_NAME "pgs-generate-markers")

# install
include(GNUInstallDirs)
install(TARGETS
    global_scaler
    gen_aruco_markers
)