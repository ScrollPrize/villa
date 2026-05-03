cmake_minimum_required(VERSION 3.16)

project(dense_batch_min_cut LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc ximgproc)
find_package(TIFF REQUIRED)

add_executable(dense_batch_preprocess src/dense_batch_preprocess.cpp)
target_link_libraries(dense_batch_preprocess PRIVATE ${OpenCV_LIBS} TIFF::TIFF)
target_compile_features(dense_batch_preprocess PRIVATE cxx_std_17)

add_library(dense_batch_flow SHARED src/dense_batch_preprocess.cpp)
target_compile_definitions(dense_batch_flow PRIVATE DENSE_BATCH_MIN_CUT_NO_MAIN)
target_link_libraries(dense_batch_flow PRIVATE ${OpenCV_LIBS} TIFF::TIFF)
target_compile_features(dense_batch_flow PRIVATE cxx_std_17)
