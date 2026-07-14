cmake_minimum_required(VERSION 3.5)
project(HDBSCAN)

# Set the build type if it's not already set
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE HDBSCAN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Distance
    ${CMAKE_CURRENT_SOURCE_DIR}/Hdbscan
    ${CMAKE_CURRENT_SOURCE_DIR}/HdbscanStar
    ${CMAKE_CURRENT_SOURCE_DIR}/Runner
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils
)

add_library(hdbscan STATIC ${HDBSCAN_SOURCES})
