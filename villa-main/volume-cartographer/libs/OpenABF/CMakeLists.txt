# OpenABF - Angle-Based Flattening library
# https://gitlab.com/educelab/OpenABF
# Licensed under Apache 2.0

add_library(OpenABF INTERFACE)
target_include_directories(OpenABF INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(OpenABF INTERFACE Eigen3::Eigen)
