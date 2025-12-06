# OpenABF - Angle-Based Flattening library
# https://gitlab.com/educelab/OpenABF
# Licensed under Apache 2.0

add_library(OpenABF INTERFACE)
target_include_directories(OpenABF INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(OpenABF INTERFACE Eigen3::Eigen)

# Install the interface library
if(VC_INSTALL_LIBS)
    install(
        TARGETS OpenABF
        EXPORT "${targets_export_name}"
        COMPONENT "Libraries"
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
        DESTINATION "${include_install_dir}"
        COMPONENT "Libraries"
        FILES_MATCHING PATTERN "*.hpp"
    )
endif()
