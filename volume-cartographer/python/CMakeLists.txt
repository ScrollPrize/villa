find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE VC_NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(VC_NANOBIND_CMAKE_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${VC_NANOBIND_CMAKE_DIR}")
endif()

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(vc_python
    NOMINSIZE
    volume_bindings.cpp
)

nanobind_add_module(vc_lasagna_amgx
    NOMINSIZE
    lasagna_amgx_bindings.cpp
)

set_target_properties(vc_python PROPERTIES
    OUTPUT_NAME vc
    INSTALL_RPATH "$ORIGIN"
)

set_target_properties(vc_lasagna_amgx PROPERTIES
    OUTPUT_NAME vc_lasagna_amgx
    INSTALL_RPATH "$ORIGIN"
)

target_link_libraries(vc_python PRIVATE vc_core)
target_link_libraries(vc_lasagna_amgx PRIVATE vc_lasagna)

set_target_properties(vc_core vc_lasagna utils utils_c3d_codec c3d PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS vc_python vc_lasagna_amgx
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

install(TARGETS vc_core vc_lasagna utils utils_c3d_codec c3d
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
)
