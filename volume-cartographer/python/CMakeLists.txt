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

set_target_properties(vc_python PROPERTIES
    OUTPUT_NAME vc
)

target_link_libraries(vc_python PRIVATE vc_core)

install(TARGETS vc_python
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
