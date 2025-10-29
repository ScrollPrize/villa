set(PYTHON_BINDINGS_OUTPUT_DIR "${CMAKE_BINARY_DIR}/python/vc/tracing")

pybind11_add_module(vc_tracing MODULE
    src/vc/tracing/module.cpp
)

target_link_libraries(vc_tracing
    PRIVATE
        vc_tracer
        vc_core
        vc_ui
)

set_target_properties(vc_tracing PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/$<CONFIG>/vc/tracing"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/$<CONFIG>/vc/tracing"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/$<CONFIG>/vc/tracing"
)

install(
    TARGETS vc_tracing
    COMPONENT python
    LIBRARY DESTINATION vc/tracing
    RUNTIME DESTINATION vc/tracing
    ARCHIVE DESTINATION vc/tracing
)

install(
    DIRECTORY src/vc
    DESTINATION .
    COMPONENT python
    FILES_MATCHING
        PATTERN "*.py"
)
