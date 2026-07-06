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

configure_file(vc/__init__.py
    "${CMAKE_CURRENT_BINARY_DIR}/vc/__init__.py"
    COPYONLY
)
configure_file(vc/compression/__init__.py
    "${CMAKE_CURRENT_BINARY_DIR}/vc/compression/__init__.py"
    COPYONLY
)
configure_file(vc/compression/vcz1_numcodecs.py
    "${CMAKE_CURRENT_BINARY_DIR}/vc/compression/vcz1_numcodecs.py"
    COPYONLY
)

nanobind_add_module(vc_lasagna_amgx
    NOMINSIZE
    lasagna_amgx_bindings.cpp
)

set_target_properties(vc_lasagna_amgx PROPERTIES
    OUTPUT_NAME vc_lasagna_amgx
    INSTALL_RPATH "$ORIGIN"
)

target_link_libraries(vc_lasagna_amgx PRIVATE vc_lasagna)

set_target_properties(vc_core vc_lasagna utils utils_c3d_codec c3d PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

nanobind_add_module(vc_volume
    NOMINSIZE
    vc/volume.cpp
)
set_target_properties(vc_volume PROPERTIES
    OUTPUT_NAME volume
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    # Modules live in vc/ but libvc_core.so etc. install to the
    # site-packages root one level up, so search both.
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
)
target_link_libraries(vc_volume PRIVATE vc_core)
target_compile_features(vc_volume PRIVATE cxx_std_23)

nanobind_add_module(vc_compression_vcz1
    NOMINSIZE
    vc/compression/vcz1.cpp
)
set_target_properties(vc_compression_vcz1 PROPERTIES
    OUTPUT_NAME vcz1
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc/compression"
    # Module lives in vc/compression; libs install to site-packages root.
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../.."
)
target_link_libraries(vc_compression_vcz1 PRIVATE vc_core)
target_compile_features(vc_compression_vcz1 PRIVATE cxx_std_23)

nanobind_add_module(vc_surface_index
    vc/surface_index.cpp
)
set_target_properties(vc_surface_index PROPERTIES
    OUTPUT_NAME surface_index
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    # See vc_volume above: libs are one dir up in site-packages root.
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
)
target_link_libraries(vc_surface_index PRIVATE vc_core)
target_compile_features(vc_surface_index PRIVATE cxx_std_23)

install(TARGETS vc_lasagna_amgx
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

install(TARGETS vc_core vc_lasagna utils utils_c3d_codec c3d
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
)

install(TARGETS vc_volume vc_surface_index
    LIBRARY DESTINATION vc
    RUNTIME DESTINATION vc
)
install(TARGETS vc_compression_vcz1
    LIBRARY DESTINATION vc/compression
    RUNTIME DESTINATION vc/compression
)
install(FILES vc/__init__.py DESTINATION vc)
install(FILES vc/compression/__init__.py DESTINATION vc/compression)
install(FILES vc/compression/vcz1_numcodecs.py DESTINATION vc/compression)
