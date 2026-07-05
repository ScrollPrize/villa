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

nanobind_add_module(vc_cache_codec
    NOMINSIZE
    vc/cache_codec.cpp
)
set_target_properties(vc_cache_codec PROPERTIES
    OUTPUT_NAME cache_codec
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    # See vc_volume above: libs are one dir up in site-packages root.
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
)
target_link_libraries(vc_cache_codec PRIVATE vc_core)
target_compile_features(vc_cache_codec PRIVATE cxx_std_23)

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

install(TARGETS vc_volume vc_surface_index vc_cache_codec
    LIBRARY DESTINATION vc
    RUNTIME DESTINATION vc
)
install(FILES vc/__init__.py DESTINATION vc)
