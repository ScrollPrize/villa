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
# Removed native target migration: extension modules take precedence over the
# forwarding .py module, so discard only stale generated vcz1 modules from an
# existing build tree before copying the compatibility shim.
file(GLOB VC_STALE_VCZ1_MODULES
    "${CMAKE_CURRENT_BINARY_DIR}/vc/compression/vcz1*${CMAKE_SHARED_MODULE_SUFFIX}")
if(VC_STALE_VCZ1_MODULES)
    file(REMOVE ${VC_STALE_VCZ1_MODULES})
endif()
configure_file(vc/compression/vcz1_numcodecs.py
    "${CMAKE_CURRENT_BINARY_DIR}/vc/compression/vcz1_numcodecs.py"
    COPYONLY
)
configure_file(vc/compression/vcz1.py
    "${CMAKE_CURRENT_BINARY_DIR}/vc/compression/vcz1.py"
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

set_target_properties(vc_core vc_lasagna utils utils_c3d_codec c3d utils_volcomp_codec volcomp PROPERTIES
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

nanobind_add_module(vc_grid_raycast
    NOMINSIZE
    vc/grid_raycast.cpp
)
set_target_properties(vc_grid_raycast PROPERTIES
    OUTPUT_NAME grid_raycast
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
)
target_link_libraries(vc_grid_raycast PRIVATE OpenMP::OpenMP_CXX)
target_compile_features(vc_grid_raycast PRIVATE cxx_std_23)
target_compile_options(vc_grid_raycast PRIVATE
    $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-O3>
)

install(TARGETS vc_lasagna_amgx
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

install(TARGETS vc_core vc_lasagna utils utils_c3d_codec c3d utils_volcomp_codec volcomp
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
)

install(TARGETS vc_volume vc_surface_index vc_grid_raycast
    LIBRARY DESTINATION vc
    RUNTIME DESTINATION vc
)
install(FILES vc/__init__.py DESTINATION vc)
install(FILES vc/compression/__init__.py DESTINATION vc/compression)
install(FILES
    vc/compression/vcz1.py
    vc/compression/vcz1_numcodecs.py
    DESTINATION vc/compression
)
