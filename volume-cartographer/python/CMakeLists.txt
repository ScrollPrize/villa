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

# $ORIGIN is ELF-only; Mach-O resolves the same relative search via @loader_path.
if(APPLE)
    set(VC_PY_RPATH_SELF "@loader_path")
    set(VC_PY_RPATH_SELF_AND_PARENT "@loader_path;@loader_path/..")
else()
    set(VC_PY_RPATH_SELF "$ORIGIN")
    set(VC_PY_RPATH_SELF_AND_PARENT "$ORIGIN:$ORIGIN/..")
endif()

configure_file(vc/__init__.py
    "${CMAKE_CURRENT_BINARY_DIR}/vc/__init__.py"
    COPYONLY
)
configure_file(vc/fiber_evaluation.py "${CMAKE_CURRENT_BINARY_DIR}/vc/fiber_evaluation.py" COPYONLY)
install(FILES vc/fiber_evaluation.py DESTINATION vc)
configure_file(vc/fiber_evaluation_vis.py "${CMAKE_CURRENT_BINARY_DIR}/vc/fiber_evaluation_vis.py" COPYONLY)
install(FILES vc/fiber_evaluation_vis.py DESTINATION vc)

nanobind_add_module(vc_lasagna_amgx
    NOMINSIZE
    lasagna_amgx_bindings.cpp
)

set_target_properties(vc_lasagna_amgx PROPERTIES
    OUTPUT_NAME vc_lasagna_amgx
    INSTALL_RPATH "${VC_PY_RPATH_SELF}"
)

target_link_libraries(vc_lasagna_amgx PRIVATE vc_lasagna)

set_target_properties(vc_core vc_lasagna utils PROPERTIES
    INSTALL_RPATH "${VC_PY_RPATH_SELF}"
)

nanobind_add_module(vc_volume
    NOMINSIZE
    vc/volume.cpp
    vc/normal_sampler.cpp
)
set_target_properties(vc_volume PROPERTIES
    OUTPUT_NAME volume
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    # Modules live in vc/ but libvc_core.so etc. install to the
    # site-packages root one level up, so search both.
    INSTALL_RPATH "${VC_PY_RPATH_SELF_AND_PARENT}"
)
target_link_libraries(vc_volume PRIVATE vc_core vc_lasagna)
target_compile_features(vc_volume PRIVATE cxx_std_23)

nanobind_add_module(vc_grid_raycast
    NOMINSIZE
    vc/grid_raycast.cpp
)
set_target_properties(vc_grid_raycast PROPERTIES
    OUTPUT_NAME grid_raycast
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    INSTALL_RPATH "${VC_PY_RPATH_SELF_AND_PARENT}"
)
target_link_libraries(vc_grid_raycast PRIVATE OpenMP::OpenMP_CXX)
target_compile_features(vc_grid_raycast PRIVATE cxx_std_23)
target_compile_options(vc_grid_raycast PRIVATE
    $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-O3>
)

nanobind_add_module(vc_fiber_trace
    NOMINSIZE
    vc/fiber_trace.cpp
)
set_target_properties(vc_fiber_trace PROPERTIES
    OUTPUT_NAME fiber_trace
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
)
target_link_libraries(vc_fiber_trace PRIVATE vc_fiber_tracer)
target_compile_features(vc_fiber_trace PRIVATE cxx_std_23)
set_target_properties(vc_fiber_tracer PROPERTIES INSTALL_RPATH "$ORIGIN")

install(TARGETS vc_lasagna_amgx
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)

# The modules link vc_reader's shared native library, found through $ORIGIN.
if(TARGET vc_reader_native)
    get_target_property(_vc_reader_native_type vc_reader_native TYPE)
    if(_vc_reader_native_type STREQUAL "SHARED_LIBRARY")
        install(TARGETS vc_reader_native LIBRARY DESTINATION . RUNTIME DESTINATION .)
    endif()
endif()

install(TARGETS vc_core vc_lasagna vc_fiber_tracer utils
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
)

install(TARGETS vc_volume vc_grid_raycast vc_fiber_trace
    LIBRARY DESTINATION vc
    RUNTIME DESTINATION vc
)
install(FILES vc/__init__.py DESTINATION vc)
install(FILES vc/fiberlets_cli.py DESTINATION vc)
if(TARGET vc_fiberlets)
    install(TARGETS vc_fiberlets RUNTIME DESTINATION vc/bin)
endif()
configure_file(vc/fiberlets_cli.py "${CMAKE_CURRENT_BINARY_DIR}/vc/fiberlets_cli.py" COPYONLY)
