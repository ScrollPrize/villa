nanobind_add_module(vc_volume
    NOMINSIZE
    vc/volume.cpp
)
set_target_properties(vc_volume PROPERTIES
    OUTPUT_NAME volume
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    INSTALL_RPATH "$ORIGIN"
)
target_link_libraries(vc_volume PRIVATE vc_core)
target_compile_features(vc_volume PRIVATE cxx_std_23)

nanobind_add_module(vc_surface_index
    vc/surface_index.cpp
)
set_target_properties(vc_surface_index PROPERTIES
    OUTPUT_NAME surface_index
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc"
    INSTALL_RPATH "$ORIGIN"
)
target_link_libraries(vc_surface_index PRIVATE vc_core)
target_compile_features(vc_surface_index PRIVATE cxx_std_23)

install(TARGETS vc_volume vc_surface_index
    LIBRARY DESTINATION vc
    RUNTIME DESTINATION vc
)
install(FILES vc/__init__.py DESTINATION vc)
