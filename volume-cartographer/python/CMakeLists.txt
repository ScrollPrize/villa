nanobind_add_module(vc_surface_index
    vc_surface_index.cpp
)

target_link_libraries(vc_surface_index PRIVATE vc_core)
target_compile_features(vc_surface_index PRIVATE cxx_std_23)

install(TARGETS vc_surface_index LIBRARY DESTINATION .)
