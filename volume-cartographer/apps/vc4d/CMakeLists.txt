# vc4d — ground-up volume viewer on libvc
# Plugin-based architecture. Core deps: Qt6, libvc.
# Future: VTK, ITK.

set(CMAKE_AUTOMOC ON)

add_executable(vc4d
    main.cpp
    app.hpp
    viewer.hpp
    plugin.hpp
)

target_include_directories(vc4d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(vc4d PUBLIC cxx_std_26)

target_link_libraries(vc4d
    libvc
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    dl
)

set_target_properties(vc4d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# vc_import — convert TIFF stacks / zarr v2 → canonical shard format
add_executable(vc_import vc_import.cpp)
target_include_directories(vc_import PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(vc_import PUBLIC cxx_std_26)
target_link_libraries(vc_import libvc TIFF::TIFF)
set_target_properties(vc_import PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build plugins
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/plugins/CMakeLists.txt)
    add_subdirectory(plugins)
endif()
