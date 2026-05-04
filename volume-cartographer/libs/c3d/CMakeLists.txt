add_library(c3d STATIC c3d.c)

target_include_directories(c3d PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_features(c3d PRIVATE c_std_23)

target_compile_options(c3d PRIVATE -fno-math-errno -fno-trapping-math -w)

target_link_libraries(c3d PRIVATE m)

set_target_properties(c3d PROPERTIES POSITION_INDEPENDENT_CODE ON)
