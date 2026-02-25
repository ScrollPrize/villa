add_library(vc_zarr
    src/Zarr.cpp
)

target_include_directories(vc_zarr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(vc_zarr
    PUBLIC
        nlohmann_json::nlohmann_json
        xtensor
    PRIVATE
        blosc_static
)

# Mark blosc headers as SYSTEM to suppress warnings
get_target_property(_blosc_inc blosc_static INTERFACE_INCLUDE_DIRECTORIES)
if(_blosc_inc)
    target_include_directories(vc_zarr SYSTEM PRIVATE ${_blosc_inc})
endif()


