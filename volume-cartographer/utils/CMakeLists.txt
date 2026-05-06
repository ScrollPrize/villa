add_library(utils STATIC src/zarr.cpp src/Json.cpp src/aws_auth.cpp)
target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(utils PUBLIC cxx_std_23)
target_link_libraries(utils PUBLIC
    nlohmann_json::nlohmann_json
    CURL::libcurl
    zstd::libzstd
    LZ4::lz4
    ZLIB::ZLIB
)
target_compile_definitions(utils PUBLIC UTILS_HAS_CURL=1)
set_target_properties(utils PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)

add_library(utils_c3d_codec STATIC src/c3d_codec.cpp)
target_include_directories(utils_c3d_codec PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(utils_c3d_codec PUBLIC cxx_std_23)
target_link_libraries(utils_c3d_codec PUBLIC c3d)
target_compile_definitions(utils_c3d_codec PUBLIC UTILS_HAS_C3D_CODEC=1)
set_target_properties(utils_c3d_codec PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)
