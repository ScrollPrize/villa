add_library(utils src/zarr.cpp src/Json.cpp src/aws_auth.cpp src/http_fetch.cpp src/sftp_fetch.cpp
    src/process_shutdown.cpp)
# SFTP protocol client shared with the object store's hosted hubs.
target_link_libraries(utils PRIVATE vc_object_store::vc_sftp)
target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(utils PUBLIC cxx_std_23)
target_link_libraries(utils PUBLIC
    nlohmann_json::nlohmann_json
    CURL::libcurl
    zstd::libzstd
    LZ4::lz4
    ZLIB::ZLIB
)
set_target_properties(utils PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)

add_library(utils_c3d_codec src/c3d_codec.cpp)
target_include_directories(utils_c3d_codec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(utils_c3d_codec PUBLIC cxx_std_23)
target_link_libraries(utils_c3d_codec PUBLIC c3d)
set_target_properties(utils_c3d_codec PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)

add_library(utils_volcomp_codec src/volcomp_codec.cpp)
target_include_directories(utils_volcomp_codec PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(utils_volcomp_codec PUBLIC cxx_std_23)
target_link_libraries(utils_volcomp_codec PUBLIC volcomp)
set_target_properties(utils_volcomp_codec PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)
