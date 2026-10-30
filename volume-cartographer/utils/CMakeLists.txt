add_library(utils src/Json.cpp src/aws_auth.cpp src/http_fetch.cpp src/sftp_fetch.cpp)
# SFTP protocol client shared with the object store's hosted hubs.
target_link_libraries(utils PRIVATE vc_reader::sftp)
target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(utils PUBLIC cxx_std_23)
target_link_libraries(utils PUBLIC
    nlohmann_json::nlohmann_json
    CURL::libcurl
)
set_target_properties(utils PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)
