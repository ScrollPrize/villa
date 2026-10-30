add_library(utils src/Json.cpp)
target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(utils PUBLIC cxx_std_23)
target_link_libraries(utils PUBLIC
    nlohmann_json::nlohmann_json
)
set_target_properties(utils PROPERTIES AUTOMOC OFF AUTORCC OFF AUTOUIC OFF)
