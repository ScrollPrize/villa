# libvc — minimal volume cartographer core library
# C++26, no OpenCV, no Qt, no Boost

find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)
find_package(TIFF REQUIRED)

pkg_check_modules(LIBDE265 REQUIRED libde265)
find_library(X265_LIB x265 REQUIRED)
find_path(X265_INC x265.h REQUIRED)

add_library(libvc STATIC json.cpp)

target_include_directories(libvc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(libvc PUBLIC cxx_std_26)

target_link_libraries(libvc
    PUBLIC
        nlohmann_json::nlohmann_json
        CURL::libcurl
        TIFF::TIFF
        ${X265_LIB}
        ${LIBDE265_LIBRARIES}
)

target_include_directories(libvc PRIVATE ${X265_INC} ${LIBDE265_INCLUDE_DIRS})

set_target_properties(libvc PROPERTIES
    AUTOMOC OFF
    AUTORCC OFF
    AUTOUIC OFF
)
