find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(BLOSC REQUIRED IMPORTED_TARGET blosc)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)
find_package(SQLite3 3.24 REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
if(NOT TARGET vc_delta3d::vc_delta3d)
    find_package(vc_delta3d CONFIG QUIET)
endif()
if(NOT TARGET vc_delta3d::vc_delta3d)
    include(FetchContent)
    set(VC_DELTA3D_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(VC_DELTA3D_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(VC_DELTA3D_INSTALL_CPP OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(vc_delta3d
        GIT_REPOSITORY https://github.com/ScrollPrize/vc-delta3d.git
        GIT_TAG 457f612aa0379a73fd750d8b8e61eb8b11bf45fd)
    FetchContent_MakeAvailable(vc_delta3d)
endif()
if(TARGET vc_delta3d)
    set_target_properties(vc_delta3d PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_library(vc_reader_native SHARED "${CMAKE_CURRENT_LIST_DIR}/../src/cpp/native_reader.cpp")
add_library(vc_reader::reader ALIAS vc_reader_native)
set_target_properties(vc_reader_native PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME reader)
target_compile_features(vc_reader_native PUBLIC cxx_std_20)
target_include_directories(vc_reader_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(vc_reader_native PRIVATE CURL::libcurl PkgConfig::BLOSC PkgConfig::ZSTD
    SQLite::SQLite3 vc_delta3d::vc_delta3d Threads::Threads nlohmann_json::nlohmann_json)

include(GNUInstallDirs)
install(TARGETS vc_reader_native EXPORT vc_readerTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT vc_readerTargets NAMESPACE vc_reader:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vc_reader)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/vc_readerConfig.cmake" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vc_reader)
