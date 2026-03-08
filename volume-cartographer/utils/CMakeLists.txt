# utils — header-only C++23 utility library (vendored)
# Source: https://github.com/SuperOptimizer/utils

find_package(nlohmann_json REQUIRED)

add_library(utils INTERFACE)
target_include_directories(utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(utils INTERFACE cxx_std_23)
target_link_libraries(utils INTERFACE nlohmann_json::nlohmann_json)

# Individual header targets (for granular dependency management)
set(UTILS_HEADERS
    hash
    lru_cache
    lock_pool
    priority_queue
    compositing
    zarr
    http_fetch
    thread_pool
    compression
)

foreach(header ${UTILS_HEADERS})
    add_library(utils_${header} INTERFACE)
    target_include_directories(utils_${header} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(utils_${header} INTERFACE cxx_std_23)
endforeach()

# Optional: find curl for http_fetch.hpp
find_package(CURL QUIET)
if(CURL_FOUND)
    target_link_libraries(utils INTERFACE CURL::libcurl)
    target_compile_definitions(utils INTERFACE UTILS_HAS_CURL=1)
    target_link_libraries(utils_http_fetch INTERFACE CURL::libcurl)
    target_link_libraries(utils_zarr INTERFACE CURL::libcurl)
endif()

# zarr needs nlohmann_json
target_link_libraries(utils_zarr INTERFACE nlohmann_json::nlohmann_json)

# Optional: find compression libraries for compression.hpp / zarr.hpp
find_library(ZSTD_LIB zstd QUIET)
find_library(LZ4_LIB lz4 QUIET)
find_library(ZLIB_LIB z QUIET)
set(UTILS_COMPRESSION_LIBS "")
if(ZSTD_LIB)
    list(APPEND UTILS_COMPRESSION_LIBS ${ZSTD_LIB})
endif()
if(LZ4_LIB)
    list(APPEND UTILS_COMPRESSION_LIBS ${LZ4_LIB})
endif()
if(ZLIB_LIB)
    list(APPEND UTILS_COMPRESSION_LIBS ${ZLIB_LIB})
endif()
if(UTILS_COMPRESSION_LIBS)
    target_link_libraries(utils INTERFACE ${UTILS_COMPRESSION_LIBS})
    target_link_libraries(utils_zarr INTERFACE ${UTILS_COMPRESSION_LIBS})
    target_link_libraries(utils_compression INTERFACE ${UTILS_COMPRESSION_LIBS})
endif()

# Optional: video codec support (H264/H265/AV1)
option(VC_WITH_VIDEO_CODEC "Enable video codec compression (H264/H265/AV1)" ON)
include(FindPkgConfig OPTIONAL)

set(_VC_VIDEO_LIBS "")
set(_VC_VIDEO_INCS "")
set(_VC_VIDEO_DEFS "")
set(_VC_VIDEO_FOUND FALSE)

if(PKG_CONFIG_FOUND AND VC_WITH_VIDEO_CODEC)
    # H.264: OpenH264
    pkg_check_modules(OPENH264 QUIET openh264)
    if(OPENH264_FOUND)
        list(APPEND _VC_VIDEO_LIBS ${OPENH264_LIBRARIES})
        list(APPEND _VC_VIDEO_INCS ${OPENH264_INCLUDE_DIRS})
        list(APPEND _VC_VIDEO_DEFS UTILS_HAS_H264=1)
        set(_VC_VIDEO_FOUND TRUE)
        message(STATUS "Video codec H.264: enabled (OpenH264)")
    endif()

    # H.265: x265 (encoder) + libde265 (decoder)
    pkg_check_modules(LIBDE265 QUIET libde265)
    find_library(X265_LIB x265 QUIET)
    find_path(X265_INC x265.h QUIET)
    if(LIBDE265_FOUND AND X265_LIB AND X265_INC)
        list(APPEND _VC_VIDEO_LIBS ${X265_LIB} ${LIBDE265_LIBRARIES})
        list(APPEND _VC_VIDEO_INCS ${X265_INC} ${LIBDE265_INCLUDE_DIRS})
        list(APPEND _VC_VIDEO_DEFS UTILS_HAS_H265=1)
        set(_VC_VIDEO_FOUND TRUE)
        message(STATUS "Video codec H.265: enabled (x265 + libde265)")
    else()
        message(STATUS "Video codec H.265: disabled (need x265 + libde265)")
    endif()

    # AV1: libaom (encoder) + dav1d (decoder)
    pkg_check_modules(DAV1D QUIET dav1d)
    pkg_check_modules(AOM QUIET aom)
    if(DAV1D_FOUND AND AOM_FOUND)
        list(APPEND _VC_VIDEO_LIBS ${AOM_LIBRARIES} ${DAV1D_LIBRARIES})
        list(APPEND _VC_VIDEO_INCS ${AOM_INCLUDE_DIRS} ${DAV1D_INCLUDE_DIRS})
        list(APPEND _VC_VIDEO_DEFS UTILS_HAS_AV1=1)
        set(_VC_VIDEO_FOUND TRUE)
        message(STATUS "Video codec AV1: enabled (libaom + dav1d)")
    else()
        message(STATUS "Video codec AV1: disabled (need libaom + dav1d)")
    endif()
endif()

add_library(utils_video_codec STATIC src/video_codec.cpp src/compress3d.c)
target_include_directories(utils_video_codec PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(utils_video_codec PUBLIC cxx_std_23)

if(_VC_VIDEO_FOUND)
    target_link_libraries(utils_video_codec PUBLIC ${_VC_VIDEO_LIBS})
    target_include_directories(utils_video_codec PRIVATE ${_VC_VIDEO_INCS})
    target_compile_definitions(utils_video_codec PUBLIC
        UTILS_HAS_VIDEO_CODEC=1
        ${_VC_VIDEO_DEFS}
    )
else()
    message(STATUS "Video codec support: disabled (no codecs found or VC_WITH_VIDEO_CODEC=OFF)")
endif()

# Tests
if(VC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
