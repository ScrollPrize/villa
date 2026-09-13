option(BUILD_OPENCV "Build OpenCV" ON)
if(BUILD_OPENCV)
externalproject_add(
    opencv
    DEPENDS jpeg
    URL https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz
    URL_HASH SHA512=b4f7248f89f1cd146dbbae7860a17131cd29bd3cb81db1e678abfcfbf2d8fa4a7633bfd0edbf50afae7b838c8700e8c0d0bb05828139d5cb5662df6bbf3eb92c
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DWITH_VTK:BOOL=OFF
        -DBUILD_JPEG:BOOL=OFF
        -DBUILD_TESTS:BOOL=OFF
        -DBUILD_PERF_TESTS:BOOL=OFF
        -DBUILD_EXAMPLES:BOOL=OFF
        -DWITH_FFMPEG:BOOL=OFF
        -DWITH_OPENEXR:BOOL=OFF
        -DOBSENSOR_USE_ORBBEC_SDK:BOOL=OFF
        -DWITH_OBSENSOR:BOOL=OFF
)
else()
  find_package(OpenCV CONFIG REQUIRED)
  add_custom_target(opencv)
endif()
