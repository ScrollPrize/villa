option(BUILD_JPEG "Build JPEG" ON)
cmake_dependent_option(BUILD_JPEG_TURBO "Build jpeg-turbo" OFF "BUILD_JPEG" OFF)
if(BUILD_JPEG)
  # Pick which libjpeg to build
  if(BUILD_JPEG_TURBO)
    # jpeg-turbo (Default)
    find_package(NASM REQUIRED)
    ExternalProject_Add(
      jpeg
      URL https://github.com/libjpeg-turbo/libjpeg-turbo/archive/3.0.3.tar.gz
      URL_HASH SHA256=46c44be837654e201d11bbf8d9fbb35b775a7d4bf653e9e709279437b10d5c8b0825ece4c8ee33f66689c263234fa2b08240fb5f5ba80e76e03891da8f64eda8
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DENABLE_SHARED:BOOL=FALSE
    )

  else()
    # libjpeg
    ExternalProject_Add(
      jpeg
      URL https://github.com/csparker247/jpeg-cmake/archive/v1.3.0.tar.gz
      URL_HASH SHA512=663faa4edcaa8f40815d8ef480de71ebc14e546e8c57d92f3ddafc10ae1badf73739221e63667c5b6edba374a075e94d7593c11264d1bbaaebeae2e635073762
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DBUILD_SHARED_LIBS:BOOL=FALSE
    )
  endif(BUILD_JPEG_TURBO)

else()
  find_package(JPEG REQUIRED)
  add_custom_target(jpeg)
endif()
