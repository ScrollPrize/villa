option(BUILD_EIGEN "Build Eigen" ON)
if(BUILD_EIGEN)
  ExternalProject_Add(
      eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
      URL_HASH SHA512=ba75ecb760e32acf4ceaf27115468e65d4f77c44f8d519b5a13e7940af2c03a304ad433368cb6d55431f307c5c39e2666ab41d34442db3cf441638e51f5c3b6a
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
  )
else()
  find_package(Eigen3 3.3 REQUIRED)
  add_custom_target(eigen)
endif()
