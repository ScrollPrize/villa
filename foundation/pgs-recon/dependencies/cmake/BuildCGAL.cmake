option(BUILD_CGAL "Build CGAL" ON)
if(BUILD_CGAL)
  ExternalProject_Add(
      CGAL
      DEPENDS eigen
      URL https://github.com/CGAL/cgal/releases/download/v5.6.1/CGAL-5.6.1-library.tar.xz
      URL_HASH SHA512=6c0d0d2c4afcf2e69aba1c77c2db2a883cfbd80347e2f0974431ebf002f7ce24c84fda02ae85eeb8cdcd84821e78a259400faebf162a4514581bae41c30bbbaf
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DCMAKE_BUILD_TYPE:STRING=Release
          -DWITH_CGAL_Qt5:BOOL=OFF
          -DWITH_examples:BOOL=OFF
          -DWITH_demos:BOOL=OFF
  )
else()
  find_package(CGAL CONFIG REQUIRED)
  add_custom_target(CGAL)
endif()
