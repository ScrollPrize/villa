# Note: Setting VCG_ROOT forces looking in the install prefix *first*
ExternalProject_Add(
    openMVS
    DEPENDS eigen VCG CGAL jpeg opencv
    GIT_REPOSITORY https://github.com/cdcseacave/openMVS
    GIT_TAG 36ffb22
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    PATCH_COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVS-v2.3-FixCPP20.diff || true
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DVCG_ROOT:PATH=${CMAKE_INSTALL_PREFIX}/include/vcg
        -DEigen3_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake/
        -DOpenMVS_USE_PYTHON:BOOL=OFF
        -DOpenMVS_USE_CUDA:BOOL=OFF
        -DOpenMVS_USE_SSE:BOOL=${HAS_SSE}
        -DENABLE_PRECOMPILED_HEADERS:BOOL=OFF
)
