ExternalProject_Add(
    pgs-recon-utilities
    DEPENDS eigen opencv openMVG
    URL ${PROJECT_SOURCE_DIR}/utilities
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DEigen3_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake/
)