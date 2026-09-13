ExternalProject_Add(
    openMVG
    DEPENDS eigen jpeg
    GIT_REPOSITORY https://github.com/openMVG/openMVG.git
    GIT_TAG 6d6b1dd
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    SOURCE_SUBDIR src
    PATCH_COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVG-v1.6-UpdateSensorDB.diff || true
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DEigen3_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake/
        -DEIGENSPARSE:BOOL=ON
        -DOpenMVG_BUILD_DOC:BOOL=OFF
        -DOpenMVG_BUILD_EXAMPLES:BOOL=OFF
        -DOpenMVG_BUILD_GUI_SOFTWARES:BOOL=OFF
)
ExternalProject_Get_Property(openMVG SOURCE_DIR)
ExternalProject_Add_Step(
    openMVG
    extra_patch
    COMMENT "Performing patch step #2 for 'openMVG'"
    COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVG-v2.1-FixExpectOpenMP.diff || true
    WORKING_DIRECTORY ${SOURCE_DIR}
    DEPENDEES patch
    DEPENDERS configure
)