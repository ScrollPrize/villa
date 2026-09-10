# Note: Must use the cdcseacave repo
ExternalProject_Add(
    VCG
    URL https://github.com/cdcseacave/VCG/archive/10d73fc.tar.gz
    URL_HASH SHA512=391b0fe7633553755a53dd1b1d6dea8c6308eba80d534fd365b899ebecfe8bbc791b40f20afe54361e52215b654278d3aca009b6c62d7fcc6bc38073eb368ed6
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    BUILD_IN_SOURCE ON
    CONFIGURE_COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/vcg
    BUILD_COMMAND ""
    INSTALL_COMMAND cp -R . ${CMAKE_INSTALL_PREFIX}/include/vcg
)
