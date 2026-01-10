# Minimal Blosc finder wired to the vendored c-blosc2 submodule.

if (TARGET blosc2)
    set(BLOSC_FOUND TRUE)
    set(BLOSC_LIBRARIES blosc2)
    set(BLOSC_INCLUDE_DIR
        "${CMAKE_SOURCE_DIR}/thirdparty/blosc-compat/include"
        "${CMAKE_SOURCE_DIR}/thirdparty/c-blosc2/include"
    )
else()
    message(FATAL_ERROR "BLOSC requested but vendored c-blosc2 target was not found.")
endif()
