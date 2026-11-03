# libigl via FetchContent for flatboi (SLIM).

include(FetchContent)

set(FLATBOI_LIBIGL_COMMIT "ae8f959ea26d7059abad4c698aba8d6b7c3205e8"
    CACHE STRING "libigl commit to fetch")

FetchContent_Declare(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        ${FLATBOI_LIBIGL_COMMIT}
    PATCH_COMMAND  ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_SOURCE_DIR}/libs/libigl_changes <SOURCE_DIR>
)
FetchContent_MakeAvailable(libigl)
