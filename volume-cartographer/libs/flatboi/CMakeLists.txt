cmake_minimum_required(VERSION 3.22)
project(flatboi LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(LIBIGL_WITH_CHOLMOD ON CACHE BOOL "" FORCE)

set(FLATBOI_LIBIGL_COMMIT "ae8f959ea26d7059abad4c698aba8d6b7c3205e8"
    CACHE STRING "libigl commit to fetch")

FetchContent_Declare(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        ${FLATBOI_LIBIGL_COMMIT}
    PATCH_COMMAND  ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_LIST_DIR}/../libigl_changes <SOURCE_DIR>
)
FetchContent_MakeAvailable(libigl)

add_executable(flatboi flatboi.cpp)
target_link_libraries(flatboi PRIVATE igl::core igl::cholmod)
# SLIM_CACHED reuses the AᵀWA sparsity pattern across iters (only updates
# values), matching the CHOLMOD analyzePattern-once pattern.
target_compile_definitions(flatboi PRIVATE SLIM_CACHED)

if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(OpenMP)
endif()
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(flatboi PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(OpenCV REQUIRED COMPONENTS imgcodecs core)
target_link_libraries(flatboi PRIVATE ${OpenCV_LIBS})
