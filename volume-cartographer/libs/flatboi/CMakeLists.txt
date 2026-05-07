cmake_minimum_required(VERSION 3.22)
project(flatboi LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(FLATBOI_LIBIGL_COMMIT "ae8f959ea26d7059abad4c698aba8d6b7c3205e8"
    CACHE STRING "libigl commit to fetch")

FetchContent_Declare(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        ${FLATBOI_LIBIGL_COMMIT}
    PATCH_COMMAND  ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_LIST_DIR}/../libigl_changes <SOURCE_DIR>
)
FetchContent_MakeAvailable(libigl)

# PaStiX 6.x — the SPD solver SLIM uses inside slim.cpp. CHOLMOD plateaus at
# ~1.4× cores on SLIM-shaped sparse SPD systems; PaStiX's task scheduler hits
# 2.5–3× and runs ~8× faster end-to-end. Pulled via FetchContent because no
# distro currently packages PaStiX 6.
set(FLATBOI_PASTIX_TAG "v6.4.0" CACHE STRING "PaStiX tag to fetch")
FetchContent_Declare(pastix
    GIT_REPOSITORY https://gitlab.inria.fr/solverstack/pastix.git
    GIT_TAG        ${FLATBOI_PASTIX_TAG}
    GIT_SUBMODULES_RECURSE TRUE
)
set(PASTIX_INT64        OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_PARSEC  OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_STARPU  OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_MPI     OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_CUDA    OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS   ON  CACHE BOOL "" FORCE)
set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_FORTRAN OFF CACHE BOOL "" FORCE)

FetchContent_GetProperties(pastix)
if(NOT pastix_POPULATED)
    FetchContent_Populate(pastix)

    # PaStiX's RulesPrecisions.cmake keys its dictionary/precision config off
    # ${CMAKE_PROJECT_NAME}, which here is "flatboi" (the top-level project),
    # not "PASTIX". Without these the python codegen is invoked with -D ""
    # which fails silently — every generated *_spm.h / precision-templated
    # source ends up missing.
    set(RP_flatboi_DICTIONNARY
        "${pastix_SOURCE_DIR}/cmake_modules/local_subs.py"
        CACHE INTERNAL "")
    set(RP_flatboi_PRECISIONS "p;s;d;c;z" CACHE INTERNAL "")

    # PaStiX 6.4 upstream quirks when consumed as a subproject:
    # 1. Uses ${CMAKE_SOURCE_DIR} where it means ${CMAKE_CURRENT_SOURCE_DIR}.
    # 2. Adds test/ unconditionally; the precision-templated test sources
    #    don't get generated under FetchContent and the targets end up empty.
    execute_process(
        COMMAND sed -i
            -e "s|\${CMAKE_SOURCE_DIR}|\${CMAKE_CURRENT_SOURCE_DIR}|g"
            -e "/add_subdirectory(test)/d"
            -e "/enable_testing()/d"
            "${pastix_SOURCE_DIR}/CMakeLists.txt"
    )
    add_subdirectory(${pastix_SOURCE_DIR} ${pastix_BINARY_DIR})
endif()

add_executable(flatboi flatboi.cpp)

# SLIM_CACHED reuses the AᵀWA sparsity pattern across iters (only updates
# values). Combined with analyzePattern-once on the PaStiX side this lets
# every iter past the first skip both libigl-level and solver-level setup.
target_compile_definitions(flatboi PRIVATE SLIM_CACHED PASTIX6)
target_link_libraries(flatboi PRIVATE igl::core pastix spm)

if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(OpenMP)
endif()
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(flatboi PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(OpenCV REQUIRED COMPONENTS imgcodecs core)
target_link_libraries(flatboi PRIVATE ${OpenCV_LIBS})
