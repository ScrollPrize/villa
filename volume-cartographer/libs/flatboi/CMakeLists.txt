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
set(PASTIX_INT64           OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_PARSEC     OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_STARPU     OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_MPI        OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_CUDA       OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_FORTRAN    OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING          OFF CACHE BOOL "" FORCE)
# flatboi only ever solves double-precision real systems (Pastix6LLT<double>,
# SpmDouble). PaStiX otherwise auto-generates and compiles s/c/z variants of
# every precision-templated TU, ~5× the work.
set(PASTIX_PRECISIONS      "d" CACHE STRING "" FORCE)
set(SPM_PRECISIONS         "d" CACHE STRING "" FORCE)
# Build PaStiX as a static lib so flatboi doesn't need libpastix.so at runtime
# and so the static parent-tree archives don't have to be PIC. Scope to
# PaStiX's add_subdirectory; restore after so it doesn't leak to siblings.
set(_flatboi_saved_bsl "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF)

FetchContent_GetProperties(pastix)
if(NOT pastix_POPULATED)
    FetchContent_Populate(pastix)

    # PaStiX's RulesPrecisions.cmake keys its precision-codegen dictionary off
    # ${CMAKE_PROJECT_NAME}; without it the python codegen runs with -D "" and
    # silently produces missing *_spm.h / precision-templated sources.
    set(RP_${CMAKE_PROJECT_NAME}_DICTIONNARY
        "${pastix_SOURCE_DIR}/cmake_modules/local_subs.py"
        CACHE INTERNAL "")
    set(RP_${CMAKE_PROJECT_NAME}_PRECISIONS "d" CACHE INTERNAL "")

    # PaStiX 6.4 upstream quirks when consumed as a subproject:
    # 1. Uses ${CMAKE_SOURCE_DIR} where it means ${CMAKE_CURRENT_SOURCE_DIR}.
    # 2. Adds test/, example/, wrappers/ unconditionally; we only want the
    #    libpastix/libspm libraries linked into flatboi. Bundled example
    #    binaries (analyze, compress, simple, ...) would otherwise land in
    #    the shared bin/ next to vc tools and get picked up by the
    #    Dockerfile's bulk copy.
    execute_process(
        COMMAND sed -i
            -e "s|\${CMAKE_SOURCE_DIR}|\${CMAKE_CURRENT_SOURCE_DIR}|g"
            -e "/add_subdirectory(test)/d"
            -e "/add_subdirectory(example)/d"
            -e "/add_subdirectory(wrappers)/d"
            -e "/enable_testing()/d"
            "${pastix_SOURCE_DIR}/CMakeLists.txt"
    )
    execute_process(
        COMMAND sed -i
            -e "/add_subdirectory(tests)/d"
            -e "/add_subdirectory(examples)/d"
            -e "/add_subdirectory(wrappers)/d"
            "${pastix_SOURCE_DIR}/spm/CMakeLists.txt"
    )
    add_subdirectory(${pastix_SOURCE_DIR} ${pastix_BINARY_DIR})
endif()

set(BUILD_SHARED_LIBS "${_flatboi_saved_bsl}")
unset(_flatboi_saved_bsl)

add_executable(flatboi flatboi.cpp)

# flatboi has no Qt sources; opt out of parent-inherited AUTOMOC/UIC/RCC so
# CMake doesn't scan the PaStiX headers we pull in via FetchContent.
set_target_properties(flatboi PROPERTIES
    AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

# SLIM_CACHED reuses the AᵀWA sparsity pattern across iters (only updates
# values). Combined with analyzePattern-once on the PaStiX side this lets
# every iter past the first skip both libigl-level and solver-level setup.
target_compile_definitions(flatboi PRIVATE SLIM_CACHED PASTIX6)
target_link_libraries(flatboi PRIVATE igl::core pastix spm OpenMP::OpenMP_CXX ${OpenCV_LIBS})
