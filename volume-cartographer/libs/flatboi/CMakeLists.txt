# Buildable both standalone and as a subdirectory of the VC3D parent build.
# When standalone, set up the project; when consumed by the parent, inherit
# its C++ standard and project context.
if(NOT PROJECT_NAME)
    cmake_minimum_required(VERSION 3.22)
    project(flatboi LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 17)
endif()

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
set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_FORTRAN OFF CACHE BOOL "" FORCE)
# Build PaStiX as a static lib so flatboi doesn't need libpastix.so at runtime
# and so the static parent-tree archives (vc_core/utils) don't have to be PIC.
# Scope this to PaStiX's add_subdirectory; restore after so it doesn't leak
# to sibling subdirectories built later by the parent.
set(_flatboi_saved_bsl "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF)

FetchContent_GetProperties(pastix)
if(NOT pastix_POPULATED)
    FetchContent_Populate(pastix)

    # PaStiX's RulesPrecisions.cmake keys its dictionary/precision config off
    # ${CMAKE_PROJECT_NAME}, not "PASTIX". Without these the python codegen
    # is invoked with -D "" which fails silently — every generated *_spm.h
    # / precision-templated source ends up missing. Set under both the
    # standalone ("flatboi") and parent ("VC3D") project names so this works
    # whether flatboi is built standalone or as a VC3D subdirectory.
    set(_rp_var "RP_${CMAKE_PROJECT_NAME}_DICTIONNARY")
    set(${_rp_var}
        "${pastix_SOURCE_DIR}/cmake_modules/local_subs.py"
        CACHE INTERNAL "")
    set(_rp_prec "RP_${CMAKE_PROJECT_NAME}_PRECISIONS")
    set(${_rp_prec} "p;s;d;c;z" CACHE INTERNAL "")

    # PaStiX 6.4 upstream quirks when consumed as a subproject:
    # 1. Uses ${CMAKE_SOURCE_DIR} where it means ${CMAKE_CURRENT_SOURCE_DIR}.
    # 2. Adds test/, example/, wrappers/ unconditionally; we only want the
    #    libpastix/libspm libraries linked into flatboi. The bundled example
    #    binaries (analyze, compress, simple, ...) would otherwise land in
    #    the shared bin/ directory next to vc tools and get picked up by
    #    the Dockerfile's bulk copy.
    execute_process(
        COMMAND sed -i
            -e "s|\${CMAKE_SOURCE_DIR}|\${CMAKE_CURRENT_SOURCE_DIR}|g"
            -e "/add_subdirectory(test)/d"
            -e "/add_subdirectory(example)/d"
            -e "/add_subdirectory(wrappers)/d"
            -e "/enable_testing()/d"
            "${pastix_SOURCE_DIR}/CMakeLists.txt"
    )
    # spm (the matrix description sublib) ships its own tests/, examples/,
    # and wrappers/ — same reasoning as above, only the library is needed.
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
target_link_libraries(flatboi PRIVATE igl::core pastix spm)

if(NOT TARGET OpenMP::OpenMP_CXX)
    find_package(OpenMP)
endif()
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(flatboi PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(OpenCV REQUIRED COMPONENTS imgcodecs core)
target_link_libraries(flatboi PRIVATE ${OpenCV_LIBS})
