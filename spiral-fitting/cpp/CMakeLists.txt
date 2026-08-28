cmake_minimum_required(VERSION 3.22)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(vc_spiral_bindings LANGUAGES CXX)
    option(VC_SPIRAL_BUILD_PYTHON "Build the vc_spiral extension modules" ON)
    option(VC_SPIRAL_BUILD_TOOLS "Build native command-line tools" ON)
    find_package(TIFF REQUIRED)
    find_package(nlohmann_json REQUIRED)
    find_package(OpenMP QUIET COMPONENTS CXX)
    add_library(vc_spiral_surface_core STATIC
        ../spiral_graph/src/mapped_tifxyz.cpp
    )
    add_library(vc_spiral::surface_core ALIAS vc_spiral_surface_core)
    target_compile_features(vc_spiral_surface_core PUBLIC cxx_std_23)
    target_include_directories(vc_spiral_surface_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../spiral_graph/include"
    )
    target_link_libraries(vc_spiral_surface_core PUBLIC TIFF::TIFF)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(vc_spiral_surface_core PUBLIC OpenMP::OpenMP_CXX)
    endif()
    set_target_properties(vc_spiral_surface_core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

add_library(vc_spiral_patch_merger_core STATIC patch_merger.cpp)
add_library(vc_spiral::patch_merger_core ALIAS vc_spiral_patch_merger_core)
target_compile_features(vc_spiral_patch_merger_core PUBLIC cxx_std_23)
target_include_directories(vc_spiral_patch_merger_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(vc_spiral_patch_merger_core PUBLIC
    vc_spiral::surface_core
    TIFF::TIFF
    nlohmann_json::nlohmann_json
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vc_spiral_patch_merger_core PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(vc_spiral_patch_merger_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(VC_SPIRAL_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE VC_SPIRAL_NANOBIND_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(VC_SPIRAL_NANOBIND_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${VC_SPIRAL_NANOBIND_CMAKE_DIR}")
    endif()

    find_package(nanobind CONFIG REQUIRED)

    function(vc_spiral_module target source output_name)
        nanobind_add_module(${target} NOMINSIZE ${source})
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME "${output_name}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc_spiral"
        )
        target_compile_features(${target} PRIVATE cxx_std_23)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        install(TARGETS ${target}
            LIBRARY DESTINATION vc_spiral
            RUNTIME DESTINATION vc_spiral
        )
    endfunction()

    vc_spiral_module(vc_spiral_sampling spiral_sampling.cpp spiral_sampling)
    target_include_directories(vc_spiral_sampling PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../spiral_graph/include"
    )
    vc_spiral_module(vc_spiral_track_crossings track_crossings.cpp track_crossings)
    vc_spiral_module(vc_spiral_track_store track_store.cpp track_store)
    vc_spiral_module(vc_spiral_surface_index surface_index.cpp surface_index)
    vc_spiral_module(vc_spiral_patch_merger patch_merger_bindings.cpp patch_merger)
    target_link_libraries(vc_spiral_surface_index PRIVATE vc_spiral::surface_core)
    target_link_libraries(vc_spiral_patch_merger PRIVATE vc_spiral::patch_merger_core)

    target_compile_options(vc_spiral_track_crossings PRIVATE
        $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-O3>
    )
endif()

if(VC_SPIRAL_BUILD_TOOLS)
    add_executable(merge_overlapping_patches patch_merger_cli.cpp)
    target_compile_features(merge_overlapping_patches PRIVATE cxx_std_23)
    target_link_libraries(merge_overlapping_patches PRIVATE
        vc_spiral::patch_merger_core
    )
    set_target_properties(merge_overlapping_patches PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    install(TARGETS merge_overlapping_patches RUNTIME DESTINATION bin)
endif()
