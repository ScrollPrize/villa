cmake_minimum_required(VERSION 3.22)
project(vc_spiral_bindings LANGUAGES CXX)

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
find_package(OpenMP QUIET COMPONENTS CXX)

function(vc_spiral_module target source output_name)
    nanobind_add_module(${target} NOMINSIZE ${source})
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${output_name}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/vc_spiral"
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    if(OpenMP_CXX_FOUND)
        # On MSVC, FindOpenMP selects /openmp: the OpenMP 2.0 runtime, vcomp.
        # The kernels use only 2.0 constructs (parallel for, +, || and the
        # by-hand min/max in resample_tracks), so they build and run in
        # parallel there too, and vcomp coexists with the libiomp5md that
        # torch's Windows wheels load. /openmp:llvm is deliberately not used:
        # it brings a second Intel-style runtime (OMP Error #15 at import)
        # and miscompiled atomic capture on MSVC 14.44 (#1633).
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MSVC)
        # MSVC's <cmath> only defines M_PI behind this macro.
        target_compile_definitions(${target} PRIVATE _USE_MATH_DEFINES)
    endif()
    install(TARGETS ${target}
        LIBRARY DESTINATION vc_spiral
        RUNTIME DESTINATION vc_spiral
    )
endfunction()

vc_spiral_module(vc_spiral_sampling spiral_sampling.cpp spiral_sampling)
vc_spiral_module(vc_spiral_track_crossings track_crossings.cpp track_crossings)
vc_spiral_module(vc_spiral_track_store track_store.cpp track_store)
vc_spiral_module(vc_spiral_surface_index surface_index.cpp surface_index)

target_compile_options(vc_spiral_track_crossings PRIVATE
    $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>>:-O3>
)

# Share the exact crossing kernel with vc_object_links; keep nanobind out of it.
set(VC_TRACK_CROSSINGS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../vc_object_store"
    CACHE PATH "vc_object_store checkout containing the shared track crossing kernel")
target_sources(vc_spiral_track_crossings PRIVATE
    "${VC_TRACK_CROSSINGS_ROOT}/src/TrackCrossings.cpp")
target_include_directories(vc_spiral_track_crossings PRIVATE
    "${VC_TRACK_CROSSINGS_ROOT}/include")
