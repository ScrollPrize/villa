# volcomp — single-header lossy u8 volumetric codec (128^3 chunks, zarr v3
# sharding_indexed friendly). Upstream: https://github.com/SuperOptimizer/volume-compressor
# (volcomp.h vendored verbatim). volcomp_lib.c compiles the header once and
# exports a plain C surface; see volcomp_lib.h.
#
# The header's kernels are explicit AVX2+FMA, so the codec is built only on
# x86-64 with GCC/Clang. Elsewhere the same target still exists and every
# entry point returns VOLCOMP_LIB_UNSUPPORTED, so consumers link
# unconditionally and check volcomp_lib_available() at runtime (which also
# covers x86-64 CPUs without AVX2).
add_library(volcomp volcomp_lib.c)
target_include_directories(volcomp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(MSVC)
    target_compile_options(volcomp PRIVATE $<$<COMPILE_LANGUAGE:C>:/std:clatest> /w)
else()
    target_compile_features(volcomp PRIVATE c_std_23)
    # Upstream vendored code — silence its warnings under our curated set.
    target_compile_options(volcomp PRIVATE -w)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
        target_compile_options(volcomp PRIVATE -mavx2 -mfma)
    endif()
endif()
# Strict IEEE math per AGENTS.md §3.1: no -ffast-math. volcomp only asks for
# FMA contraction inside its transform kernels via a scoped pragma.

if(NOT MSVC)
    target_link_libraries(volcomp PRIVATE m)
endif()
