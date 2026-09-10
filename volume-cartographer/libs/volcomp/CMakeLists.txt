# volcomp — single-header lossy u8 volumetric codec (128^3 chunks, zarr v3
# sharding_indexed friendly). Upstream: https://github.com/SuperOptimizer/volume-compressor
# (volcomp.h vendored verbatim). volcomp_lib.c compiles the header once and
# exports a plain C surface; see volcomp_lib.h.
#
# Portable: the header needs no arch flags. On x86-64 it compiles its AVX2+FMA
# kernels with a target attribute and selects them at runtime when the CPU has
# them; arm64 and other targets use its plain C kernels.
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
endif()
# Strict IEEE math per AGENTS.md §3.1: no -ffast-math. volcomp only asks for
# FMA contraction inside its transform kernels via a scoped pragma.

if(NOT MSVC)
    target_link_libraries(volcomp PRIVATE m)
endif()
