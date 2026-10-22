# volcodec_ng: lossless u8 volumetric codec (128^3 chunks, zarr v3 sharding_indexed
# friendly), next to volcomp. volcodec_ng_lossless.h is the lossless part of Chase
# Hendrick's volcodec-ng codec (MIT, LICENSE here; see README.md). volcodec_ng_lib.c
# compiles the header once and exports a plain C surface; see volcodec_ng_lib.h.
#
# Portable: the header needs no arch flags. On x86-64 it compiles AVX2 kernels with
# a target attribute and selects them at runtime; every target gets the same bytes.
add_library(volcodec_ng volcodec_ng_lib.c)
target_include_directories(volcodec_ng PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(MSVC)
    target_compile_options(volcodec_ng PRIVATE $<$<COMPILE_LANGUAGE:C>:/std:clatest> /w)
else()
    target_compile_features(volcodec_ng PRIVATE c_std_23)
    # Vendored code: silence its warnings under our curated set.
    target_compile_options(volcodec_ng PRIVATE -w)
endif()
# Strict IEEE math per AGENTS.md section 3.1: no -ffast-math. The decoder is
# integer only; the encoder's least-squares fit turns floating-point contraction
# off with a scoped pragma, so every build writes the same bytes.

if(NOT MSVC)
    target_link_libraries(volcodec_ng PRIVATE m)
endif()
