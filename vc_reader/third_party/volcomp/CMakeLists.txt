# volcomp: single-header lossy u8 volumetric codec (128^3 chunks, zarr v3
# sharding_indexed friendly). Upstream: https://github.com/SuperOptimizer/volume-compressor
# (volcomp.h vendored verbatim; 1.3.0, format revision 4). volcomp_lib.c compiles the header once and
# exports a plain C surface; see volcomp_lib.h.
#
# Portable: the header needs no arch flags. On x86-64 it compiles its AVX2+FMA
# kernels with a target attribute and selects them at runtime when the CPU has
# them; arm64 and other targets use its plain C kernels.
add_library(vc_reader_volcomp STATIC volcomp_lib.c)
add_library(vc_reader::volcomp ALIAS vc_reader_volcomp)
set_target_properties(vc_reader_volcomp PROPERTIES EXPORT_NAME volcomp POSITION_INDEPENDENT_CODE ON)
target_include_directories(vc_reader_volcomp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/vc_reader/volcomp>)
if(MSVC)
    target_compile_options(vc_reader_volcomp PRIVATE $<$<COMPILE_LANGUAGE:C>:/std:clatest> /w)
else()
    target_compile_features(vc_reader_volcomp PRIVATE c_std_23)
    # Upstream vendored code: silence its warnings.
    target_compile_options(vc_reader_volcomp PRIVATE -w)
    target_link_libraries(vc_reader_volcomp PRIVATE m)
endif()
# Strict IEEE math: no -ffast-math. volcomp only asks for FMA contraction
# inside its transform kernels via a scoped pragma.
install(FILES volcomp_lib.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vc_reader/volcomp)
