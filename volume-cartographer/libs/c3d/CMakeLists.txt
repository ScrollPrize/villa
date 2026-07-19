# c3d — 3D volumetric u8 compression codec (single TU, C23, libc only).
# Upstream: ~/c3d (vendored as a copy of c3d.c + c3d.h).
add_library(c3d c3d.c)

target_include_directories(c3d PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(MSVC)
    # CMake has no c_std_23 mapping for MSVC's C frontend; request the latest C
    # mode directly. Scope it to C sources via a genex so it is not applied to
    # the (C++) AUTOMOC mocs_compilation.cpp stub, which already gets
    # /std:c++latest and would otherwise trip D8016 (the two /std flags are
    # mutually exclusive).
    target_compile_options(c3d PRIVATE $<$<COMPILE_LANGUAGE:C>:/std:clatest>)
else()
    target_compile_features(c3d PRIVATE c_std_23)
endif()

# The repo-wide policy (AGENTS.md §3.1) is strict IEEE math — no
# -ffast-math / -Ofast. Keep it here too: c3d is a compression codec,
# and relaxing FP semantics across compilers/architectures would make
# encoded bytes non-reproducible between platforms. The upstream
# throughput targets were measured with -ffast-math but the reference
# decoder is fine without it.
if(MSVC)
    # /w silences the vendored code's warnings. MSVC keeps its default
    # (precise) FP model; -fno-math-errno / -fno-trapping-math are GCC/Clang
    # micro-optimizations, not correctness requirements, so they are dropped.
    target_compile_options(c3d PRIVATE /w)
else()
    target_compile_options(c3d PRIVATE
        -fno-math-errno
        -fno-trapping-math
        # Upstream vendored code — we don't own the style, silence the noise.
        -w
    )
endif()

# libm is a Unix-ism; on MSVC the math routines live in the CRT. (MinGW ships
# an empty libm, so linking -lm is harmless there — hence NOT MSVC, not
# NOT WIN32.)
if(NOT MSVC)
    target_link_libraries(c3d PRIVATE m)
endif()
