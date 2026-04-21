# c3d — 3D volumetric u8 compression codec (single TU, C23, libc only).
# Upstream: ~/c3d (vendored as a copy of c3d.c + c3d.h).
add_library(c3d STATIC c3d.c)

target_include_directories(c3d PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_features(c3d PRIVATE c_std_23)

# The repo-wide policy (AGENTS.md §3.1) is strict IEEE math — no
# -ffast-math / -Ofast. Keep it here too: c3d is a compression codec,
# and relaxing FP semantics across compilers/architectures would make
# encoded bytes non-reproducible between platforms. The upstream
# throughput targets were measured with -ffast-math but the reference
# decoder is fine without it.
target_compile_options(c3d PRIVATE
    -fno-math-errno
    -fno-trapping-math
    # Upstream vendored code — we don't own the style, silence the noise.
    -w
)

target_link_libraries(c3d PRIVATE m)

# -fPIC so vc_core (potentially built as a shared lib downstream) can link it.
set_target_properties(c3d PROPERTIES POSITION_INDEPENDENT_CODE ON)
