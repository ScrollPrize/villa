# GCCToolchain.cmake - GCC/glibc optimization flags

set(VC_LTO_FLAGS "-flto=auto -ffunction-sections -fdata-sections")
set(VC_LINKER_FLAGS "-Wl,--gc-sections -Wl,--as-needed")
set(VC_UNSAFE_CXX_FLAGS "-ffast-math -fno-finite-math-only -funroll-loops -ffp-contract=fast -fdevirtualize-at-ltrans")
set(VC_UNSAFE_LINKER_FLAGS "")
set(VC_THINLTO_CACHE_FLAGS "")
set(VC_STRIP_FLAGS "")
set(VC_SIZE_FLAGS "")
set(VC_SIZE_LINKER_FLAGS "")

message(STATUS "GCC toolchain loaded")

# Keep default GCC builds quiet from diagnostics that are noisy in our normal
# dependency/header mix or existing legacy code paths.
add_compile_options(
    -Wno-psabi
    -Wno-ignored-attributes
    -Wno-narrowing
)

# ---- Developer warnings ------------------------------------------------------
if(VC_DEVELOPER_WARNINGS)
    add_compile_options(
        -Wall -Wextra -pedantic
        -Wattributes -Wcast-align -Wcast-qual -Wchar-subscripts -Wcomment
        -Wconversion -Wdelete-incomplete -Wdelete-non-virtual-dtor
        -Wenum-compare -Wmain -Wmissing-field-initializers -Wmissing-noreturn
        -Wold-style-cast -Woverloaded-virtual -Wpointer-arith
        -Wtautological-compare -Wundef -Wuninitialized -Wunreachable-code
        -Wunused -Wvla -Wunused-parameter
        -Wlogical-op -Wduplicated-cond -Wduplicated-branches
        -Wnull-dereference -Wuseless-cast -Wsuggest-override
    )
    message(STATUS "Developer warnings enabled (GCC)")
endif()
