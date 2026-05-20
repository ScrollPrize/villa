# GCCToolchain.cmake - GCC/glibc optimization flags

set(VC_GCC_SECTION_FLAGS "-ffunction-sections -fdata-sections")
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    set(_VC_ENABLE_GCC_LTO_DEFAULT OFF)
else()
    set(_VC_ENABLE_GCC_LTO_DEFAULT ON)
endif()
option(VC_ENABLE_GCC_LTO "Enable GCC link-time optimization for release builds" ${_VC_ENABLE_GCC_LTO_DEFAULT})

if(VC_ENABLE_GCC_LTO)
    set(VC_LTO_FLAGS "-flto=auto ${VC_GCC_SECTION_FLAGS}")
else()
    set(VC_LTO_FLAGS "${VC_GCC_SECTION_FLAGS}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
        message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION}: LTO disabled by default to avoid GCC 15 internal compiler errors")
    else()
        message(STATUS "GCC LTO disabled")
    endif()
endif()
if(APPLE)
    set(VC_LINKER_FLAGS "")
else()
    set(VC_LINKER_FLAGS "-Wl,--gc-sections -Wl,--as-needed")
endif()
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
