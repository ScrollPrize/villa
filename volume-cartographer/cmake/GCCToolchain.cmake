# GCCToolchain.cmake - GCC/glibc optimization flags
#
# Warning policy lives in VCWarnings.cmake; this file is concerned only with
# code generation, LTO, and linker flags.

set(VC_LTO_FLAGS "-flto=auto -ffunction-sections -fdata-sections")
set(VC_LINKER_FLAGS "-Wl,--gc-sections -Wl,--as-needed")
set(VC_UNSAFE_CXX_FLAGS "-ffast-math -fno-finite-math-only -funroll-loops -ffp-contract=fast -fdevirtualize-at-ltrans")
set(VC_UNSAFE_LINKER_FLAGS "")
set(VC_THINLTO_CACHE_FLAGS "")
set(VC_STRIP_FLAGS "")
set(VC_SIZE_FLAGS "")
set(VC_SIZE_LINKER_FLAGS "")

message(STATUS "GCC toolchain loaded")
