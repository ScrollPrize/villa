# VCCompilerFlags.cmake - Compiler flag orchestration

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -DWITH_BLOSC=1 -DWITH_ZLIB=1 -march=native -mcpu=native -pipe ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mcpu=native -pipe")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -DWITH_BLOSC=1 -DWITH_ZLIB=1 -march=native -pipe ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -pipe")
endif()
set(CMAKE_EXE_LINKER_FLAGS " ")

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# ---- Compiler-specific flags -------------------------------------------------
set(VC_LIBCXX_FLAGS "")
set(VC_LTO_FLAGS "")
set(VC_LINKER_FLAGS "")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    include(LLVMToolchain)
else()
    include(GCCToolchain)
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LIBCXX_FLAGS}")

# ---- Debug info: applied to ALL build types so the runtime crash handler -----
# (libbacktrace) can resolve file:line, function names, and inline frames in
# user-submitted bug reports.
#   -g3                          : full DWARF including macros and types
#   -fno-omit-frame-pointer      : keep frame pointers for accurate unwinding
#   -fasynchronous-unwind-tables : .eh_frame for backtrace from any PC
#   -fno-eliminate-unused-debug-types (clang -fstandalone-debug equivalent on
#                                      gcc) : keep debug info for forward-decl'd
#                                      types that may appear in stack frames
set(VC_DEBUG_INFO_FLAGS "-g3 -fno-omit-frame-pointer -fasynchronous-unwind-tables")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(VC_DEBUG_INFO_FLAGS "${VC_DEBUG_INFO_FLAGS} -fstandalone-debug")
else()
    set(VC_DEBUG_INFO_FLAGS "${VC_DEBUG_INFO_FLAGS} -fno-eliminate-unused-debug-types")
endif()

# ---- Build type flags --------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 ${VC_DEBUG_INFO_FLAGS}")
    if(NOT APPLE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -Wl,--compress-debug-sections=zlib")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "QuickBuild")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 ${VC_DEBUG_INFO_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS}")
    # QuickBuild: no LTO, enable PCH for fast iteration
    set(VC_USE_PCH ON CACHE BOOL "Enable precompiled headers (faster builds)" FORCE)
    message(STATUS "QuickBuild: PCH enabled, LTO disabled")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${VC_DEBUG_INFO_FLAGS} ${VC_LTO_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS} ${VC_STRIP_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "ReleaseUnsafe")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${VC_DEBUG_INFO_FLAGS} ${VC_LTO_FLAGS} ${VC_UNSAFE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_UNSAFE_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # No LTO — incompatible with -gsplit-dwarf and makes debugging harder.
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 ${VC_DEBUG_INFO_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 ${VC_DEBUG_INFO_FLAGS} -gsplit-dwarf")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS} -Wl,--compress-debug-sections=zlib")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Oz ${VC_DEBUG_INFO_FLAGS} ${VC_SIZE_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS} ${VC_SIZE_LINKER_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os ${VC_DEBUG_INFO_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS}")
    endif()
endif()

# ---- Status summary ----------------------------------------------------------
if(VC_USE_CCACHE AND CCACHE_PROGRAM AND VC_USE_PCH)
    message(STATUS "Using ccache + PCH")
elseif(VC_USE_CCACHE AND CCACHE_PROGRAM)
    message(STATUS "Using ccache")
elseif(VC_USE_PCH)
    message(STATUS "Using PCH (no ccache)")
endif()
