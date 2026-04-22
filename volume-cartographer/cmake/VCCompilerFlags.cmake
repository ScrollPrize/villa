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

# ---- Build type flags --------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    if(NOT APPLE)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -Wl,--compress-debug-sections=zlib")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "QuickBuild")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${VC_LTO_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS} ${VC_STRIP_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "ReleaseUnsafe")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${VC_LTO_FLAGS} ${VC_UNSAFE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LTO_FLAGS} ${VC_LINKER_FLAGS} ${VC_UNSAFE_LINKER_FLAGS} ${VC_THINLTO_CACHE_FLAGS} ${VC_STRIP_FLAGS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    # No LTO — incompatible with -gsplit-dwarf and makes debugging harder.
    if(APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -gsplit-dwarf")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS} -Wl,--compress-debug-sections=zlib")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    # -fno-omit-frame-pointer: keep frame pointers so `perf record --call-graph fp`
    # can unwind call stacks cheaply (DWARF unwinding is slow and often fails to
    # resolve symbols in heavily-templated code, leaving samples attributed to
    # the wrong function).
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Oz -g1 -fno-omit-frame-pointer ${VC_SIZE_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS} ${VC_SIZE_LINKER_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -g1 -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${VC_LINKER_FLAGS}")
    endif()
endif()
