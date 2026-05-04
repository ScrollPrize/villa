# LLVMToolchain.cmake - Clang/lld/ThinLTO optimization flags

set(VC_DEVIRT_FLAGS "-fstrict-vtable-pointers")
set(VC_DEVIRT_LTO_FLAGS "-fwhole-program-vtables")
set(VC_AGGRESSIVE_MATH "-ffast-math -fno-finite-math-only -funroll-loops -ffp-contract=fast")

set(VC_EXTRA_PERF_FLAGS "-fno-plt -fno-math-errno -fomit-frame-pointer")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    if(VC_NATIVE_ARCH)
        set(VC_EXTRA_PERF_FLAGS "${VC_EXTRA_PERF_FLAGS} -mcpu=native")
    endif()
    set(VC_EXTRA_PERF_FLAGS "${VC_EXTRA_PERF_FLAGS} -mno-outline-atomics")
endif()

# Unsafe flags: devirtualization + aggressive math (can break correctness)
set(VC_UNSAFE_CXX_FLAGS "${VC_DEVIRT_FLAGS} ${VC_DEVIRT_LTO_FLAGS} ${VC_AGGRESSIVE_MATH} ${VC_EXTRA_PERF_FLAGS}")
if(APPLE)
    set(VC_VISIBILITY_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden")
else()
    set(VC_VISIBILITY_FLAGS "-fno-semantic-interposition -fvisibility=hidden -fvisibility-inlines-hidden")
endif()

string(CONCAT VC_LLVM_LINKER_PASSES
    " -Wl,-mllvm,-inline-threshold=1000"
    " -Wl,-mllvm,-inlinehint-threshold=1200"
    " -Wl,-mllvm,-hot-callsite-threshold=1000"
    " -Wl,-mllvm,-polly"
    " -Wl,-mllvm,-polly-vectorizer=stripmine"
    " -Wl,-mllvm,-polly-tiling"
    " -Wl,-mllvm,-polly-2nd-level-tiling"
    " -Wl,-mllvm,-polly-register-tiling"
    " -Wl,-mllvm,-enable-loopinterchange"
    " -Wl,-mllvm,-enable-interleaved-mem-accesses"
    " -Wl,-mllvm,-enable-masked-interleaved-mem-accesses"
    " -Wl,-mllvm,-hot-cold-split"
    " -Wl,-mllvm,-enable-ext-tsp-block-placement"
    " -Wl,-mllvm,-import-instr-limit=1000"
    " -Wl,-O3"
    " -Wl,--icf=all"
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    string(CONCAT VC_ARCH_LINKER_PASSES
        " -Wl,-mllvm,-aarch64-use-aa"
        " -Wl,-mllvm,-aarch64-enable-global-merge"
    )
else()
    set(VC_ARCH_LINKER_PASSES "")
endif()

set(VC_SIZE_FLAGS "-fno-ident")
if(APPLE)
    set(VC_SIZE_LINKER_FLAGS "")
else()
    set(VC_SIZE_LINKER_FLAGS "-Wl,-mllvm,-enable-machine-outliner")
endif()

include(ProcessorCount)
ProcessorCount(NPROC)
# Fallback: ProcessorCount() returns 1 in some cpuset-limited containers even
# when the host has many cores; cmake_host_system_information sees the kernel's
# count.
cmake_host_system_information(RESULT NPROC_LOGICAL QUERY NUMBER_OF_LOGICAL_CORES)
if(NPROC_LOGICAL AND NPROC_LOGICAL GREATER NPROC)
    set(NPROC ${NPROC_LOGICAL})
endif()
if(NOT NPROC OR NPROC EQUAL 0)
    set(NPROC 4)
endif()

set(VC_UNSAFE_LINKER_FLAGS "${VC_LLVM_LINKER_PASSES}${VC_ARCH_LINKER_PASSES}")

if(APPLE)
    find_program(LLD_LINKER ld.lld HINTS "/opt/homebrew/opt/llvm/bin")
    if(LLD_LINKER)
        set(VC_LTO_FLAGS "-flto=thin -fsplit-lto-unit -faddrsig -fmerge-all-constants -falign-functions=32 -falign-loops=16 ${VC_VISIBILITY_FLAGS}")
        string(CONCAT VC_LINKER_FLAGS
            "-fuse-ld=lld"
            " -Wl,-dead_strip"
            " -Wl,--icf=all"
            " -Wl,--deduplicate-strings"
            " -Wl,--lto-O3"
            " -Wl,--lto-CGO3"
            " -Wl,--thinlto-jobs=${NPROC}"
            " -Wl,--call-graph-profile-sort"
        )
        message(STATUS "Clang/macOS: ThinLTO + lld (${NPROC} jobs)")
    else()
        set(VC_LTO_FLAGS "${VC_VISIBILITY_FLAGS}")
        message(STATUS "Clang/macOS: no lld found, LTO disabled")
    endif()
    # Homebrew LLVM libc++
    if(EXISTS "/opt/homebrew/opt/llvm/lib/c++")
        set(VC_LIBCXX_FLAGS "-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++")
        message(STATUS "Using Homebrew libc++")
    endif()
else()
    # Linux Clang
    set(VC_LTO_FLAGS "-flto=thin -fsplit-lto-unit -faddrsig -fmerge-all-constants -falign-functions=32 -falign-loops=16 ${VC_VISIBILITY_FLAGS} -ffunction-sections -fdata-sections")
    string(CONCAT VC_LINKER_FLAGS
        "-fuse-ld=lld"
        " -Wl,--icf=all"
        " -Wl,--gc-sections"
        " -Wl,--as-needed"
        " -Wl,-O2"
        " -Wl,--hash-style=gnu"
        " -Wl,--lto-O3"
        " -Wl,--lto-CGO3"
        " -Wl,--thinlto-jobs=${NPROC}"
        " -Wl,--call-graph-profile-sort=hfsort"
    )
    message(STATUS "Clang/Linux: ThinLTO + lld (${NPROC} jobs)")
endif()

# ThinLTO cache
set(VC_THINLTO_CACHE_DIR "${CMAKE_BINARY_DIR}/lto-cache")
file(MAKE_DIRECTORY ${VC_THINLTO_CACHE_DIR})
if(APPLE)
    set(VC_THINLTO_CACHE_FLAGS "-Wl,-cache_path_lto,${VC_THINLTO_CACHE_DIR} -Wl,--thinlto-cache-policy=cache_size_bytes=1g")
    set(VC_STRIP_FLAGS "-Wl,-S")
else()
    set(VC_THINLTO_CACHE_FLAGS "-Wl,--thinlto-cache-dir=${VC_THINLTO_CACHE_DIR} -Wl,--thinlto-cache-policy=cache_size_bytes=1g")
    set(VC_STRIP_FLAGS "-Wl,--strip-all")
endif()
if(NOT VC_STRIP_BINARIES)
    set(VC_STRIP_FLAGS "")
endif()

if(VC_VECTORIZATION_REPORT)
    add_compile_options(
        -Rpass=loop-vectorize
        -Rpass-missed=loop-vectorize
        -Rpass-analysis=loop-vectorize
        -Rpass=slp-vectorize
        -Rpass-missed=slp-vectorize
        -fsave-optimization-record
    )
    add_link_options(
        -Wl,-mllvm,-pass-remarks=loop-vectorize
        -Wl,-mllvm,-pass-remarks-missed=loop-vectorize
        -Wl,-mllvm,-pass-remarks-analysis=loop-vectorize
        -Wl,-mllvm,-pass-remarks=slp-vectorize
        -Wl,-mllvm,-pass-remarks-missed=slp-vectorize
    )
    message(STATUS "Clang vectorization remarks enabled (compile + LTO link)")
endif()
