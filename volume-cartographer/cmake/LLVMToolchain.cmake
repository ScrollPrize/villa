# LLVMToolchain.cmake - Clang/lld/ThinLTO optimization flags

set(VC_DEVIRT_FLAGS "-fstrict-vtable-pointers")
set(VC_DEVIRT_LTO_FLAGS "-fwhole-program-vtables")
set(VC_AGGRESSIVE_MATH "-ffast-math -fno-finite-math-only -funroll-loops -ffp-contract=fast")

# Unsafe flags: devirtualization + aggressive math (can break correctness)
set(VC_UNSAFE_CXX_FLAGS "${VC_DEVIRT_FLAGS} ${VC_DEVIRT_LTO_FLAGS} ${VC_AGGRESSIVE_MATH}")
if(APPLE)
    set(VC_VISIBILITY_FLAGS "-fvisibility=hidden -fvisibility-inlines-hidden")
else()
    set(VC_VISIBILITY_FLAGS "-fno-semantic-interposition -fvisibility=hidden -fvisibility-inlines-hidden")
endif()

# LLVM backend passes — aggressive, passed via linker for ThinLTO (ReleaseUnsafe only)
string(CONCAT VC_LLVM_LINKER_PASSES
    " -Wl,-mllvm,-inline-threshold=500"
    " -Wl,-mllvm,-inlinehint-threshold=600"
    " -Wl,-mllvm,-hot-callsite-threshold=500"
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
    " -Wl,-mllvm,-import-instr-limit=500"
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
if(NOT NPROC OR NPROC EQUAL 0)
    set(NPROC 4)
endif()

# Unsafe linker passes (Polly, aggressive inlining, LLVM backend opts)
set(VC_UNSAFE_LINKER_FLAGS "${VC_LLVM_LINKER_PASSES}${VC_ARCH_LINKER_PASSES}")

if(APPLE)
    find_program(LLD_LINKER ld.lld HINTS "/opt/homebrew/opt/llvm/bin")
    if(LLD_LINKER)
        set(VC_LTO_FLAGS "-flto=thin -fsplit-lto-unit -faddrsig -fmerge-all-constants -falign-functions=32 -falign-loops=16 -march=native ${VC_VISIBILITY_FLAGS}")
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
        set(VC_LTO_FLAGS "-march=native ${VC_VISIBILITY_FLAGS}")
        message(STATUS "Clang/macOS: no lld found, LTO disabled")
    endif()
    # Homebrew LLVM libc++
    if(EXISTS "/opt/homebrew/opt/llvm/lib/c++")
        set(VC_LIBCXX_FLAGS "-L/opt/homebrew/opt/llvm/lib/c++ -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++")
        message(STATUS "Using Homebrew libc++")
    endif()
else()
    # Linux Clang
    set(VC_LTO_FLAGS "-flto=thin -fsplit-lto-unit -faddrsig -fmerge-all-constants -falign-functions=32 -falign-loops=16 -march=native ${VC_VISIBILITY_FLAGS} -ffunction-sections -fdata-sections")
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
        # -Weverything with sensible exclusions
        -Weverything
        # Compatibility (we use C++23)
        -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-compat-local-type-template-args
        -Wno-c++98-compat-unnamed-type-template-args
        -Wno-c++98-compat-extra-semi
        -Wno-c++11-compat -Wno-c++14-compat -Wno-c++17-compat -Wno-c++20-compat
        -Wno-pre-c++14-compat -Wno-pre-c++17-compat
        -Wno-pre-c++20-compat -Wno-pre-c++23-compat -Wno-pre-c++26-compat
        # Noisy/unhelpful
        -Wno-padded
        -Wno-global-constructors -Wno-exit-time-destructors
        -Wno-covered-switch-default -Wno-switch-enum
        -Wno-float-equal
        -Wno-missing-prototypes -Wno-missing-variable-declarations
        -Wno-documentation -Wno-documentation-unknown-command
        -Wno-shadow -Wno-shadow-field -Wno-shadow-field-in-constructor
        -Wno-shadow-uncaptured-local
        -Wno-implicit-int-conversion -Wno-implicit-float-conversion
        -Wno-implicit-int-float-conversion -Wno-shorten-64-to-32
        -Wno-sign-conversion -Wno-double-promotion
        -Wno-disabled-macro-expansion -Wno-reserved-macro-identifier
        -Wno-reserved-identifier
        -Wno-newline-eof -Wno-extra-semi-stmt -Wno-extra-semi
        -Wno-unsafe-buffer-usage -Wno-unsafe-buffer-usage-in-libc-call
        -Wno-unsafe-buffer-usage-in-container
        -Wno-weak-vtables -Wno-packed -Wno-missing-braces
        -Wno-unused-template -Wno-unused-member-function
        -Wno-ctad-maybe-unsupported -Wno-undefined-reinterpret-cast
        -Wno-format-nonliteral
        -Wno-nested-anon-types -Wno-gnu-anonymous-struct
        -Wno-nan-infinity-disabled -Wno-switch-default
        -Wno-thread-safety-negative -Wno-nrvo -Wno-unused-lambda-capture
    )
    message(STATUS "Developer warnings enabled (-Weverything with sensible exclusions)")
endif()
