cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(VC3D VERSION 1.0.0)

# We don't use C++20/23 modules. CMake 3.28+ would otherwise invoke
# clang-scan-deps on every C++ source to discover module imports, which
# requires the `clang-tools-N` package that isn't part of Ubuntu 24.04's
# default `clang` install. Skip the scan globally.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Shared libs to shrink binaries and CI disk usage. (flatboi forces this
# OFF locally for PaStiX.)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build vc_* and FetchContent deps as .so")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# GCC/MinGW export every symbol from a shared library by default, which is what
# the vc_* .so/.dll split relies on. MSVC exports nothing unless asked, so a
# shared lib with no __declspec(dllexport) yields no import .lib and downstream
# links fail (LNK1181). Auto-generate the export table for every shared target
# to match the GCC behavior the codebase assumes.
if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
# Exes default to -fPIE; force -fPIC to match the shared libs' PCH. -no-pie
# only on exe links (mold/lld reject it on a .so). PE/COFF has no PIC/PIE
# distinction, so skip both on Windows.
if(NOT WIN32)
    add_compile_options(-fPIC)
    add_link_options($<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-no-pie>)
endif()
# Find the .so via $ORIGIN without LD_LIBRARY_PATH.
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
if(APPLE)
    # Packaged layout (cmake/MacPackaging.cmake): executables live in
    # VC3D.app/Contents/MacOS, every dylib in ../Frameworks. Mach-O has no
    # $ORIGIN; @loader_path is its equivalent and covers both the executables
    # and dylib->dylib references (Frameworks/../Frameworks resolves to
    # itself).
    set(CMAKE_INSTALL_RPATH "@loader_path/../Frameworks")
else()
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)

option(VC_MARCH_NATIVE     "Tune for the host CPU (-march=native)"        OFF)
option(VC_WARNINGS        "Enable curated warning set"                   OFF)
option(VC_WERROR          "Treat compiler warnings as errors"            OFF)
option(VC_ENABLE_ASAN     "AddressSanitizer"                             OFF)
option(VC_ENABLE_UBSAN    "UndefinedBehaviorSanitizer"                   OFF)
option(VC_ENABLE_TSAN     "ThreadSanitizer"                              OFF)
option(VC_ENABLE_LSAN     "LeakSanitizer (standalone)"                   OFF)
option(VC_ENABLE_TYSAN    "TypeSanitizer (clang-only, LLVM 18+)"         OFF)
option(VC_ENABLE_NSAN     "NumericalStabilitySanitizer (clang-only, LLVM 19+)" OFF)
option(VC_ENABLE_COVERAGE "Code coverage instrumentation"                OFF)
option(VC_TESTING         "Build and run unit tests"                     OFF)
option(VC_DEAD_CODE_REPORT "Enable dead-code reporting (curated -Wunused* + linker --print-gc-sections; use a Debug build to avoid LTO)" OFF)
option(VC_PROFILE_COMPILE  "Emit per-TU compile-time profiles (-ftime-trace on clang, -ftime-report on gcc)" OFF)
# flatboi drags in PaStiX (Fortran + autotools + Scotch), which has no
# practical Windows build — default it off there.
if(WIN32)
    option(VC_BUILD_FLATBOI "Build the flatboi SLIM flattening tool"      OFF)
else()
    option(VC_BUILD_FLATBOI "Build the flatboi SLIM flattening tool"      ON)
endif()
option(VC_BUILD_PYTHON     "Build Python bindings"                       OFF)
option(VC_USE_SYSTEM_LIBBACKTRACE "Prefer installed libbacktrace when available" ON)
option(VC_LIBBACKTRACE_UPDATE_DISCONNECTED
    "Avoid network update checks for vendored libbacktrace during ordinary builds"
    ON)
option(VC_USE_PCH "Precompile common headers into neutral carrier targets; downstream targets REUSE_FROM them" OFF)
option(VC_USE_CCACHE "Route the compiler through ccache" OFF)
option(VC_USE_SCCACHE "Route the compiler through sccache" OFF)
option(VC_ENABLE_ECL_MAXFLOW "Build CUDA ECL-MaxFlow integration for Lasagna maxflow" OFF)
option(VC_ENABLE_AMGX "Build external AMGX-backed Lasagna screened-Laplace solver" OFF)
set(AMGX_ROOT "" CACHE PATH "External AMGX installation prefix")
set(VC_CCACHE_PCH_SLOPPINESS "pch_defines,time_macros" CACHE STRING
    "ccache sloppiness required when VC_USE_PCH and VC_USE_CCACHE are both ON")

if(VC_ENABLE_ECL_MAXFLOW)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "VC_ENABLE_ECL_MAXFLOW=ON requires a CUDA compiler")
    endif()
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures for ECL-MaxFlow" FORCE)
    endif()
endif()

set(_vc_build_types Debug Release RelWithDebInfo MinSizeRel QuickBuild)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type." FORCE)
endif()
if(DEFINED CMAKE_BUILD_TYPE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${_vc_build_types})
endif()

# Fast local iteration: keep some optimization so optimized code paths still
# compile, but skip LTO and force the shared vc_core PCH back on.
if(CMAKE_BUILD_TYPE STREQUAL "QuickBuild")
    set(VC_USE_PCH ON CACHE BOOL "Precompile common headers into neutral carrier targets; downstream targets REUSE_FROM them" FORCE)
    message(STATUS "QuickBuild: -O1, PCH enabled, LTO disabled")
endif()

if(VC_USE_CCACHE AND VC_USE_SCCACHE)
    message(FATAL_ERROR "VC_USE_CCACHE and VC_USE_SCCACHE are mutually exclusive.")
endif()

if(VC_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(NOT CCACHE_PROGRAM)
        message(FATAL_ERROR "VC_USE_CCACHE=ON but ccache was not found on PATH.")
    endif()
    if(VC_USE_PCH)
        set(_vc_ccache_launcher
            "${CMAKE_COMMAND};-E;env;CCACHE_SLOPPINESS=${VC_CCACHE_PCH_SLOPPINESS};${CCACHE_PROGRAM}")
        message(STATUS "ccache with PCH: CCACHE_SLOPPINESS=${VC_CCACHE_PCH_SLOPPINESS}")
    else()
        set(_vc_ccache_launcher "${CCACHE_PROGRAM}")
    endif()
    set(CMAKE_C_COMPILER_LAUNCHER "${_vc_ccache_launcher}" CACHE STRING "Compiler launcher for C" FORCE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${_vc_ccache_launcher}" CACHE STRING "Compiler launcher for CXX" FORCE)
endif()

if(VC_USE_SCCACHE)
    find_program(SCCACHE_PROGRAM sccache)
    if(NOT SCCACHE_PROGRAM)
        message(FATAL_ERROR "VC_USE_SCCACHE=ON but sccache was not found on PATH.")
    endif()
    set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "Compiler launcher for C" FORCE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "Compiler launcher for CXX" FORCE)
    message(STATUS "Compiler cache: sccache (${SCCACHE_PROGRAM})")
endif()

# Skip building app executables (vc_*, VC3D) when generating coverage:
# the apps add ~3 GB of debug+coverage-instrumented binaries to the
# build dir and aren't exercised by the unit tests, so they only inflate
# the build without contributing coverage data. The static libs the
# tests need (e.g. vc_merge_grid) are still built unconditionally.
# Override with -DVC_BUILD_APPS=ON if you need the apps in a coverage
# build (e.g. for end-to-end tests that spawn the binaries).
if(VC_ENABLE_COVERAGE)
    option(VC_BUILD_APPS  "Build app executables (vc_*, VC3D)"           OFF)
else()
    option(VC_BUILD_APPS  "Build app executables (vc_*, VC3D)"           ON)
endif()
option(VC_BUILD_UI_TRACER  "Build Qt UI/tracer helper libraries"         ${VC_BUILD_APPS})

# ---- Toolchain --------------------------------------------------------------
foreach(_path "$ENV{HOME}/miniconda3" "$ENV{HOME}/anaconda3")
    if(EXISTS "${_path}")
        list(APPEND CMAKE_IGNORE_PREFIX_PATH "${_path}")
    endif()
endforeach()

# ---- Git revision (used for ProjectInfo::VersionString) ---------------------
# Query git in the source tree. Build contexts with no .git (e.g. the Docker
# AppImage build, whose context is only the volume-cartographer/ subtree) can
# inject the revision via -DVC_GIT_SHA1=/-DVC_GIT_COMMIT_DATE=.
set(VC_GIT_SHA1 "" CACHE STRING "Full commit SHA fallback when the tree has no .git")
set(VC_GIT_COMMIT_DATE "" CACHE STRING "Commit date (YYYY-MM-DD) paired with VC_GIT_SHA1")
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${CMAKE_SOURCE_DIR}" rev-parse HEAD
        OUTPUT_VARIABLE GIT_SHA1 OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${CMAKE_SOURCE_DIR}" log -1 --format=%cs HEAD
        OUTPUT_VARIABLE GIT_COMMIT_DATE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
# The git query wins whenever it succeeds, so a stale cached VC_GIT_SHA1 can
# never mask the real revision in a normal checkout; the injected value is used
# only when the tree has no .git (or git is unavailable).
if(NOT GIT_SHA1)
    set(GIT_SHA1 "${VC_GIT_SHA1}")
    set(GIT_COMMIT_DATE "${VC_GIT_COMMIT_DATE}")
endif()
if(GIT_SHA1)
    string(SUBSTRING ${GIT_SHA1} 0 7 GIT_SHA1_SHORT)
else()
    set(GIT_SHA1 Untracked)
    set(GIT_SHA1_SHORT Untracked)
    set(GIT_COMMIT_DATE Untracked)
endif()
set(VC_VERSION_STRING "${GIT_SHA1_SHORT}-${GIT_COMMIT_DATE}")

# ---- Base + build-type flags ------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(VC_DEAD_CODE_REPORT)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()
if(VC_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Portable shippable defaults; VC_MARCH_NATIVE=ON (set by dev-* presets) opts
# into host tuning. Windows installers use the general x86-64 baseline so they
# run on all supported amd64 CPUs; Linux retains the existing x86-64-v3 floor.
# armv8.2-a avoids SVE so Apple M1-M3 + Qualcomm Oryon work alongside Graviton.
if(MSVC)
    # MSVC has no -march/-pipe. It targets the portable x86-64 (SSE2) baseline
    # by default, matching the Windows floor used on the GCC path below;
    # VC_MARCH_NATIVE opts into AVX2 host tuning.
    if(VC_MARCH_NATIVE)
        add_compile_options(/arch:AVX2)
    endif()
elseif(VC_MARCH_NATIVE)
    add_compile_options(-pipe -march=native)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    if(WIN32)
        add_compile_options(-pipe -march=x86-64)
    else()
        add_compile_options(-pipe -march=x86-64-v3)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_compile_options(-pipe -march=armv8.2-a)
else()
    add_compile_options(-pipe)
endif()

if(MSVC)
    # MSVC toolchain equivalents of the GCC/Clang setup below:
    #   /Zi   PDB debug info (moral equivalent of -g3; kept in every config so
    #         release crash dumps still symbolize)
    #   /Gy /Gw  function/data-level linking so /OPT:REF,ICF can strip unused
    #         code (~ -ffunction-sections/-fdata-sections + --gc-sections)
    #   /bigobj  lift the COFF 32k-section limit (~ -Wa,-mbig-obj)
    #   /permissive- /Zc:__cplusplus /Zc:preprocessor  standards conformance
    #   /utf-8   treat sources as UTF-8 (the GCC/Clang default)
    #   /MP      parallel TU compilation
    #   /wd4068  ignore the unknown `#pragma omp` (OpenMP is stubbed, see below)
    set(_dbg "")
    set(_lto "")
    set(_link "")
    add_compile_options(/Zi /Gy /Gw /bigobj /permissive- /Zc:__cplusplus
                        /Zc:preprocessor /utf-8 /MP /wd4068)
    add_link_options(/DEBUG /OPT:REF /OPT:ICF)
    # Windows.h hygiene + the libm-style math constants (M_PI, ...) that
    # MinGW/GCC expose by default but MSVC gates behind these macros.
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES
                            _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_dbg -g3 -fno-omit-frame-pointer -fasynchronous-unwind-tables -fstandalone-debug)
    set(_lto -flto=thin)
    if(APPLE)
        set(_link "")
        # ld64.lld (the LLVM Mach-O linker the macOS preset selects via
        # -fuse-ld=lld) mishandles C++ exception unwind info for our large
        # ThinLTO shared libraries: a `throw` inside e.g. libvc_core that should
        # be caught in an executable instead unwinds past every handler into
        # std::terminate (SIGABRT) once it crosses the dylib boundary. The
        # failure is in how lld lays out the dylib's unwind tables — relinking
        # the *same* objects with Apple's ld makes cross-dylib exceptions
        # propagate correctly, and no ld64.lld flag (-keep_dwarf_unwind,
        # -no_compact_unwind, -dead_strip) reliably fixes it for a library this
        # size. So on macOS we link shared libraries with Apple's system ld
        # (always available via the SDK) while keeping lld for executables.
        # -fuse-ld= with an empty value selects the driver default (Apple ld).
        # A later -fuse-ld= (empty) on the link line overrides the preset's
        # earlier -fuse-ld=lld for shared-library targets only; clang honors the
        # last one. Applied per-target via a genex so it's idempotent across
        # reconfigures and leaves executables on lld.
        if(CMAKE_EXE_LINKER_FLAGS MATCHES "lld" OR CMAKE_SHARED_LINKER_FLAGS MATCHES "lld")
            add_link_options(
                "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:-fuse-ld=>")
        endif()
    else()
        set(_link -fuse-ld=lld -Wl,--gc-sections -Wl,--as-needed)
    endif()
else()
    set(_dbg -g3 -fno-omit-frame-pointer -fasynchronous-unwind-tables -fno-eliminate-unused-debug-types)
    set(_lto -flto=auto)
    if(APPLE)
        set(_link "")
    elseif(WIN32)
        # binutils PE linker: --gc-sections works, mold/lld are not options.
        set(_link -Wl,--gc-sections -Wl,--as-needed)
    else()
        find_program(VC_MOLD_BINARY mold)
        if(VC_MOLD_BINARY)
            set(_link -fuse-ld=mold -Wl,--gc-sections -Wl,--as-needed)
        else()
            set(_link -Wl,--gc-sections -Wl,--as-needed)
        endif()
    endif()
endif()
if(MSVC)
    add_compile_options(${_dbg})
else()
    add_compile_options(${_dbg} -ffunction-sections -fdata-sections)
endif()
add_link_options(${_link})

if(WIN32 AND NOT MSVC)
    # Several large TUs (Qt-heavy VC3D sources, template-dense core) exceed
    # the default COFF 32k-section limit; -mbig-obj lifts it for the binutils
    # assembler. (MSVC gets /bigobj in the compiler-family block above.)
    add_compile_options(-Wa,-mbig-obj)
endif()

# ---- Sanitizers + coverage --------------------------------------------------
if(VC_ENABLE_ASAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "ASan + TSan are mutually exclusive")
endif()
if(VC_ENABLE_LSAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "LSan + TSan are mutually exclusive")
endif()
if(VC_ENABLE_TYSAN AND (VC_ENABLE_ASAN OR VC_ENABLE_TSAN OR VC_ENABLE_NSAN))
    message(FATAL_ERROR "TySan + (ASan/TSan/NSan) are mutually exclusive")
endif()
if(VC_ENABLE_NSAN AND (VC_ENABLE_ASAN OR VC_ENABLE_TSAN))
    message(FATAL_ERROR "NSan + (ASan or TSan) are mutually exclusive")
endif()
if((VC_ENABLE_TYSAN OR VC_ENABLE_NSAN) AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "TySan / NSan require clang")
endif()

if(VC_ENABLE_ASAN)
    list(APPEND _san -fsanitize=address,leak -DADDRESS_SANITIZER)
endif()
if(VC_ENABLE_UBSAN)
    list(APPEND _san -fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND _san -fsanitize=nullability)
    endif()
endif()
if(VC_ENABLE_TSAN)
    list(APPEND _san -fsanitize=thread)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
if(VC_ENABLE_TYSAN)
    list(APPEND _san -fsanitize=type)
endif()
if(VC_ENABLE_NSAN)
    list(APPEND _san -fsanitize=numerical)
endif()
if(VC_ENABLE_LSAN AND NOT VC_ENABLE_ASAN)
    list(APPEND _san -fsanitize=leak)
endif()
# Sanitizers default to "recover and continue" on most checks, which lets
# CI silently pass on real UB / type confusion / numerical issues. Force
# abort on first detection across every enabled sanitizer.
if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN
   OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN OR VC_ENABLE_LSAN)
    list(APPEND _san -fno-sanitize-recover=all)
endif()
if(VC_ENABLE_COVERAGE)
    # clang source-based coverage only: accurate at any -O level, reads
    # .profraw + the binary's coverage-mapping section. gcc/gcov was retired
    # (only accurate at -O0). -g1 keeps line tables for useful backtraces.
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "VC_ENABLE_COVERAGE requires clang — gcc/gcov was retired (only accurate at -O0). Use the ci-coverage-clang preset.")
    endif()
    list(APPEND _san -fprofile-instr-generate -fcoverage-mapping)
    list(APPEND _san -g1)
endif()
# Sanitizer/coverage flags applied below, after libs/ — see that block.

# ---- Optimization + LTO -----------------------------------------------------
# GCC/Clang optimization levels. MSVC takes its optimization from CMake's
# per-config flags (/O2 for Release/RelWithDebInfo, /Od for Debug), so it is
# intentionally left out of this block.
if(NOT MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0)
    elseif(CMAKE_BUILD_TYPE STREQUAL "QuickBuild")
        add_compile_options(-O1)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3)
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        add_compile_options(-O2)
    elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
        add_compile_options(-Os)
    endif()
endif()
# LTO stays off on Windows: GCC LTO across DLLs interacts badly with PE
# auto-import/export-all, and link times on the CI runners balloon.
if(NOT _san AND NOT WIN32 AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT CMAKE_BUILD_TYPE STREQUAL "QuickBuild")
    add_compile_options(${_lto})
    add_link_options(${_lto})
endif()

# ---- PIC for python bindings -----------------------------------------------------
# Python bindings link the static libs (vc_core, utils, ...) into .so modules,
# which requires every translation unit to be -fPIC (otherwise thread_local
# statics emit local-exec TPOFF32 relocations that ld rejects).
if(VC_BUILD_PYTHON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# ---- Warnings ---------------------------------------------------------------
if(MSVC)
    # Highest-volume MSVC-only warnings that GCC/Clang never emit for this
    # code: narrowing/precision conversions, signed/unsigned compares, and the
    # CRT "deprecation" of standard/POSIX names. Noise, not bugs.
    add_compile_options(/wd4244 /wd4267 /wd4305 /wd4018 /wd4146 /wd4996)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-unknown-warning-option)
else()
    add_compile_options(-Wno-psabi -Wno-ignored-attributes -Wno-narrowing)
endif()

if(VC_WARNINGS AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
        -Weverything
        -Wno-padded
        -Wno-unsafe-buffer-usage -Wno-unsafe-buffer-usage-in-libc-call -Wno-unsafe-buffer-usage-in-container
        -Wno-ctad-maybe-unsupported -Wno-disabled-macro-expansion
        -Wno-exit-time-destructors -Wno-global-constructors
        -Wno-newline-eof -Wno-extra-semi -Wno-extra-semi-stmt
        -Wno-weak-vtables -Wno-packed -Wno-nrvo
        -Wno-missing-prototypes -Wno-missing-variable-declarations
        -Wno-reserved-identifier -Wno-reserved-macro-identifier
        -Wno-covered-switch-default -Wno-switch-enum -Wno-switch-default
        -Wno-source-uses-openmp -Wno-old-style-cast
        -Wno-zero-as-null-pointer-constant -Wno-format-nonliteral
        -Wno-comment -Wno-documentation -Wno-documentation-unknown-command -Wno-comma
        -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-c++98-compat-local-type-template-args -Wno-c++98-compat-unnamed-type-template-args
        -Wno-c++98-compat-extra-semi
        -Wno-c++11-compat -Wno-c++14-compat -Wno-c++17-compat -Wno-c++20-compat
        -Wno-pre-c++14-compat -Wno-pre-c++17-compat
        -Wno-pre-c++20-compat -Wno-pre-c++23-compat -Wno-pre-c++26-compat
        -Wno-nested-anon-types -Wno-gnu-anonymous-struct
    )
elseif(VC_WARNINGS AND NOT MSVC)
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
endif()
if(VC_WERROR)
    if(MSVC)
        add_compile_options(/WX)
    else()
        add_compile_options(-Werror)
    endif()
endif()

if(VC_PROFILE_COMPILE)
    if(MSVC)
        # MSVC has no stable per-TU timing flag; leave compile profiling to the
        # GCC/Clang builds.
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-ftime-trace -ftime-trace-granularity=200)
    else()
        add_compile_options(-ftime-report)
    endif()
endif()

if(VC_DEAD_CODE_REPORT AND NOT MSVC)
    add_compile_options(
        -Wunused -Wunused-function -Wunused-variable -Wunused-parameter
        -Wunused-value -Wunused-result
        -Wunused-but-set-variable -Wunused-but-set-parameter
        -Wunused-local-typedef
        -Wunused-macros
        -Wunreachable-code
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(
            -Wunused-template
            -Wunused-member-function
            -Wunused-private-field
            -Wunreachable-code-aggressive
        )
    endif()
    add_link_options(-Wl,--print-gc-sections)
endif()

# ---- Dependencies -----------------------------------------------------------
if(VC_BUILD_UI_TRACER OR VC_BUILD_APPS)
    find_package(Qt6 QUIET REQUIRED COMPONENTS Widgets Gui Core Network Concurrent OpenGLWidgets)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
endif()
# Ceres is required unconditionally: core/CMakeLists.txt always builds
# vc_inpaint, vc_lasagna, vc_atlas and vc_tracer, all of which link
# Ceres::ceres (vc_lasagna is also pulled into the Python bindings via
# vc_lasagna_amgx).
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
# OpenCV 5 split calib3d into geometry/calib/stereo/ptcloud. VC uses
# Rodrigues (calib) and estimateAffine3D (geometry), so select both new
# modules there while retaining the OpenCV 4 module name elsewhere.
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
    set(VC_OPENCV_CALIB_TARGETS opencv_calib opencv_geometry)
else()
    set(VC_OPENCV_CALIB_TARGETS opencv_calib3d)
endif()

# Homebrew's OpenCV package can occasionally report module names without
# defining corresponding imported targets. Normalize any such module to an
# absolute dylib path so it never degrades into a bare, unresolvable -l flag.
if(APPLE)
    set(_vc_opencv_modules
        core imgproc imgcodecs video videoio flann ximgproc)
    if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
        list(APPEND _vc_opencv_modules calib geometry)
    else()
        list(APPEND _vc_opencv_modules calib3d)
    endif()
    foreach(_vc_opencv_module IN LISTS _vc_opencv_modules)
        set(_vc_opencv_target "opencv_${_vc_opencv_module}")
        if(NOT TARGET ${_vc_opencv_target})
            unset(_vc_opencv_library)
            find_library(_vc_opencv_library
                NAMES ${_vc_opencv_target}
                HINTS
                    "$ENV{HOMEBREW_PREFIX}/opt/opencv/lib"
                    "${OpenCV_DIR}/../.."
                NO_CACHE)
            if(NOT _vc_opencv_library)
                message(FATAL_ERROR
                    "OpenCV module ${_vc_opencv_target} was not provided as a target and its library was not found")
            endif()
            add_library(${_vc_opencv_target} SHARED IMPORTED GLOBAL)
            set_target_properties(${_vc_opencv_target} PROPERTIES
                IMPORTED_LOCATION "${_vc_opencv_library}"
                INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
        endif()
    endforeach()
    unset(_vc_opencv_library)
endif()
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)
find_package(TIFF REQUIRED)
find_package(ZLIB REQUIRED)
if(MSVC)
    # On MSVC, zstd/lz4 come from vcpkg CMake configs (below), and the
    # pkg-config-only consumers (vendored libbacktrace, avahi) are Linux-only,
    # so pkg-config is optional rather than required.
    find_package(PkgConfig)
else()
    find_package(PkgConfig REQUIRED)
endif()
# Blosc1 (libblosc-dev on Debian/Ubuntu; c-blosc on Homebrew). No upstream
# CMake config is shipped, so locate by header+lib and wrap in an imported
# target consumers link as Blosc::blosc.
#
# Note on package collision: libblosc-dev installs <blosc.h>; libblosc2-dev
# installs <blosc2.h> (a *different* header, not a compat shim). Searching
# `NAMES blosc.h` therefore only matches Blosc1. The matching libraries
# also have distinct SONAMEs (libblosc.so vs libblosc2.so), so a system
# with both packages installed cannot accidentally hybridize.
find_path(BLOSC_INCLUDE_DIR NAMES blosc.h
    HINTS $ENV{HOMEBREW_PREFIX}/opt/c-blosc/include
          $ENV{HOMEBREW_PREFIX}/include)
find_library(BLOSC_LIBRARY NAMES blosc
    HINTS $ENV{HOMEBREW_PREFIX}/opt/c-blosc/lib
          $ENV{HOMEBREW_PREFIX}/lib)
if(NOT BLOSC_INCLUDE_DIR OR NOT BLOSC_LIBRARY)
    message(FATAL_ERROR "Blosc1 (libblosc-dev / c-blosc) not found")
endif()
if(NOT TARGET Blosc::blosc)
    add_library(Blosc::blosc UNKNOWN IMPORTED)
    set_target_properties(Blosc::blosc PROPERTIES
        IMPORTED_LOCATION "${BLOSC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${BLOSC_INCLUDE_DIR}")
endif()

if(VC_ENABLE_AMGX)
    if(DEFINED AMGX_INCLUDE_DIR AND "${AMGX_INCLUDE_DIR}" STREQUAL "")
        unset(AMGX_INCLUDE_DIR CACHE)
    endif()
    if(DEFINED AMGX_LIBRARY AND "${AMGX_LIBRARY}" STREQUAL "")
        unset(AMGX_LIBRARY CACHE)
    endif()
    find_path(AMGX_INCLUDE_DIR
        NAMES amgx_c.h
        HINTS "${AMGX_ROOT}" "$ENV{AMGX_ROOT}"
        PATH_SUFFIXES include)
    find_library(AMGX_LIBRARY
        NAMES amgxsh amgx
        HINTS "${AMGX_ROOT}" "$ENV{AMGX_ROOT}"
        PATH_SUFFIXES lib lib64)
    if(NOT AMGX_INCLUDE_DIR OR NOT AMGX_LIBRARY)
        message(FATAL_ERROR
            "VC_ENABLE_AMGX=ON requires external AMGX. Set AMGX_ROOT or "
            "AMGX_INCLUDE_DIR and AMGX_LIBRARY.")
    endif()
    if(NOT TARGET AMGX::amgx)
        add_library(AMGX::amgx UNKNOWN IMPORTED)
        set_target_properties(AMGX::amgx PROPERTIES
            IMPORTED_LOCATION "${AMGX_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${AMGX_INCLUDE_DIR}")
    endif()
    message(STATUS "Using AMGX include=${AMGX_INCLUDE_DIR} library=${AMGX_LIBRARY}")
endif()
if(MSVC)
    # vcpkg ships CMake configs for these (no .pc/pkg-config on MSVC). Normalize
    # whatever target each exports to the names the tree consumes: utils links
    # zstd::libzstd and LZ4::lz4.
    find_package(zstd CONFIG REQUIRED)
    find_package(lz4 CONFIG REQUIRED)
    if(NOT TARGET zstd::libzstd)
        if(TARGET zstd::libzstd_shared)
            add_library(zstd::libzstd ALIAS zstd::libzstd_shared)
        elseif(TARGET zstd::libzstd_static)
            add_library(zstd::libzstd ALIAS zstd::libzstd_static)
        endif()
    endif()
    if(NOT TARGET LZ4::lz4 AND TARGET lz4::lz4)
        add_library(LZ4::lz4 ALIAS lz4::lz4)
    endif()
else()
    pkg_check_modules(zstd REQUIRED IMPORTED_TARGET libzstd)
    pkg_check_modules(lz4 REQUIRED IMPORTED_TARGET liblz4)
    if(NOT TARGET zstd::libzstd)
        add_library(zstd::libzstd ALIAS PkgConfig::zstd)
    endif()
    if(NOT TARGET LZ4::lz4)
        add_library(LZ4::lz4 ALIAS PkgConfig::lz4)
    endif()
endif()
if(VC_BUILD_APPS)
    find_package(CGAL REQUIRED)
endif()
if(VC_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE VC_NANOBIND_CMAKE_DIR
        RESULT_VARIABLE VC_NANOBIND_CMAKE_DIR_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(VC_NANOBIND_CMAKE_DIR_RESULT EQUAL 0 AND VC_NANOBIND_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${VC_NANOBIND_CMAKE_DIR}")
    endif()
    find_package(nanobind CONFIG REQUIRED)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR MSVC)
    # Clang and MSVC both use the no-op OpenMP shim: -fopenmp / -openmp is never
    # enabled, so every `#pragma omp` is ignored and the omp_*() runtime calls
    # resolve to the inline stubs in core/openmp_stub/omp.h. These builds run the
    # parallel regions serially, which is why the ignored `critical`/`atomic`
    # pragmas remain correct. Only the GCC build links real OpenMP. (MSVC
    # silences the resulting unknown-pragma C4068 in the compiler-family block.)
    include_directories(${CMAKE_SOURCE_DIR}/core/openmp_stub)
    add_library(openmp_stub INTERFACE)
    add_library(OpenMP::OpenMP_CXX ALIAS openmp_stub)
    add_library(OpenMP::OpenMP_C  ALIAS openmp_stub)
else()
    find_package(OpenMP REQUIRED)
endif()

set(VC_LIBBACKTRACE "")
if(VC_USE_SYSTEM_LIBBACKTRACE AND NOT WIN32)
    pkg_check_modules(LIBBACKTRACE QUIET IMPORTED_TARGET libbacktrace)
    if(TARGET PkgConfig::LIBBACKTRACE)
        set(VC_LIBBACKTRACE PkgConfig::LIBBACKTRACE)
        message(STATUS "Using system libbacktrace")
    endif()
endif()
# The crash handler (sole libbacktrace consumer) is Linux-only, and the
# vendored build is autotools — skip it entirely on Windows.
if(NOT VC_LIBBACKTRACE AND NOT WIN32)
    include(ExternalProject)
    include(ProcessorCount)
    ProcessorCount(_nproc)
    if(_nproc EQUAL 0)
        set(_nproc 1)
    endif()
    set(_libbt_install ${CMAKE_BINARY_DIR}/_deps/libbacktrace-install)
    set(_libbt_lib ${_libbt_install}/lib/libbacktrace.a)
    file(MAKE_DIRECTORY ${_libbt_install}/include)
    ExternalProject_Add(libbacktrace_build
        GIT_REPOSITORY https://github.com/ianlancetaylor/libbacktrace.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        UPDATE_DISCONNECTED ${VC_LIBBACKTRACE_UPDATE_DISCONNECTED}
        PREFIX ${CMAKE_BINARY_DIR}/_deps/libbacktrace
        CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=${_libbt_install} --enable-static --disable-shared --with-pic
        BUILD_COMMAND make -j${_nproc}
        INSTALL_COMMAND make install
        BUILD_BYPRODUCTS ${_libbt_lib})
    add_library(libbacktrace_static STATIC IMPORTED GLOBAL)
    set_target_properties(libbacktrace_static PROPERTIES
        IMPORTED_LOCATION ${_libbt_lib}
        INTERFACE_INCLUDE_DIRECTORIES ${_libbt_install}/include)
    add_dependencies(libbacktrace_static libbacktrace_build)
    set(VC_LIBBACKTRACE libbacktrace_static)
endif()

# ---- Subdirectories ---------------------------------------------------------
include_directories(core/include)

add_subdirectory(libs/OpenABF)
add_subdirectory(libs/c3d)
# igl::core is needed by flatboi (SLIM) and by apps/vc_obj_uv_lift; fetch it
# here — before the sanitizer flags below — so both consumers see the target
# even when flatboi is off (Windows).
if(VC_BUILD_FLATBOI OR VC_BUILD_APPS)
    include(cmake/FetchLibigl.cmake)
endif()
if(VC_BUILD_FLATBOI)
    add_subdirectory(libs/flatboi)
endif()

# Sanitizers/coverage on our code only — libs/ (PaStiX, libigl, ...) builds
# clean to keep sanitizer build times reasonable.
if(_san)
    add_compile_options(${_san})
    add_link_options(${_san})
endif()

add_subdirectory(utils)
add_subdirectory(core)
add_subdirectory(apps)
if(VC_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ---- Install rules ----------------------------------------------------------
# `cmake --install <build> --prefix <p> --component vc_runtime` lays the tree
# out as <p>/bin (executables) + <p>/lib (their internal .so), which the apps
# resolve via CMAKE_INSTALL_RPATH ($ORIGIN/../lib, set above) — no
# LD_LIBRARY_PATH or ldconfig needed. Used by the Docker image.
#
# Rather than enumerate ~35 executables and the vc_* libs by hand, walk each
# subdirectory's targets and install every EXECUTABLE and SHARED_LIBRARY
# (static/interface/PCH helper targets are skipped — they aren't needed at
# runtime). The vc_runtime component keeps this isolated from unrelated
# dependency install rules (e.g. libigl's CMake config exports).
set(_vc_install_dirs core utils libs/c3d)
if(VC_BUILD_APPS)
    list(APPEND _vc_install_dirs apps apps/VC3D)
endif()
if(VC_BUILD_FLATBOI)
    list(APPEND _vc_install_dirs libs/flatboi)
endif()
foreach(_dir IN LISTS _vc_install_dirs)
    get_property(_dir_targets DIRECTORY "${CMAKE_SOURCE_DIR}/${_dir}"
                 PROPERTY BUILDSYSTEM_TARGETS)
    foreach(_tgt IN LISTS _dir_targets)
        get_target_property(_tgt_type ${_tgt} TYPE)
        if(_tgt_type STREQUAL "EXECUTABLE" OR _tgt_type STREQUAL "SHARED_LIBRARY")
            if(WIN32)
                # Register each binary in a dependency set so the packaging
                # step (cmake/WindowsPackaging.cmake) can bundle the full DLL
                # closure. DLLs land in bin next to the executables — Windows
                # has no RPATH.
                install(TARGETS ${_tgt}
                    COMPONENT vc_runtime
                    RUNTIME_DEPENDENCY_SET vc_runtime_deps
                    RUNTIME DESTINATION bin
                    LIBRARY DESTINATION lib)
            elseif(APPLE AND VC_BUILD_APPS)
                # Everything ships inside the app bundle
                # (cmake/MacPackaging.cmake): CLI tools next to VC3D in
                # Contents/MacOS, vc_* dylibs in Contents/Frameworks, both
                # wired up via the @loader_path rpath set above.
                install(TARGETS ${_tgt}
                    COMPONENT vc_runtime
                    RUNTIME DESTINATION "VC3D.app/Contents/MacOS"
                    LIBRARY DESTINATION "VC3D.app/Contents/Frameworks")
            else()
                install(TARGETS ${_tgt}
                    COMPONENT vc_runtime
                    RUNTIME DESTINATION bin
                    LIBRARY DESTINATION lib)
            endif()
        endif()
    endforeach()
endforeach()

include(${CMAKE_SOURCE_DIR}/cmake/WindowsPackaging.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/MacPackaging.cmake)

if(VC_TESTING)
    enable_testing()
    add_subdirectory(core/test)
    add_subdirectory(apps/VC3D/agent_bridge/test)
    add_subdirectory(apps/VC3D/test)
    add_custom_target(check
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        USES_TERMINAL)

    if(VC_ENABLE_COVERAGE)
        find_program(LLVM_PROFDATA NAMES llvm-profdata REQUIRED)
        find_program(LLVM_COV NAMES llvm-cov REQUIRED)
        get_property(_vc_test_targets GLOBAL PROPERTY VC_TEST_TARGETS)
        string(REPLACE ";" "," _vc_test_targets_csv "${_vc_test_targets}")
        add_custom_target(coverage-report
            COMMAND ${CMAKE_COMMAND}
                    -D "BUILD_DIR=${CMAKE_BINARY_DIR}"
                    -D "CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
                    -D "LLVM_PROFDATA=${LLVM_PROFDATA}"
                    -D "LLVM_COV=${LLVM_COV}"
                    -D "TEST_TARGETS=${_vc_test_targets_csv}"
                    -P "${CMAKE_SOURCE_DIR}/cmake/run_coverage_report.cmake"
            DEPENDS ${_vc_test_targets}
            USES_TERMINAL
            COMMENT "Running clang source-based coverage")
    endif()
endif()
