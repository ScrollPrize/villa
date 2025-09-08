cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
include(FetchContent)

set(VC_VERSION 3.0.2)

list(PREPEND CMAKE_PREFIX_PATH "$ENV{HOME}/vc-dependencies")

option(VC_VERSION_DATESTAMP "Append date stamp to version number" off)
if(VC_VERSION_DATESTAMP)
string(TIMESTAMP VC_VERSION_TWEAK "%s")
set(VC_VERSION ${VC_VERSION}.${VC_VERSION_TWEAK})
endif()

project(VC3D VERSION ${VC_VERSION})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

option(VC_USE_OPENMP "Enable OpenMP support" ON)
option(VC_USE_VALGRIND "Enable Valgrind support" off)

set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -fPIC -Wno-narrowing -std=c++23 -DWITH_BLOSC=1 -DWITH_ZLIB=1 " )
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
include_directories(core/include)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -g3 -gdwarf-5 -funwind-tables -fasynchronous-unwind-tables")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -g3 -gdwarf-5 -rdynamic -fPIE")
elseif(CMAKE_BUILD_TYPE STREQUAL "QuickBuild")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3  -flto=auto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto ")
elseif(CMAKE_BUILD_TYPE STREQUAL "ReleaseUnsafe")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3   -flto=auto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto -fuse-ld=lld")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -flto=auto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto")
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os  -flto=auto")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=auto")
endif()


option(VC_ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
option(VC_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(VC_ENABLE_TSAN "Enable ThreadSanitizer for data race detection" OFF)
option(VC_ENABLE_LSAN "Enable LeakSanitizer for memory leak detection (standalone)" OFF)

if(VC_ENABLE_ASAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be used together")
endif()

if(VC_ENABLE_LSAN AND VC_ENABLE_ASAN)
    message(WARNING "LeakSanitizer is already included with AddressSanitizer. Disabling standalone LeakSanitizer.")
    set(VC_ENABLE_LSAN OFF)
endif()

if(VC_ENABLE_LSAN AND VC_ENABLE_TSAN)
    message(FATAL_ERROR "LeakSanitizer and ThreadSanitizer cannot be used together")
endif()

if(VC_ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer  -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DADDRESS_SANITIZER")
else()
    message(STATUS "Addressanitizer disabled")
endif()

if(VC_ENABLE_UBSAN)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
    set(UBSAN_FLAGS "-fsanitize=undefined -fno-sanitize-recover=all")
    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=float-divide-by-zero")
    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=float-cast-overflow")
    set(UBSAN_FLAGS "${UBSAN_FLAGS} -fsanitize=nullability")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${UBSAN_FLAGS} -fno-omit-frame-pointer  -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${UBSAN_FLAGS} -fno-omit-frame-pointer  -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${UBSAN_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${UBSAN_FLAGS}")
else()
    message(STATUS "UndefinedBehaviorSanitizer disabled")
endif()

if(VC_ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-omit-frame-pointer  -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread -fno-omit-frame-pointer  -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message(STATUS "ThreadSanitizer disabled")
endif()

if(VC_ENABLE_LSAN)
    message(STATUS "LeakSanitizer enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -fno-omit-frame-pointer  -Og")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak -fno-omit-frame-pointer  -Og")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=leak")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=leak")
else()
    message(STATUS "LeakSanitizer disabled ")
endif()

if(VC_USE_VALGRIND)
    message(STATUS "Valgrind support enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-lto -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-avx512f -mno-fma -mno-bmi -mno-bmi2 -Og -g -fno-omit-frame-pointer " )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-avx512f -mno-avx512cd -mno-avx512bw -mno-avx512dq -mno-avx512vl -mno-fma")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=x86-64 -mtune=generic -mno-avx -mno-avx2 -mno-avx512f -mno-avx512cd -mno-avx512bw -mno-avx512dq -mno-avx512vl -mno-fma")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-lto ")
else()
    message(STATUS "Valgrind support disabled")
endif()

find_program(CCACHE_PROGRAM "ccache")
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
endif()

# Use modern C++ for everything and generate compile_commands database.
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# Choose what to build
option(VC_BUILD_APPS     "Compile VC core programs"    on)
option(VC_BUILD_GUI      "Compile GUI programs"        on)
option(VC_BUILD_UTILS    "Compile VC utility programs" on)

# Choose what to install
option(VC_INSTALL_APPS      "Install VC core programs"    on)
option(VC_INSTALL_LIBS      "Install VC libraries"        on)
option(VC_INSTALL_UTILS     "Install VC utility programs" on)

# Some helpful constants to be used in subprojects.
include(VCConstants)

# Look for external dependencies.
include(VCFindDependencies)

# Weird bug on macOS where system includes aren't included with the right flag
if (APPLE)
    set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
endif()

if (VC_BUILD_TESTS)
    enable_testing()
endif()

option(VC_DEVELOPER_WARNINGS "For developers: Enable extensive compiler warnings" OFF)
mark_as_advanced(VC_DEVELOPER_WARNINGS)
if(VC_DEVELOPER_WARNINGS)
    include(VCWarnings)
endif()

add_subdirectory(misc)
add_subdirectory(core)


if (VC_BUILD_APPS)
    add_subdirectory(apps)
endif()

include(VCPackageConfig)
include(VCInstall)
include(Packing)
