# VCToolchain.cmake - Environment and toolchain detection

# Prevent z5 from auto-detecting conda
set(WITHIN_TRAVIS ON CACHE BOOL "Skip z5 conda detection" FORCE)

# Ignore conda paths
if(EXISTS "$ENV{HOME}/miniconda3")
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{HOME}/miniconda3")
endif()
if(EXISTS "$ENV{HOME}/anaconda3")
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{HOME}/anaconda3")
endif()

# Use Homebrew LLVM toolchain on macOS
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    get_filename_component(_LLVM_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    foreach(_tool IN ITEMS llvm-ar llvm-nm llvm-ranlib llvm-strip llvm-objdump llvm-objcopy)
        find_program(_TOOL_PATH ${_tool} HINTS "${_LLVM_BIN_DIR}" NO_DEFAULT_PATH)
        if(_TOOL_PATH)
            string(REGEX REPLACE "^llvm-" "" _cmake_name "${_tool}")
            string(TOUPPER "${_cmake_name}" _cmake_name)
            set(CMAKE_${_cmake_name} "${_TOOL_PATH}" CACHE FILEPATH "" FORCE)
            message(STATUS "Toolchain: ${_tool} -> ${_TOOL_PATH}")
        endif()
        unset(_TOOL_PATH CACHE)
    endforeach()
endif()

# ccache (disabled when PCH is on unless explicitly opted in)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND NOT VC_USE_PCH)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache")
elseif(CCACHE_PROGRAM AND VC_USE_PCH)
    option(VC_USE_CCACHE "Use ccache alongside PCH" OFF)
    if(VC_USE_CCACHE)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
        message(STATUS "Using ccache + PCH")
    else()
        message(STATUS "Using PCH (ccache available but disabled)")
    endif()
elseif(VC_USE_PCH)
    message(STATUS "Using PCH (no ccache)")
endif()
