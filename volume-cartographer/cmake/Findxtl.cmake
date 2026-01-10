# Prefer the vendored xtl submodule when available.

if (TARGET xtl)
    set(xtl_FOUND TRUE)
    get_target_property(xtl_INCLUDE_DIRS xtl INTERFACE_INCLUDE_DIRECTORIES)
    set(xtl_LIBRARIES xtl)
else()
    set(_vc_xtl_root "${CMAKE_SOURCE_DIR}/thirdparty/xtl")
    set(xtl_INCLUDE_DIRS "${_vc_xtl_root}/include")
    if (EXISTS "${xtl_INCLUDE_DIRS}/xtl/xtl_config.hpp")
        add_library(xtl INTERFACE)
        target_include_directories(xtl INTERFACE
            $<BUILD_INTERFACE:${xtl_INCLUDE_DIRS}>
        )
        set(xtl_LIBRARIES xtl)
        set(xtl_FOUND TRUE)
    endif()
endif()

if (NOT xtl_FOUND)
    message(FATAL_ERROR "xtl requested but vendored submodule was not found.")
endif()
