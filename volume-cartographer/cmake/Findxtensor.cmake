# Prefer the vendored xtensor submodule when available.

if (TARGET xtensor)
    set(xtensor_FOUND TRUE)
    get_target_property(xtensor_INCLUDE_DIRS xtensor INTERFACE_INCLUDE_DIRECTORIES)
    set(xtensor_LIBRARIES xtensor)
else()
    set(_vc_xtensor_root "${CMAKE_SOURCE_DIR}/thirdparty/xtensor")
    set(xtensor_INCLUDE_DIRS "${_vc_xtensor_root}/include")
    if (EXISTS "${xtensor_INCLUDE_DIRS}/xtensor/xtensor.hpp")
        add_library(xtensor INTERFACE)
        target_include_directories(xtensor INTERFACE
            $<BUILD_INTERFACE:${xtensor_INCLUDE_DIRS}>
        )
        set(xtensor_LIBRARIES xtensor)
        set(xtensor_FOUND TRUE)
    endif()
endif()

if (NOT xtensor_FOUND)
    message(FATAL_ERROR "xtensor requested but vendored submodule was not found.")
endif()
