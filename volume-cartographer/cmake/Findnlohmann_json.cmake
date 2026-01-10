# Prefer the vendored nlohmann/json submodule when available.

if (TARGET nlohmann_json::nlohmann_json)
    set(nlohmann_json_FOUND TRUE)
else()
    set(_vc_json_root "${CMAKE_SOURCE_DIR}/thirdparty/json")
    if (EXISTS "${_vc_json_root}/include/nlohmann/json.hpp")
        add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
        set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_vc_json_root}/include"
        )
        set(nlohmann_json_INCLUDE_DIRS "${_vc_json_root}/include")
        set(nlohmann_json_FOUND TRUE)
    endif()
endif()

if (NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json requested but vendored submodule was not found.")
endif()
