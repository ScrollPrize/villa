cmake_minimum_required(VERSION 3.15)
project(vc3d-web LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vc3d-web
    src/main.cpp
    src/RenderCore_wasm.cpp
)

target_include_directories(vc3d-web PRIVATE src)

if(EMSCRIPTEN)
    # Emscripten-specific link flags:
    # -s EXPORTED_FUNCTIONS: list of C functions to export (with _ prefix)
    # -s EXPORTED_RUNTIME_METHODS: JS helper methods to expose
    # -s ALLOW_MEMORY_GROWTH: let WASM heap grow as needed
    # -s INITIAL_MEMORY: start with 64MB
    # -s MAXIMUM_MEMORY: cap at 2GB
    # -s NO_EXIT_RUNTIME: keep runtime alive after main()
    # -s MODULARIZE: wrap in a factory function for cleaner loading
    # -s EXPORT_NAME: name of the factory function
    set(EXPORTED_FUNCS
        "_wasm_init"
        "_wasm_resize"
        "_wasm_add_level"
        "_wasm_reset_levels"
        "_wasm_set_plane"
        "_wasm_pan"
        "_wasm_zoom"
        "_wasm_scroll"
        "_wasm_set_window_level"
        "_wasm_set_cache_max"
        "_wasm_load_chunk"
        "_wasm_render"
        "_wasm_pixels"
        "_wasm_width"
        "_wasm_height"
        "_wasm_cache_count"
        "_wasm_cache_bytes"
        "_wasm_scale"
        "_wasm_z_offset"
        "_wasm_pyramid_level"
        "_wasm_num_levels"
        "_wasm_origin_x"
        "_wasm_origin_y"
        "_wasm_origin_z"
        "_malloc"
        "_free"
    )
    # Join into comma-separated string
    string(REPLACE ";" "," EXPORTED_FUNCS_STR "${EXPORTED_FUNCS}")

    target_link_options(vc3d-web PRIVATE
        "SHELL:-s EXPORTED_FUNCTIONS=[${EXPORTED_FUNCS_STR}]"
        "SHELL:-s EXPORTED_RUNTIME_METHODS=[ccall,cwrap,HEAPU8,getValue]"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s INITIAL_MEMORY=67108864"
        "SHELL:-s MAXIMUM_MEMORY=2147483648"
        "SHELL:-s NO_EXIT_RUNTIME=1"
        "SHELL:-s MODULARIZE=1"
        "SHELL:-s EXPORT_NAME=createVC3D"
        "SHELL:-s ENVIRONMENT=web"
    )

    # Optimization: -O2 for reasonable speed, -O3 for max
    target_compile_options(vc3d-web PRIVATE -O2)

    set(CMAKE_EXECUTABLE_SUFFIX ".js")
endif()
