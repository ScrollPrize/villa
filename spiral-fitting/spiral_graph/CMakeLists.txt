add_library(spiral_winding_graph_core STATIC
    src/winding_graph.cpp
    src/theta_topology.cpp
    src/input_graph.cpp
    src/fiber_io.cpp
    src/fiber_layout.cpp
    src/registration.cpp
    src/track_io.cpp
    src/track_spatial_index.cpp
)
add_library(spiral_graph::winding_core ALIAS spiral_winding_graph_core)
target_compile_features(spiral_winding_graph_core PUBLIC cxx_std_23)
target_include_directories(spiral_winding_graph_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(spiral_winding_graph_core PUBLIC
    vc_spiral::surface_core
    TIFF::TIFF
    nlohmann_json::nlohmann_json
    Ceres::ceres
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(spiral_winding_graph_core PUBLIC OpenMP::OpenMP_CXX)
endif()
set_target_properties(spiral_winding_graph_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(VC_SPIRAL_BUILD_PYTHON)
    if(NOT TARGET nanobind-static)
        find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            OUTPUT_VARIABLE SPIRAL_GRAPH_NANOBIND_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(SPIRAL_GRAPH_NANOBIND_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${SPIRAL_GRAPH_NANOBIND_CMAKE_DIR}")
        endif()
        find_package(nanobind CONFIG REQUIRED)
    endif()
    nanobind_add_module(spiral_winding_graph_python NOMINSIZE src/bindings.cpp)
    target_compile_features(spiral_winding_graph_python PRIVATE cxx_std_23)
    target_link_libraries(spiral_winding_graph_python PRIVATE spiral_graph::winding_core)
    set_target_properties(spiral_winding_graph_python PROPERTIES
        OUTPUT_NAME "_spiral_graph"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python/spiral_graph"
    )
    install(TARGETS spiral_winding_graph_python
        LIBRARY DESTINATION spiral_graph
        RUNTIME DESTINATION spiral_graph
    )
    install(DIRECTORY python/spiral_graph/
        DESTINATION spiral_graph
        FILES_MATCHING PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
    )
endif()

if(BUILD_TESTING)
    add_executable(spiral_winding_graph_tests tests/test_winding_graph.cpp)
    target_compile_features(spiral_winding_graph_tests PRIVATE cxx_std_23)
    target_link_libraries(spiral_winding_graph_tests PRIVATE spiral_graph::winding_core)
    add_test(NAME spiral_winding_graph_tests COMMAND spiral_winding_graph_tests)
endif()

if(VC_SPIRAL_BUILD_TOOLS)
    add_executable(benchmark_winding_graph_core
        ../benchmarks/benchmark_winding_graph_core.cpp)
    target_compile_features(benchmark_winding_graph_core PRIVATE cxx_std_23)
    target_link_libraries(benchmark_winding_graph_core PRIVATE spiral_graph::winding_core)
    set_target_properties(benchmark_winding_graph_core PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
