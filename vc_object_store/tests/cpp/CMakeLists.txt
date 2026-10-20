function(vc_object_store_test name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(${name} PRIVATE vc_objects nlohmann_json::nlohmann_json ${ARGN})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

foreach(name test_object_records test_object_store test_progress test_typed_geometry
             test_fiber_region)
    vc_object_store_test(${name})
endforeach()
foreach(name test_surface_channels test_object_lifecycle test_structural_geometry)
    vc_object_store_test(${name} OpenSSL::Crypto)
endforeach()

add_executable(bench_structural_edits bench_structural_edits.cpp)
target_include_directories(bench_structural_edits PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
target_link_libraries(bench_structural_edits PRIVATE vc_objects nlohmann_json::nlohmann_json OpenSSL::Crypto)

if(VC_OBJECT_STORE_BUILD_ADAPTERS)
    vc_object_store_test(test_object_links vc_object_links vc_track_crossings OpenMP::OpenMP_CXX opencv_core)
    vc_object_store_test(test_patch_queries vc_object_links opencv_core)

    add_executable(bench_object_edits bench_object_edits.cpp)
    target_include_directories(bench_object_edits PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(bench_object_edits PRIVATE vc_objects vc_object_formats nlohmann_json::nlohmann_json)

    add_executable(bench_source_import bench_source_import.cpp)
    target_include_directories(bench_source_import PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
    target_link_libraries(bench_source_import PRIVATE vc_object_geometry nlohmann_json::nlohmann_json OpenSSL::Crypto)
    add_test(NAME test_source_import_work COMMAND bench_source_import 1)
endif()
