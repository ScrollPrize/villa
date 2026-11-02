# Workflow benchmarks (VC_BUILD_BENCHMARKS): real VC3D viewers, the Line
# Annotation views and vc_render_tifxyz on real data. The same sources build in
# trees with VC's own ChunkCache pipeline (origin/main) and in vc_reader trees;
# only bench_pipeline_*.cpp is chosen per tree. See README.md.

if(EXISTS "${CMAKE_SOURCE_DIR}/core/include/vc/core/render/ChunkCache.hpp")
    set(_vc_bench_pipeline bench_pipeline_chunkcache.cpp)
    set(_vc_bench_vc_reader OFF)
else()
    set(_vc_bench_pipeline bench_pipeline_vcr.cpp)
    set(_vc_bench_vc_reader ON)
endif()

add_library(vc_bench_report STATIC bench_report.cpp)
target_include_directories(vc_bench_report PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vc_bench_report PUBLIC ${OpenCV_LIBS})
target_include_directories(vc_bench_report SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})

# The viewer benchmarks link VC3D's own objects (vc3d_app), not copies of them.
add_library(vc_bench_session STATIC bench_common.cpp ${_vc_bench_pipeline})
set_target_properties(vc_bench_session PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
target_link_libraries(vc_bench_session PUBLIC vc3d_app vc_bench_report)

foreach(_bench bench_viewer_navigation bench_line_annotation)
    add_executable(${_bench} ${_bench}.cpp)
    set_target_properties(${_bench} PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
    target_link_libraries(${_bench} PRIVATE vc_bench_session)
endforeach()

add_executable(bench_render_tifxyz bench_render_tifxyz.cpp)
target_link_libraries(bench_render_tifxyz PRIVATE vc_bench_report opencv_imgcodecs)
target_compile_definitions(bench_render_tifxyz PRIVATE
    VC_BENCH_RENDER_TIFXYZ="$<TARGET_FILE:vc_render_tifxyz>"
    $<$<BOOL:${_vc_bench_vc_reader}>:VC_BENCH_VC_READER>)
add_dependencies(bench_render_tifxyz vc_render_tifxyz)
