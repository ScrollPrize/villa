# In-tree minimal doctest replacement (vc_test.hpp). The doctest single-header
# was the heaviest include in the test build (~350s across the suite); ours is
# tiny. The doctest_compat/ shim keeps `#include <doctest/doctest.h>` working,
# and vc_test.hpp provides the doctest API subset the tests use.
add_library(doctest::doctest INTERFACE IMPORTED)
set_target_properties(doctest::doctest PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
        "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/doctest_compat")

find_package(Qt6 REQUIRED COMPONENTS Concurrent Core Gui Test Widgets)

# Dedicated PCH carrier for the test suite. Keep this STATIC and separate from
# production targets so tests can add test-only headers without bloating the
# common production PCH.
if(VC_USE_PCH)
    add_library(vc_test_pch STATIC vc_test_pch_dummy.cpp)
    set_target_properties(vc_test_pch PROPERTIES
        AUTOMOC OFF AUTOUIC OFF AUTORCC OFF
        POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(vc_test_pch PRIVATE vc_core)
    target_precompile_headers(vc_test_pch PRIVATE ${VC_PCH_HEADERS})
endif()

# vc_add_test(NAME ... [SOURCES ...] [LIBS ...] [INCLUDES ...] [DEFS ...])
#
# Wraps the add_executable / target_link_libraries / add_test trio. SOURCES
# defaults to <name>.cpp, LIBS to "doctest::doctest vc_core" (pass LIBS to
# override, e.g. Qt tests). Appends misc/sanitizer_config.cpp when any *San is
# enabled so __tsan_default_suppressions et al. are present in the test exe
# itself — otherwise the suppressions only land in the canary/VC3D and never
# fire for the real tests. VC_TEST_FIXTURES_DIR is baked into every exe.
function(vc_add_test)
    cmake_parse_arguments(T "" "NAME" "SOURCES;LIBS;INCLUDES;DEFS" ${ARGN})
    if(NOT T_NAME)
        message(FATAL_ERROR "vc_add_test: NAME is required")
    endif()
    if(NOT T_SOURCES)
        set(T_SOURCES ${T_NAME}.cpp)
    endif()
    if(NOT DEFINED T_LIBS)
        set(T_LIBS doctest::doctest vc_core)
    endif()
    if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN OR VC_ENABLE_LSAN
       OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN)
        list(APPEND T_SOURCES "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.cpp")
    endif()
    add_executable(${T_NAME} ${T_SOURCES})
    if(T_LIBS)
        target_link_libraries(${T_NAME} PRIVATE ${T_LIBS})
    endif()
    if(VC_USE_PCH AND T_LIBS AND "vc_core" IN_LIST T_LIBS)
        target_precompile_headers(${T_NAME} REUSE_FROM vc_test_pch)
    endif()
    if(T_INCLUDES)
        target_include_directories(${T_NAME} PRIVATE ${T_INCLUDES})
    endif()
    target_compile_definitions(${T_NAME} PRIVATE
        VC_TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    if(T_DEFS)
        target_compile_definitions(${T_NAME} PRIVATE ${T_DEFS})
    endif()
    add_test(NAME ${T_NAME} COMMAND ${T_NAME})
    set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS ${T_NAME})
endfunction()

vc_add_test(NAME test_smoke LIBS doctest::doctest)

vc_add_test(NAME test_appendmask_render)

vc_add_test(NAME test_chunked_plane_sampler_fallback)

vc_add_test(NAME test_volume_chunk_errors)

vc_add_test(NAME test_segmentation_autosave_state
    LIBS doctest::doctest
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_surface_display_name
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_segmentation_intersection_invalidation
    LIBS Qt6::Core Qt6::Gui Qt6::Test vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(ink_detection_overlay_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/InkDetectionOverlayController.hpp)
qt6_wrap_cpp(viewer_overlay_base_ink_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/ViewerOverlayControllerBase.hpp)
qt6_wrap_cpp(viewer_manager_overlay_test_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/ViewerManager.hpp)
qt6_wrap_cpp(cstate_ink_detection_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.hpp)
qt6_wrap_cpp(vccollection_ink_detection_moc ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
vc_add_test(NAME test_ink_detection_overlay
    SOURCES test_ink_detection_overlay.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/InkDetectionOverlayController.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/ViewerOverlayControllerBase.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataSegmentCache.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataSegmentCacheIO.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataManifest.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/VolumeViewerCmaps.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/COutlinedTextItem.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/CVolumeViewerView.cpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            test_viewer_manager_stub.cpp
            ${ink_detection_overlay_moc}
            ${viewer_overlay_base_ink_moc}
            ${viewer_manager_overlay_test_moc}
            ${cstate_ink_detection_moc}
            ${vccollection_ink_detection_moc}
    LIBS doctest::doctest Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Test vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D
    DEFS VC_TESTING)

qt6_wrap_cpp(viewer_overlay_base_surface_primitives_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/ViewerOverlayControllerBase.hpp)
qt6_wrap_cpp(fiber_overlay_surface_primitives_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/FiberOverlayController.hpp)
vc_add_test(NAME test_viewer_overlay_surface_primitives
    SOURCES test_viewer_overlay_surface_primitives.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/ViewerOverlayControllerBase.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/FiberOverlayController.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/AtlasOverlayController.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/AtlasControlPointsOverlayController.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/AtlasControlPointsDock.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/COutlinedTextItem.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/CVolumeViewerView.cpp
            test_viewer_manager_stub.cpp
            ${viewer_overlay_base_surface_primitives_moc}
            ${fiber_overlay_surface_primitives_moc}
            ${viewer_manager_overlay_test_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Test vc_core vc_atlas
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_line_annotation_context_menu
    SOURCES test_line_annotation_context_menu.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/CVolumeViewerView.cpp
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Test vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_scale_bar_units
    SOURCES test_scale_bar_units.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/CVolumeViewerView.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/CVolumeViewerView.hpp
    LIBS doctest::doctest Qt6::Core Qt6::Gui Qt6::Widgets vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(spiral_config_profile_editor_moc
    ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/SpiralConfigProfileEditor.hpp)
vc_add_test(NAME test_spiral_config_profile_editor
    SOURCES test_spiral_config_profile_editor.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/SpiralConfigProfileEditor.cpp
            ${spiral_config_profile_editor_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_spiral_point_chain
    SOURCES test_spiral_point_chain.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/SpiralPointChain.cpp
    LIBS doctest::doctest Qt6::Core vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(cfiber_widget_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/CFiberWidget.hpp)
vc_add_test(NAME test_fiber_widget
    SOURCES test_fiber_widget.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CFiberWidget.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNameDisplay.cpp
            ${cfiber_widget_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(lasagna_service_manager_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/LasagnaServiceManager.hpp)
qt6_wrap_cpp(lasagna_batch_window_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/LasagnaBatchWindow.hpp)
vc_add_test(NAME test_lasagna_service_manager_factory
    SOURCES test_lasagna_service_manager_factory.cpp
            test_lasagna_service_manager_stub.cpp
            ${lasagna_service_manager_moc}
    LIBS Qt6::Core vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_lasagna_batch_window
    SOURCES test_lasagna_batch_window.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LasagnaBatchWindow.cpp
            test_lasagna_service_manager_stub.cpp
            ${lasagna_batch_window_moc}
            ${lasagna_service_manager_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(segmentation_lasagna_panel_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.hpp)
qt6_wrap_cpp(collapsible_settings_group_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/CollapsibleSettingsGroup.hpp)
qt6_wrap_cpp(cstate_lasagna_panel_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.hpp)
qt6_wrap_cpp(vccollection_lasagna_panel_moc ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
vc_add_test(NAME test_segmentation_lasagna_panel_ui
    SOURCES test_segmentation_lasagna_panel_ui.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LasagnaBatchWindow.cpp
            test_lasagna_service_manager_stub.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/CollapsibleSettingsGroup.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.cpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            ${segmentation_lasagna_panel_moc}
            ${lasagna_batch_window_moc}
            ${lasagna_service_manager_moc}
            ${collapsible_settings_group_moc}
            ${cstate_lasagna_panel_moc}
            ${vccollection_lasagna_panel_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Concurrent Qt6::Test vc_core vc_atlas
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D
    DEFS VC_TEST_DISABLE_LASAGNA_DISCOVERY VC_TEST_DISABLE_LASAGNA_DIALOGS)

qt6_wrap_cpp(line_annotation_vccollection_moc ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
qt6_wrap_cpp(line_annotation_cstate_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.hpp)
vc_add_test(NAME test_line_annotation_generated_views
    SOURCES test_line_annotation_generated_views.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSaveJob.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSaveJob.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberSliceGeometry.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberSliceGeometry.hpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            ${line_annotation_vccollection_moc}
            ${line_annotation_cstate_moc}
    LIBS doctest::doctest Qt6::Core vc_core vc_lasagna
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_fiber_slice_geometry
    SOURCES test_fiber_slice_geometry.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberSliceGeometry.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberSliceGeometry.hpp
    LIBS doctest::doctest vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_lasagna_maxflow_graph LIBS doctest::doctest vc_lasagna)

# Qt-free data/IO layer.
vc_add_test(NAME test_point_collections)

vc_add_test(NAME test_open_data_coordinate_identity
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

# VCCollection shim: verify the on*() hooks still emit Qt signals. Needs the
# shim TU + its MOC (it isn't built into a lib anymore).
qt6_wrap_cpp(vccollection_test_moc ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
vc_add_test(NAME test_vccollection_signals
    SOURCES test_vccollection_signals.cpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            ${vccollection_test_moc}
    LIBS doctest::doctest vc_core Qt6::Core Qt6::Test)

qt6_wrap_cpp(point_collection_widget_moc
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CPointCollectionWidget.hpp
    ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
vc_add_test(NAME test_point_collection_widget
    SOURCES test_point_collection_widget.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CPointCollectionWidget.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/Keybinds.cpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            ${point_collection_widget_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_quadsurface_save_roundtrip)

vc_add_test(NAME test_atomic_imwrite_multi)

vc_add_test(NAME test_post_process)

# Pin the on-disk wire format: chunks must remain Blosc1-readable because
# VcDataset writes Zarr v2 metadata advertising compressor_id = "blosc".
# This catches any future migration to Blosc2 that would silently produce
# chunks numcodecs / zarr-python cannot decode.
#
# Links vc_core directly so it can call vc::testing::bloscCompressForTest,
# which exercises the *real* compress path used by VcDataset.
vc_add_test(NAME test_blosc1_compat LIBS doctest::doctest vc_core Blosc::blosc)

# Parser-only regression tests for vc_merge_tifxyz: gmResolveGrid +
# gmCheckConnected are compiled into vc_core, so link that and assert the
# failure messages exactly without spawning the binary.
vc_add_test(NAME test_merge_grid_resolve)

# End-to-end test that spawns the built vc_merge_tifxyz binary on two
# synthetic 16x16 tifxyz dirs. Opt-in via VC_RUN_E2E=1; the test skips
# itself if unset, so the default ctest run stays fast.
add_executable(test_merge_e2e_small test_merge_e2e_small.cpp)
target_link_libraries(test_merge_e2e_small PRIVATE doctest::doctest vc_core nlohmann_json::nlohmann_json)
if(VC_RUN_E2E)
    add_test(NAME test_merge_e2e_small COMMAND test_merge_e2e_small)
    set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS test_merge_e2e_small)
    # The binary path probe in the test prefers VC_MERGE_TIFXYZ_BIN, so
    # CTest runs reliably regardless of the working directory.
    set_tests_properties(test_merge_e2e_small PROPERTIES
        ENVIRONMENT "VC_RUN_E2E=1;VC_MERGE_TIFXYZ_BIN=$<TARGET_FILE:vc_merge_tifxyz>")
endif()

add_executable(bench_volume_sample bench_volume_sample.cpp)
target_link_libraries(bench_volume_sample PRIVATE doctest::doctest vc_core)
# Bench has a generous wall-clock assertion; not part of default ctest
# unless the user opts in via VC_RUN_BENCHES. Expose it as a target so
# it's always built and runnable manually.
if(VC_RUN_BENCHES)
    add_test(NAME bench_volume_sample COMMAND bench_volume_sample)
    set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS bench_volume_sample)
endif()

add_executable(bench_quadsurface_gen bench_quadsurface_gen.cpp)
target_link_libraries(bench_quadsurface_gen PRIVATE doctest::doctest vc_core)
# Perf-regression guard for QuadSurface::gen() (warp helpers + per-surface
# cache reuse). Loose relative assertion; opt-in via VC_RUN_BENCHES, always
# built as a target so it's runnable manually.
if(VC_RUN_BENCHES)
    add_test(NAME bench_quadsurface_gen COMMAND bench_quadsurface_gen)
    set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS bench_quadsurface_gen)
endif()

add_executable(bench_chunked_plane_sampler bench_chunked_plane_sampler.cpp)
target_link_libraries(bench_chunked_plane_sampler PRIVATE doctest::doctest vc_core)
# Perf guard for the slice sampler hot path. Like bench_volume_sample it
# carries only a loose wall-clock assertion, so it stays out of default
# ctest unless the user opts in via VC_RUN_BENCHES; the target is always
# built and runnable manually.
if(VC_RUN_BENCHES)
    add_test(NAME bench_chunked_plane_sampler COMMAND bench_chunked_plane_sampler)
    set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS bench_chunked_plane_sampler)
endif()

add_executable(bench_lasagna_line_edit bench_lasagna_line_edit.cpp)
target_link_libraries(bench_lasagna_line_edit PRIVATE vc_lasagna)
if(VC_RUN_BENCHES)
    add_test(NAME bench_lasagna_line_edit COMMAND bench_lasagna_line_edit --help)
    set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS bench_lasagna_line_edit)
endif()

add_executable(bench_fiber_intersections bench_fiber_intersections.cpp)
target_link_libraries(bench_fiber_intersections PRIVATE vc_atlas vc_lasagna nlohmann_json::nlohmann_json)
if(VC_RUN_BENCHES)
    add_test(NAME bench_fiber_intersections COMMAND bench_fiber_intersections --help)
    set_property(GLOBAL APPEND PROPERTY VC_TEST_TARGETS bench_fiber_intersections)
endif()

# Sanitizer canaries: one exe, one subcommand per sanitizer. Each canary
# triggers a violation that the matching sanitizer must report; ctest
# expects the abort and matches the report text via PASS_REGULAR_EXPRESSION.
# A miss (sanitizer not actually instrumenting) makes the canary exit 0,
# which fails the test — that's the whole point.
set(_canary_srcs sanitizer_canary.cpp)
if(VC_ENABLE_ASAN OR VC_ENABLE_UBSAN OR VC_ENABLE_TSAN OR VC_ENABLE_LSAN
   OR VC_ENABLE_TYSAN OR VC_ENABLE_NSAN)
    list(APPEND _canary_srcs "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.cpp")
endif()
add_executable(sanitizer_canary ${_canary_srcs})
set_target_properties(sanitizer_canary PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)

# ctest: PASS_REGULAR_EXPRESSION must match in stdout/stderr for the test
# to pass. The canary aborts via the sanitizer runtime on success, or
# `exit(1)`s on the fall-through (sanitizer not actually instrumenting), so
# both the regex and a non-zero exit are required for a meaningful signal.
# Regexes are tight — generic substrings like "runtime error" can be emitted
# by unrelated failures and would mask a regression.
function(_add_canary name arg regex)
    add_test(NAME ${name} COMMAND sanitizer_canary ${arg})
    set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
endfunction()

if(VC_ENABLE_ASAN)
    _add_canary(sanitizer_canary_asan  asan  "AddressSanitizer: heap-buffer-overflow")
endif()
if(VC_ENABLE_UBSAN)
    _add_canary(sanitizer_canary_ubsan ubsan "runtime error: signed integer overflow")
endif()
if(VC_ENABLE_TSAN)
    _add_canary(sanitizer_canary_tsan  tsan  "ThreadSanitizer: data race")
endif()
if(VC_ENABLE_TYSAN)
    _add_canary(sanitizer_canary_tysan tysan "TypeSanitizer: type-aliasing-violation")
endif()
if(VC_ENABLE_NSAN)
    _add_canary(sanitizer_canary_nsan  nsan  "NumericalStabilitySanitizer: inconsistent")
endif()

vc_add_test(NAME test_gridstore)

vc_add_test(NAME test_gridstore_branches)

vc_add_test(NAME test_gridstore_malformed)

# Live network test: anonymous read of the PHerc 0172 zarr on AWS S3.
# Soft-skips on network failure so offline runs still pass; set
# VC_TEST_REQUIRE_NETWORK=1 to flip those skips into hard failures.
vc_add_test(NAME test_pherc0172_live)

vc_add_test(NAME test_tiff)

vc_add_test(NAME test_normalgridtools)

# Live S3: opens the PHerc 0172 volume and reads small chunks.
vc_add_test(NAME test_volume_live_s3)

vc_add_test(NAME test_normal_grid_volume)

vc_add_test(NAME test_surface_patch_index)

vc_add_test(NAME test_tiff_merge)

vc_add_test(NAME test_http_fetch_errors)

# Live S3: stages PHerc 0172 level 5 to a local zarr dir, then exercises
# the NormalGridGenerate library API against it. Soft-skips on network.
vc_add_test(NAME test_normal_grid_live)

vc_add_test(NAME test_normalgridtools_real)

vc_add_test(NAME test_tiff_more)

vc_add_test(NAME test_geometry)

vc_add_test(NAME test_affine_transform)

vc_add_test(NAME test_point_index)

vc_add_test(NAME test_plane_surface)

vc_add_test(NAME test_compositing)

vc_add_test(NAME test_quadsurface_basics)

vc_add_test(NAME test_load_json)

vc_add_test(NAME test_lasagna_manifest LIBS doctest::doctest vc_lasagna)

vc_add_test(NAME test_open_data_manifest
    SOURCES test_open_data_manifest.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataManifest.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataLasagna.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataLasagna.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataManifest.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataSampleProject.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataSampleProject.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataNormalGrids.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataNormalGrids.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataSegmentCache.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataSegmentCache.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataSegmentCacheIO.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataSegmentCacheIO.hpp
    LIBS doctest::doctest Qt6::Core vc_core vc_lasagna
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_lasagna_line_optimizer LIBS doctest::doctest vc_lasagna)

vc_add_test(NAME test_lasagna_line_view_surfaces LIBS doctest::doctest vc_lasagna)

vc_add_test(NAME test_lasagna_normal_sampler LIBS doctest::doctest vc_atlas)

vc_add_test(NAME test_atlas LIBS doctest::doctest vc_atlas)

vc_add_test(NAME test_fiber_intersections LIBS doctest::doctest vc_atlas)

vc_add_test(NAME test_remote_url)

vc_add_test(NAME test_logging)

vc_add_test(NAME test_plane_surface_branches)

vc_add_test(NAME test_polyline_index)

vc_add_test(NAME test_affine_transform_surface)

vc_add_test(NAME test_quadsurface_more)

vc_add_test(NAME test_thinning)

vc_add_test(NAME test_umbilicus)

vc_add_test(NAME test_colormaps)

vc_add_test(NAME test_quadsurface_io)

vc_add_test(NAME test_quadsurface_pointto)

vc_add_test(NAME test_quadsurface_snapshot)

vc_add_test(NAME test_quadsurface_save_concurrency)

vc_add_test(NAME test_quadsurface_save_paths)

vc_add_test(NAME test_segmentation)

vc_add_test(NAME test_chunked_tensor_meta LIBS doctest::doctest vc_tracer)

vc_add_test(NAME test_post_process_util)

vc_add_test(NAME test_render)

vc_add_test(NAME test_growth_config
    LIBS doctest::doctest vc_tracer
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/..)

vc_add_test(NAME test_growth_helpers
    LIBS doctest::doctest vc_tracer
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/..)

vc_add_test(NAME test_surf_tracker_data
    LIBS doctest::doctest vc_tracer
    INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/..)

vc_add_test(NAME test_sparse_chunk_spool)

vc_add_test(NAME test_neural_tracer_connection LIBS doctest::doctest vc_tracer)

vc_add_test(NAME test_quadsurface_final)

vc_add_test(NAME test_quadsurface_components)

vc_add_test(NAME test_volume_local)

vc_add_test(NAME test_zarr_helpers)

vc_add_test(NAME test_chunked_plane_sampler)

vc_add_test(NAME test_slicing)

vc_add_test(NAME test_volume_pkg)

vc_add_test(NAME test_volpkg_convert)

vc_add_test(NAME test_vcdataset)

vc_add_test(NAME test_zarr)

vc_add_test(NAME test_chunk_cache)

vc_add_test(NAME test_volume_extras)

vc_add_test(NAME test_zarr_more)

vc_add_test(NAME test_vcdataset_compressors)

vc_add_test(NAME test_volume_static_read)

vc_add_test(NAME test_volume_pyramid)

vc_add_test(NAME test_cache_compression)

vc_add_test(NAME test_chunk_cache_persist)

vc_add_test(NAME test_persistent_zarr_cache_budget)

vc_add_test(NAME test_open_data_volume_prefill
    SOURCES test_open_data_volume_prefill.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataVolumePrefill.cpp
    LIBS doctest::doctest vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_zarr_chunk_fetcher)

vc_add_test(NAME test_chunked_plane_sampler_more)

vc_add_test(NAME test_slicing_more)

vc_add_test(NAME test_vcdataset_more)

# Real PHerc 0172 segments committed under core/test/data/segments/.
vc_add_test(NAME test_quadsurface_fixtures)

vc_add_test(NAME test_volume_pkg_full)

vc_add_test(NAME test_load_quad_from_tifxyz)

vc_add_test(NAME test_surface_patch_index_more)

vc_add_test(NAME test_quadsurface_extras)

vc_add_test(NAME test_surface_patch_index_cache)

vc_add_test(NAME test_volume_pkg_more)
