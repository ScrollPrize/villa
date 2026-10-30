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
    vc_register_test_target(${T_NAME})
    vc_register_ctest(${T_NAME} ${T_NAME})
endfunction()

vc_add_test(NAME test_fiber_view LIBS doctest::doctest vc_fiber_view)
vc_add_test(NAME test_fiber_constraint_geometry LIBS doctest::doctest vc_object_store::vc_object_geometry)

vc_add_test(NAME test_omp_thread_point_collection)

vc_add_test(NAME test_surface_area_units)


vc_add_test(NAME test_segmentation_autosave_state
    LIBS doctest::doctest
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_volumetric_compositor
    SOURCES test_volumetric_compositor.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/VolumetricCompositor.cpp
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(ink_detection_overlay_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/InkDetectionOverlayController.hpp)
qt6_wrap_cpp(viewer_overlay_base_ink_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/ViewerOverlayControllerBase.hpp)
qt6_wrap_cpp(viewer_manager_overlay_test_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/ViewerManager.hpp)
qt6_wrap_cpp(cstate_ink_detection_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.hpp
             ${PROJECT_SOURCE_DIR}/apps/VC3D/PointCollectionSync.hpp)
qt6_wrap_cpp(vccollection_ink_detection_moc ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
qt6_wrap_cpp(surface_catalog_ink_detection_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/SurfaceCatalog.hpp)
vc_add_test(NAME test_ink_detection_overlay
    SOURCES test_ink_detection_overlay.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/InkDetectionOverlayController.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/ViewerOverlayControllerBase.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataManifest.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/SurfaceCatalog.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenedVolumes.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/PointCollectionSync.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/COutlinedTextItem.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/CVolumeViewerView.cpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            test_viewer_manager_stub.cpp
            ${ink_detection_overlay_moc}
            ${viewer_overlay_base_ink_moc}
            ${viewer_manager_overlay_test_moc}
            ${cstate_ink_detection_moc}
            ${vccollection_ink_detection_moc}
            ${surface_catalog_ink_detection_moc}
    LIBS vc_object_store::vc_objects vc_object_store::vc_object_geometry doctest::doctest Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Test vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D
    DEFS VC_TESTING)

qt6_wrap_cpp(viewer_overlay_base_surface_primitives_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/ViewerOverlayControllerBase.hpp)
qt6_wrap_cpp(points_overlay_surface_primitives_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/PointsOverlayController.hpp)
qt6_wrap_cpp(vccollection_surface_primitives_moc ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
vc_add_test(NAME test_viewer_overlay_surface_primitives
    SOURCES test_viewer_overlay_surface_primitives.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/ViewerOverlayControllerBase.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/PointsOverlayController.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/overlays/OverlayBatchItem.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/COutlinedTextItem.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/CVolumeViewerView.cpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            test_viewer_manager_stub.cpp
            ${viewer_overlay_base_surface_primitives_moc}
            ${points_overlay_surface_primitives_moc}
            ${vccollection_surface_primitives_moc}
            ${viewer_manager_overlay_test_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Test vc_core
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

qt6_wrap_cpp(cfiber_widget_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/CFiberWidget.hpp)
vc_add_test(NAME test_fiber_widget
    SOURCES test_fiber_widget.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CFiberWidget.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNameDisplay.cpp
            ${cfiber_widget_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_fiber_trace_review
    SOURCES test_fiber_trace_review.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSegments.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSegments.hpp
    LIBS doctest::doctest vc_core vc_lasagna vc_fiber_tracer
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(lasagna_service_manager_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/LasagnaServiceManager.hpp)
qt6_wrap_cpp(lasagna_batch_window_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/LasagnaBatchWindow.hpp)
vc_add_test(NAME test_lasagna_batch_window
    SOURCES test_lasagna_batch_window.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LasagnaBatchWindow.cpp
            test_lasagna_service_manager_stub.cpp
            ${lasagna_batch_window_moc}
            ${lasagna_service_manager_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Test vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(segmentation_lasagna_panel_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.hpp)
qt6_wrap_cpp(collapsible_settings_group_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/CollapsibleSettingsGroup.hpp)
qt6_wrap_cpp(cstate_lasagna_panel_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.hpp
             ${PROJECT_SOURCE_DIR}/apps/VC3D/PointCollectionSync.hpp)
qt6_wrap_cpp(vccollection_lasagna_panel_moc ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
qt6_wrap_cpp(surface_catalog_lasagna_panel_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/SurfaceCatalog.hpp)
vc_add_test(NAME test_segmentation_lasagna_panel_ui
    SOURCES test_segmentation_lasagna_panel_ui.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/segmentation/panels/SegmentationLasagnaPanel.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LasagnaBatchWindow.cpp
            test_lasagna_service_manager_stub.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/CollapsibleSettingsGroup.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenedVolumes.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/PointCollectionSync.cpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/SurfaceCatalog.cpp
            ${segmentation_lasagna_panel_moc}
            ${lasagna_batch_window_moc}
            ${lasagna_service_manager_moc}
            ${collapsible_settings_group_moc}
            ${cstate_lasagna_panel_moc}
            ${vccollection_lasagna_panel_moc}
            ${surface_catalog_lasagna_panel_moc}
    LIBS vc_object_store::vc_objects vc_object_store::vc_object_geometry Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Concurrent Qt6::Test vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D
    DEFS VC_TEST_DISABLE_LASAGNA_DISCOVERY VC_TEST_DISABLE_LASAGNA_DIALOGS)

qt6_wrap_cpp(spiral_config_profile_editor_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/SpiralConfigProfileEditor.hpp)
qt6_wrap_cpp(spiral_descriptions_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/SpiralDescriptions.hpp)
vc_add_test(NAME test_spiral_config_profile_editor
    SOURCES test_spiral_config_profile_editor.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/SpiralConfigProfileEditor.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/CollapsibleSettingsGroup.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/SpiralDescriptions.cpp
            ${spiral_config_profile_editor_moc}
            ${spiral_descriptions_moc}
            ${collapsible_settings_group_moc}
    LIBS Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Test vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

qt6_wrap_cpp(line_annotation_vccollection_moc ${PROJECT_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)
qt6_wrap_cpp(line_annotation_cstate_moc ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.hpp
             ${PROJECT_SOURCE_DIR}/apps/VC3D/PointCollectionSync.hpp)
vc_add_test(NAME test_line_annotation_link_validation
    SOURCES test_line_annotation_link_validation.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLineGeometry.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLineGeometry.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSegments.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSegments.hpp
    LIBS doctest::doctest Qt6::Core vc_core vc_lasagna vc_fiber_tracer
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_line_annotation_structural_edits
    SOURCES test_line_annotation_structural_edits.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLineGeometry.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLineGeometry.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSegments.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSegments.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationStructuralEdits.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationStructuralEdits.hpp
    LIBS doctest::doctest Qt6::Core vc_core vc_lasagna vc_fiber_tracer
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_line_annotation_generated_views
    SOURCES test_line_annotation_generated_views.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSegments.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSegments.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenedVolumes.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/CState.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/PointCollectionSync.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/PointCollectionSync.hpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLineGeometry.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLineGeometry.hpp
            ${PROJECT_SOURCE_DIR}/core/src/VCCollection.cpp
            ${line_annotation_vccollection_moc}
            ${line_annotation_cstate_moc}
    LIBS vc_object_store::vc_objects vc_object_store::vc_object_geometry doctest::doctest Qt6::Core vc_core vc_lasagna vc_fiber_tracer
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_fiber_line_geometry
    SOURCES test_fiber_line_geometry.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLineGeometry.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLineGeometry.hpp
    LIBS doctest::doctest vc_core
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_lasagna_maxflow_graph LIBS doctest::doctest vc_lasagna)
vc_add_test(NAME test_fiber_trace3d
    LIBS doctest::doctest vc_fiber_tracer
    DEFS VC_TESTING)
vc_add_test(NAME test_fiber_anchors LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_json LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_trace_winding_bp LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_lasagna_normal_alignment LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_prediction_channels LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_reference_replay_benchmark LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_replay_metric LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_embedding_scoring LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_width_scoring LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_reverse_consistency LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_reference_replay_core LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_chunk_trace LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_prediction_pooling LIBS doctest::doctest vc_lasagna)
vc_add_test(NAME test_fiber_case_queue LIBS doctest::doctest vc_fiber_tracer)
set_tests_properties(test_fiber_case_queue PROPERTIES TIMEOUT 30)
vc_add_test(NAME test_fiberlet_storage LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiberlet_crop_trace LIBS doctest::doctest vc_fiber_tracer)
vc_add_test(NAME test_fiber_replay_visualization LIBS doctest::doctest vc_fiber_tracer)
target_compile_definitions(test_fiber_replay_visualization PRIVATE VC_TESTING)
vc_add_test(NAME test_fiberlet_paths LIBS doctest::doctest vc_fiber_tracer DEFS VC_TESTING)
if(TARGET test_fiberlet_paths AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # The inline optimized/reference kernels assert bitwise operation-order
    # equality. Do not let the compiler contract just one expression into FMA.
    # This affects the comparison harness only, not production code.
    target_compile_options(test_fiberlet_paths PRIVATE -ffp-contract=off)
endif()

# Qt-free data/IO layer.
vc_add_test(NAME test_point_collections)

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

vc_add_test(NAME test_post_process)

# Parser-only regression tests for vc_merge_tifxyz: compile its
# gmResolveGrid + gmCheckConnected source and assert the failure messages
# exactly without spawning the binary.
vc_add_test(NAME test_merge_grid_resolve
    SOURCES test_merge_grid_resolve.cpp ${PROJECT_SOURCE_DIR}/apps/src/vc_merge_tifxyz_grid.cpp
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/src)

vc_add_test(NAME test_surface_meshing)

# These spawn the built command-line tools, so they only exist in
# configurations that build the apps -- the test shards configure with
# VC_BUILD_APPS=OFF and apps/ returns before the tool targets are created.
if(VC_BUILD_APPS)
    vc_add_test(NAME test_ngrids_zarr_output LIBS doctest::doctest vc_core Qt6::Core)
    add_dependencies(test_ngrids_zarr_output vc_ngrids)
    set_tests_properties(test_ngrids_zarr_output PROPERTIES
        ENVIRONMENT "VC_NGRIDS_BIN=$<TARGET_FILE:vc_ngrids>"
        TIMEOUT 180)

    # Spawns the built vc_grow_seg_from_seed on a synthetic sheet in store mode.
    vc_add_test(NAME test_grow_seg_store_mode)
    add_dependencies(test_grow_seg_store_mode vc_grow_seg_from_seed)
    set_tests_properties(test_grow_seg_store_mode PROPERTIES
        ENVIRONMENT "VC_GROW_SEG_FROM_SEED_BIN=$<TARGET_FILE:vc_grow_seg_from_seed>")

    # Outputs of the mesh tools (OBJ export, OBJ rasterization)
    # on small fixed inputs.
    vc_add_test(NAME test_mesh_tool_outputs)
    add_dependencies(test_mesh_tool_outputs vc_tifxyz2obj vc_obj2tifxyz vc_obj2tifxyz_legacy)
    set_tests_properties(test_mesh_tool_outputs PROPERTIES
        ENVIRONMENT "VC_TIFXYZ2OBJ_BIN=$<TARGET_FILE:vc_tifxyz2obj>;VC_OBJ2TIFXYZ_BIN=$<TARGET_FILE:vc_obj2tifxyz>;VC_OBJ2TIFXYZ_LEGACY_BIN=$<TARGET_FILE:vc_obj2tifxyz_legacy>")

    # Store modes of the command-line tools, compared with their tifxyz modes.
    vc_add_test(NAME test_tool_store_modes)
    if(TARGET flatboi)
        add_dependencies(test_tool_store_modes flatboi)
    endif()
    add_dependencies(test_tool_store_modes vc_tifxyz2obj vc_render_tifxyz vc_obj2tifxyz vc_obj2tifxyz_legacy vc_merge_patch vc_merge_tifxyz vc_objrefine vc_straighten vc_grow_seg_from_segments)
    set_tests_properties(test_tool_store_modes PROPERTIES
        ENVIRONMENT "$<$<TARGET_EXISTS:flatboi>:VC_FLATBOI_BIN=$<TARGET_FILE:flatboi>;>VC_TIFXYZ2OBJ_BIN=$<TARGET_FILE:vc_tifxyz2obj>;VC_RENDER_TIFXYZ_BIN=$<TARGET_FILE:vc_render_tifxyz>;VC_OBJ2TIFXYZ_BIN=$<TARGET_FILE:vc_obj2tifxyz>;VC_OBJ2TIFXYZ_LEGACY_BIN=$<TARGET_FILE:vc_obj2tifxyz_legacy>;VC_MERGE_PATCH_BIN=$<TARGET_FILE:vc_merge_patch>;VC_MERGE_TIFXYZ_BIN=$<TARGET_FILE:vc_merge_tifxyz>;VC_OBJREFINE_BIN=$<TARGET_FILE:vc_objrefine>;VC_STRAIGHTEN_BIN=$<TARGET_FILE:vc_straighten>;VC_GROW_SEG_FROM_SEGMENTS_BIN=$<TARGET_FILE:vc_grow_seg_from_segments>")
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
vc_register_test_target(sanitizer_canary)

# ctest: PASS_REGULAR_EXPRESSION must match in stdout/stderr for the test
# to pass. The canary aborts via the sanitizer runtime on success, or
# `exit(1)`s on the fall-through (sanitizer not actually instrumenting), so
# both the regex and a non-zero exit are required for a meaningful signal.
# Regexes are tight — generic substrings like "runtime error" can be emitted
# by unrelated failures and would mask a regression.
function(_add_canary name arg regex)
    add_test(NAME ${name} COMMAND sanitizer_canary ${arg})
    vc_register_ctest(${name} sanitizer_canary)
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
vc_add_test(NAME test_tiff)

vc_add_test(NAME test_normalgridtools)

# Live S3: PHerc 0172 sampling and PHerc 0139 catalog/object-store identity.
vc_add_test(NAME test_volume_live_s3
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)
set_property(TEST test_volume_live_s3 APPEND PROPERTY LABELS network)

vc_add_test(NAME test_normal_grid_volume)
vc_add_test(NAME test_normal_grid_generate)

vc_add_test(NAME test_surface_patch_index)

vc_add_test(NAME test_tiff_merge)

vc_add_test(NAME test_global_cache_root)

# Live S3: stages PHerc 0172 level 5 to a local zarr dir, then exercises
# the NormalGridGenerate library API against it. Soft-skips on network.
vc_add_test(NAME test_normal_grid_live)
set_property(TEST test_normal_grid_live APPEND PROPERTY LABELS network)

vc_add_test(NAME test_tiff_more)

vc_add_test(NAME test_geometry)

vc_add_test(NAME test_point_index)

vc_add_test(NAME test_plane_surface)


vc_add_test(NAME test_quadsurface_basics)

vc_add_test(NAME test_load_json)

vc_add_test(NAME test_lasagna_manifest LIBS doctest::doctest vc_lasagna)
vc_add_test(NAME test_lasagna_project_volumes LIBS doctest::doctest vc_lasagna)

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
    LIBS doctest::doctest Qt6::Core vc_core vc_lasagna vc_object_store::vc_objects vc_object_store::vc_object_geometry
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)

vc_add_test(NAME test_lasagna_line_optimizer LIBS doctest::doctest vc_lasagna Ceres::ceres)

vc_add_test(NAME test_lasagna_line_view_surfaces LIBS doctest::doctest vc_lasagna)

vc_add_test(NAME test_lasagna_normal_sampler LIBS doctest::doctest vc_lasagna)

if(TARGET vc_render_tifxyz)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    vc_register_test_target(vc_render_tifxyz)
    add_test(NAME test_render_tifxyz_logging
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_render_tifxyz_logging.py"
            $<TARGET_FILE:vc_render_tifxyz>)
    set_tests_properties(test_render_tifxyz_logging PROPERTIES TIMEOUT 90)
    vc_register_ctest(test_render_tifxyz_logging vc_render_tifxyz)
    add_test(NAME test_render_zarr_output
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_render_zarr_output.py"
            $<TARGET_FILE:vc_render_tifxyz>)
    set_tests_properties(test_render_zarr_output PROPERTIES TIMEOUT 180)
    vc_register_ctest(test_render_zarr_output vc_render_tifxyz)
endif()

vc_add_test(NAME test_fiber_side_strip_intersections LIBS doctest::doctest vc_fiber_tracer)

vc_add_test(NAME test_remote_url)

vc_add_test(NAME test_logging)

vc_add_test(NAME test_plane_surface_branches)

vc_add_test(NAME test_quadsurface_more)

vc_add_test(NAME test_thinning)

vc_add_test(NAME test_umbilicus)

vc_add_test(NAME test_colormaps)

vc_add_test(NAME test_quadsurface_io)

vc_add_test(NAME test_quadsurface_pointto)

vc_add_test(NAME test_quadsurface_snapshot)

vc_add_test(NAME test_quadsurface_save_paths)

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

vc_add_test(NAME test_volume_pkg)
vc_add_test(NAME test_opened_volumes
    SOURCES test_opened_volumes.cpp ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenedVolumes.cpp)
vc_add_test(NAME test_viewer_sampling
    SOURCES test_viewer_sampling.cpp ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenedVolumes.cpp
            ${PROJECT_SOURCE_DIR}/apps/VC3D/volume_viewers/ViewerSampling.cpp
    INCLUDES ${PROJECT_SOURCE_DIR}/apps/VC3D)



# Real PHerc 0172 segments committed under core/test/data/segments/.
vc_add_test(NAME test_quadsurface_fixtures)

vc_add_test(NAME test_volume_pkg_full)

vc_add_test(NAME test_load_quad_from_tifxyz)

vc_add_test(NAME test_surface_patch_index_more)

vc_add_test(NAME test_quadsurface_extras)
vc_add_test(NAME test_quadsurface_store)

vc_add_test(NAME test_volume_pkg_more)
