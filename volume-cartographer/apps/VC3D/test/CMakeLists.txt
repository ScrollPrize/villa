find_package(Qt6 REQUIRED COMPONENTS Concurrent Core Network Test Widgets)

add_executable(test_volume_path_labels test_volume_path_labels.cpp
    ../elements/VolumeSelector.cpp ../elements/VolumeSelector.hpp)
set_target_properties(test_volume_path_labels PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_volume_path_labels PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_volume_path_labels PRIVATE Qt6::Widgets Qt6::Test)
add_test(NAME volume_path_labels COMMAND test_volume_path_labels)
set_tests_properties(volume_path_labels PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
vc_register_test_target(test_volume_path_labels)
vc_register_ctest(volume_path_labels test_volume_path_labels)

add_executable(test_surface_render_policy test_surface_render_policy.cpp)
set_target_properties(test_surface_render_policy PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_surface_render_policy PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_surface_render_policy PRIVATE Qt6::Core Qt6::Test)
add_test(NAME surface_render_policy COMMAND test_surface_render_policy)
vc_register_test_target(test_surface_render_policy)
vc_register_ctest(surface_render_policy test_surface_render_policy)

add_executable(test_catalog_project_dialog
    test_catalog_project_dialog.cpp
    ../OpenDataCatalogWindow.cpp
    ../OpenDataManifest.cpp
    ../OpenDataSampleProject.cpp
    ../OpenDataLasagna.cpp
    ../OpenDataNormalGrids.cpp)
set_target_properties(test_catalog_project_dialog PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_catalog_project_dialog PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D ${PROJECT_SOURCE_DIR}/core/test)
target_link_libraries(test_catalog_project_dialog PRIVATE
    Qt6::Core Qt6::Widgets Qt6::Test Qt6::Concurrent vc_core vc_lasagna vc_object_store::vc_objects vc_object_store::vc_object_geometry)
add_test(NAME catalog_project_dialog COMMAND test_catalog_project_dialog)
vc_register_test_target(test_catalog_project_dialog)
vc_register_ctest(catalog_project_dialog test_catalog_project_dialog)
set_tests_properties(catalog_project_dialog PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 30)

add_executable(test_spiral_input_draft test_spiral_input_draft.cpp)
set_target_properties(test_spiral_input_draft PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_spiral_input_draft PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_spiral_input_draft PRIVATE Qt6::Core Qt6::Test)
add_test(NAME spiral_input_draft COMMAND test_spiral_input_draft)
vc_register_test_target(test_spiral_input_draft)
vc_register_ctest(spiral_input_draft test_spiral_input_draft)

add_executable(test_settings_filter
    test_settings_filter.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SettingsFilter.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SettingsFilter.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/VCSettings.ui)
# The test also filters the real VCSettings.ui form.
set_target_properties(test_settings_filter PROPERTIES
    AUTOMOC ON AUTOUIC ON AUTORCC OFF
    AUTOUIC_SEARCH_PATHS ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_include_directories(test_settings_filter PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_settings_filter PRIVATE
    Qt6::Core Qt6::Gui Qt6::Test Qt6::Widgets)
add_test(NAME settings_filter COMMAND test_settings_filter)
vc_register_test_target(test_settings_filter)
vc_register_ctest(settings_filter test_settings_filter)
set_tests_properties(settings_filter PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_line_annotation_coordinate_scale
    test_line_annotation_coordinate_scale.cpp)

set_target_properties(test_line_annotation_coordinate_scale PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_line_annotation_coordinate_scale PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_line_annotation_coordinate_scale PRIVATE
    vc_lasagna
    Qt6::Core
    Qt6::Test)

add_test(NAME line_annotation_coordinate_scale
    COMMAND test_line_annotation_coordinate_scale)
vc_register_test_target(test_line_annotation_coordinate_scale)
vc_register_ctest(line_annotation_coordinate_scale
    test_line_annotation_coordinate_scale)

add_executable(test_command_catalog
    test_command_catalog.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandCatalog.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandCatalog.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/Keybinds.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/Keybinds.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/StatusDockPanelHost.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/StatusDockPanelHost.hpp)
set_target_properties(test_command_catalog PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_command_catalog PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)
# StatusDockPanelHost reads the VC3D settings through vc_core.
target_link_libraries(test_command_catalog PRIVATE
    vc_core Qt6::Core Qt6::Gui Qt6::Test Qt6::Widgets)
add_test(NAME command_catalog COMMAND test_command_catalog)
vc_register_test_target(test_command_catalog)
vc_register_ctest(command_catalog test_command_catalog)
set_tests_properties(command_catalog PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_command_palette
    test_command_palette.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandPalette.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandPalette.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandCatalog.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandCatalog.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/Keybinds.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/Keybinds.hpp)
set_target_properties(test_command_palette PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
# VCSettings.hpp includes a vc_core header.
target_include_directories(test_command_palette PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D
    ${PROJECT_SOURCE_DIR}/core/include)
target_link_libraries(test_command_palette PRIVATE
    Qt6::Core Qt6::Gui Qt6::Test Qt6::Widgets)
add_test(NAME command_palette COMMAND test_command_palette)
vc_register_test_target(test_command_palette)
vc_register_ctest(command_palette test_command_palette)
set_tests_properties(command_palette PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_surface_rotation_transform
    test_surface_rotation_transform.cpp)

target_include_directories(test_surface_rotation_transform PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_surface_rotation_transform PRIVATE
    Qt6::Core
    Qt6::Test)

set_target_properties(test_surface_rotation_transform PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

add_test(NAME surface_rotation_transform COMMAND test_surface_rotation_transform)
vc_register_test_target(test_surface_rotation_transform)
vc_register_ctest(surface_rotation_transform test_surface_rotation_transform)

add_executable(test_keybinds_sheet
    test_keybinds_sheet.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/KeybindsSheet.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/KeybindsSheet.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandCatalog.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandCatalog.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SettingsFilter.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SettingsFilter.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/Keybinds.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/Keybinds.hpp)
set_target_properties(test_keybinds_sheet PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_keybinds_sheet PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)
# The test reads the keybind definitions to check that the help lists them.
target_compile_definitions(test_keybinds_sheet PRIVATE
    VC3D_KEYBINDS_SOURCE="${PROJECT_SOURCE_DIR}/apps/VC3D/Keybinds.cpp")
target_link_libraries(test_keybinds_sheet PRIVATE
    Qt6::Core Qt6::Gui Qt6::Test Qt6::Widgets)
add_test(NAME keybinds_sheet COMMAND test_keybinds_sheet)
vc_register_test_target(test_keybinds_sheet)
vc_register_ctest(keybinds_sheet test_keybinds_sheet)
set_tests_properties(keybinds_sheet PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_vc_settings
    test_vc_settings.cpp)

set_target_properties(test_vc_settings PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_vc_settings PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_vc_settings PRIVATE
    vc_core
    Qt6::Core
    Qt6::Test)

add_test(NAME vc_settings COMMAND test_vc_settings)
vc_register_test_target(test_vc_settings)
vc_register_ctest(vc_settings test_vc_settings)

add_executable(test_theme_engine
    test_theme_engine.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/themes/ClassicTheme.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/themes/ClassicTheme.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/themes/ThemeEngine.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/themes/ThemeEngine.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/themes/ThemeManager.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/themes/ThemeManager.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/themes/themes.qrc)

set_target_properties(test_theme_engine PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC ON)

target_include_directories(test_theme_engine PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D
    ${PROJECT_SOURCE_DIR}/core/include)

target_compile_definitions(test_theme_engine PRIVATE
    VC3D_THEME_EXAMPLES_DIR="${PROJECT_SOURCE_DIR}/apps/VC3D/themes/examples"
    VC3D_THEME_DOCS="${PROJECT_SOURCE_DIR}/docs/vc3d_themes.md")

target_link_libraries(test_theme_engine PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets)

add_test(NAME theme_engine COMMAND test_theme_engine)
vc_register_test_target(test_theme_engine)
vc_register_ctest(theme_engine test_theme_engine)
set_tests_properties(theme_engine PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 60)

add_executable(test_unified_browser_dialog
    test_unified_browser_dialog.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/UnifiedBrowserDialog.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/UnifiedBrowserDialog.hpp)

set_target_properties(test_unified_browser_dialog PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_unified_browser_dialog PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_unified_browser_dialog PRIVATE
    vc_core
    Qt6::Concurrent
    Qt6::Core
    Qt6::Test
    Qt6::Widgets)

add_test(NAME unified_browser_dialog COMMAND test_unified_browser_dialog)
vc_register_test_target(test_unified_browser_dialog)
vc_register_ctest(unified_browser_dialog test_unified_browser_dialog)
set_tests_properties(unified_browser_dialog PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_spiral_reload_comparison
    test_spiral_reload_comparison.cpp)

set_target_properties(test_spiral_reload_comparison PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_spiral_reload_comparison PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_spiral_reload_comparison PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME spiral_reload_comparison COMMAND test_spiral_reload_comparison)
vc_register_test_target(test_spiral_reload_comparison)
vc_register_ctest(spiral_reload_comparison test_spiral_reload_comparison)



add_executable(test_spiral_artifact_pins
    test_spiral_artifact_pins.cpp)

set_target_properties(test_spiral_artifact_pins PROPERTIES
    AUTOMOC ON)

target_include_directories(test_spiral_artifact_pins PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_spiral_artifact_pins PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME spiral_artifact_pins COMMAND test_spiral_artifact_pins)
vc_register_test_target(test_spiral_artifact_pins)
vc_register_ctest(spiral_artifact_pins test_spiral_artifact_pins)

add_executable(test_spiral_point_placement_mode
    test_spiral_point_placement_mode.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SpiralBrushCursorWidget.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SpiralBrushCursorWidget.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SpiralPointPlacementMode.hpp)

set_target_properties(test_spiral_point_placement_mode PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_spiral_point_placement_mode PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_spiral_point_placement_mode PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
    Qt6::Widgets)

add_test(NAME spiral_point_placement_mode COMMAND test_spiral_point_placement_mode)
vc_register_test_target(test_spiral_point_placement_mode)
vc_register_ctest(spiral_point_placement_mode test_spiral_point_placement_mode)
set_tests_properties(spiral_point_placement_mode PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_spiral_point_collection_edit
    test_spiral_point_collection_edit.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SpiralPointCollectionEdit.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SpiralPointCollectionEdit.hpp)

set_target_properties(test_spiral_point_collection_edit PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_spiral_point_collection_edit PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_spiral_point_collection_edit PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test
    opencv_core)

add_test(NAME spiral_point_collection_edit COMMAND test_spiral_point_collection_edit)
vc_register_test_target(test_spiral_point_collection_edit)
vc_register_ctest(spiral_point_collection_edit test_spiral_point_collection_edit)
set_tests_properties(spiral_point_collection_edit PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_spiral_preview_provenance
    test_spiral_preview_provenance.cpp)

set_target_properties(test_spiral_preview_provenance PROPERTIES
    AUTOMOC ON)

target_include_directories(test_spiral_preview_provenance PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_spiral_preview_provenance PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME spiral_preview_provenance COMMAND test_spiral_preview_provenance)
vc_register_test_target(test_spiral_preview_provenance)
vc_register_ctest(spiral_preview_provenance test_spiral_preview_provenance)

add_executable(test_download_queue_stats
    test_download_queue_stats.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/DownloadQueueStats.cpp)

set_target_properties(test_download_queue_stats PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_download_queue_stats PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_download_queue_stats PRIVATE
    vc_core
    Qt6::Core
    Qt6::Test)

add_test(NAME download_queue_stats COMMAND test_download_queue_stats)
vc_register_test_target(test_download_queue_stats)
vc_register_ctest(download_queue_stats test_download_queue_stats)

add_executable(test_download_queue_debug
    test_download_queue_debug.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/DownloadQueueDebugOverlay.cpp)

set_target_properties(test_download_queue_debug PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_download_queue_debug PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_download_queue_debug PRIVATE
    vc_core
    Qt6::Core
    Qt6::Gui
    Qt6::Test)

add_test(NAME download_queue_debug COMMAND test_download_queue_debug)
vc_register_test_target(test_download_queue_debug)
vc_register_ctest(download_queue_debug test_download_queue_debug)
set_tests_properties(download_queue_debug PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_seeding_batch_tracker
    test_seeding_batch_tracker.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/SeedingBatchTracker.cpp)

# The test class itself uses Q_OBJECT (QtTest) and is MOC'd via the in-file
# #include "..._tracker.moc"; SeedingBatchTracker has no Q_OBJECT so AUTOMOC has
# nothing else to do. Keep AUTOMOC ON so QtTest's moc runs.
set_target_properties(test_seeding_batch_tracker PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_seeding_batch_tracker PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_seeding_batch_tracker PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME seeding_batch_tracker COMMAND test_seeding_batch_tracker)
vc_register_test_target(test_seeding_batch_tracker)
vc_register_ctest(seeding_batch_tracker test_seeding_batch_tracker)

add_executable(test_fiber_save_batch_tracker
    test_fiber_save_batch_tracker.cpp)

set_target_properties(test_fiber_save_batch_tracker PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_save_batch_tracker PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_fiber_save_batch_tracker PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME fiber_save_batch_tracker COMMAND test_fiber_save_batch_tracker)
vc_register_test_target(test_fiber_save_batch_tracker)
vc_register_ctest(fiber_save_batch_tracker test_fiber_save_batch_tracker)



add_executable(test_fiber_network_layout
    test_fiber_network_layout.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNetworkLayout.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNetworkLayout.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.hpp)

set_target_properties(test_fiber_network_layout PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_network_layout PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D
    ${PROJECT_SOURCE_DIR}/core/include
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(test_fiber_network_layout PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Test)

add_test(NAME fiber_network_layout COMMAND test_fiber_network_layout)
vc_register_test_target(test_fiber_network_layout)
vc_register_ctest(fiber_network_layout test_fiber_network_layout)

add_executable(test_command_line_tool_runner
    test_command_line_tool_runner.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandLineToolRunner.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandLineToolRunner.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/ToolExecutable.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/ToolProcess.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/ToolProcess.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/ConsoleOutputWidget.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/ConsoleOutputWidget.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/ProgressUtil.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/ProgressUtil.hpp)

set_target_properties(test_command_line_tool_runner PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_command_line_tool_runner PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_compile_definitions(test_command_line_tool_runner PRIVATE VC_TESTING)

target_link_libraries(test_command_line_tool_runner PRIVATE
    Qt6::Core
    Qt6::Test
    Qt6::Widgets
    vc_core)

add_test(NAME command_line_tool_runner COMMAND test_command_line_tool_runner)
vc_register_test_target(test_command_line_tool_runner)
vc_register_ctest(command_line_tool_runner test_command_line_tool_runner)
set_tests_properties(command_line_tool_runner PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_open_data_volume_orientation
    test_open_data_volume_orientation.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataVolumeOrientation.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataVolumeOrientation.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataManifest.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/OpenDataManifest.hpp)

set_target_properties(test_open_data_volume_orientation PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_open_data_volume_orientation PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_open_data_volume_orientation PRIVATE
    vc_core
    Qt6::Core
    Qt6::Test)

add_test(NAME open_data_volume_orientation COMMAND test_open_data_volume_orientation)
vc_register_test_target(test_open_data_volume_orientation)
vc_register_ctest(open_data_volume_orientation test_open_data_volume_orientation)

add_executable(test_fiber_map_staleness
    test_fiber_map_staleness.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapStaleness.hpp)

set_target_properties(test_fiber_map_staleness PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_map_staleness PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_fiber_map_staleness PRIVATE
    vc_core
    Qt6::Core
    Qt6::Test)

add_test(NAME fiber_map_staleness COMMAND test_fiber_map_staleness)
vc_register_test_target(test_fiber_map_staleness)
vc_register_ctest(fiber_map_staleness test_fiber_map_staleness)

add_executable(test_fiber_winding_solver
    test_fiber_winding_solver.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.hpp)

set_target_properties(test_fiber_winding_solver PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_winding_solver PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_fiber_winding_solver PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME fiber_winding_solver COMMAND test_fiber_winding_solver)
vc_register_test_target(test_fiber_winding_solver)
vc_register_ctest(fiber_winding_solver test_fiber_winding_solver)

add_executable(test_fiber_global_layout
    test_fiber_global_layout.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNetworkLayout.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNetworkLayout.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.hpp)

set_target_properties(test_fiber_global_layout PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_global_layout PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(test_fiber_global_layout PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME fiber_global_layout COMMAND test_fiber_global_layout)
vc_register_test_target(test_fiber_global_layout)
vc_register_ctest(fiber_global_layout test_fiber_global_layout)

add_executable(test_fiber_map_auto_link
    test_fiber_map_auto_link.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapAutoLink.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapAutoLink.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNetworkLayout.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNetworkLayout.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.hpp)

set_target_properties(test_fiber_map_auto_link PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_map_auto_link PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(test_fiber_map_auto_link PRIVATE
    Qt6::Core
    Qt6::Test
    opencv_core)

add_test(NAME fiber_map_auto_link COMMAND test_fiber_map_auto_link)
vc_register_test_target(test_fiber_map_auto_link)
vc_register_ctest(fiber_map_auto_link test_fiber_map_auto_link)

add_executable(test_fiber_map_gap_field
    test_fiber_map_gap_field.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapGapField.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapGapField.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNetworkLayout.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberNetworkLayout.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberWindingSolver.hpp)

set_target_properties(test_fiber_map_gap_field PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_map_gap_field PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(test_fiber_map_gap_field PRIVATE
    Qt6::Core
    Qt6::Test
    opencv_core
    opencv_imgproc)

add_test(NAME fiber_map_gap_field COMMAND test_fiber_map_gap_field)
vc_register_test_target(test_fiber_map_gap_field)
vc_register_ctest(fiber_map_gap_field test_fiber_map_gap_field)

add_executable(test_fiber_map_rebuild_queue
    test_fiber_map_rebuild_queue.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapRebuildQueue.hpp)

set_target_properties(test_fiber_map_rebuild_queue PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_map_rebuild_queue PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_fiber_map_rebuild_queue PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME fiber_map_rebuild_queue COMMAND test_fiber_map_rebuild_queue)
vc_register_test_target(test_fiber_map_rebuild_queue)
vc_register_ctest(fiber_map_rebuild_queue test_fiber_map_rebuild_queue)

add_executable(test_fiber_map_ruler_math
    test_fiber_map_ruler_math.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapRulerMath.hpp)

set_target_properties(test_fiber_map_ruler_math PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_map_ruler_math PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_fiber_map_ruler_math PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME fiber_map_ruler_math COMMAND test_fiber_map_ruler_math)
vc_register_test_target(test_fiber_map_ruler_math)
vc_register_ctest(fiber_map_ruler_math test_fiber_map_ruler_math)

add_executable(test_fiber_length_display
    test_fiber_length_display.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberLengthDisplay.hpp)

set_target_properties(test_fiber_length_display PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_length_display PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_fiber_length_display PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME fiber_length_display COMMAND test_fiber_length_display)
vc_register_test_target(test_fiber_length_display)
vc_register_ctest(fiber_length_display test_fiber_length_display)

add_executable(test_fiber_map_kollesis
    test_fiber_map_kollesis.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapKollesis.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/FiberMapKollesis.hpp)

set_target_properties(test_fiber_map_kollesis PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_fiber_map_kollesis PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_fiber_map_kollesis PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME fiber_map_kollesis COMMAND test_fiber_map_kollesis)
vc_register_test_target(test_fiber_map_kollesis)
vc_register_ctest(fiber_map_kollesis test_fiber_map_kollesis)

add_executable(test_line_annotation_fiber_deletion
    test_line_annotation_fiber_deletion.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberDeletion.hpp)

set_target_properties(test_line_annotation_fiber_deletion PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_line_annotation_fiber_deletion PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_line_annotation_fiber_deletion PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME line_annotation_fiber_deletion COMMAND test_line_annotation_fiber_deletion)
vc_register_test_target(test_line_annotation_fiber_deletion)
vc_register_ctest(line_annotation_fiber_deletion test_line_annotation_fiber_deletion)

add_executable(test_line_annotation_optimization_queue
    test_line_annotation_optimization_queue.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationOptimizationQueue.hpp)

set_target_properties(test_line_annotation_optimization_queue PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_line_annotation_optimization_queue PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_line_annotation_optimization_queue PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME line_annotation_optimization_queue
    COMMAND test_line_annotation_optimization_queue)
vc_register_test_target(test_line_annotation_optimization_queue)
vc_register_ctest(line_annotation_optimization_queue
    test_line_annotation_optimization_queue)

add_executable(test_spiral_input_workflow
    test_spiral_input_workflow.cpp
    ../SpiralServiceManager.cpp ../SpiralServiceManager.hpp
    ../SpiralServiceManagerInputs.cpp
    ../SpiralArtifactCache.cpp ../SpiralArtifactCache.hpp
    ../SpiralSshTunnel.cpp ../SpiralSshTunnel.hpp)
set_target_properties(test_spiral_input_workflow PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_spiral_input_workflow PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_compile_definitions(test_spiral_input_workflow PRIVATE
    SPIRAL_CLIENT_SERVICE_FIXTURE="${PROJECT_SOURCE_DIR}/../spiral-fitting/tests/revisioned_client_service.py")
target_link_libraries(test_spiral_input_workflow PRIVATE vc_core Qt6::Core Qt6::Concurrent Qt6::Network Qt6::Widgets Qt6::Test)
add_test(NAME spiral_input_workflow COMMAND test_spiral_input_workflow)
vc_register_test_target(test_spiral_input_workflow)
vc_register_ctest(spiral_input_workflow test_spiral_input_workflow)
set_tests_properties(spiral_input_workflow PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
add_executable(test_spiral_brush_patch test_spiral_brush_patch.cpp)
set_target_properties(test_spiral_brush_patch PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_spiral_brush_patch PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_spiral_brush_patch PRIVATE vc_core opencv_core opencv_imgproc opencv_imgcodecs Qt6::Core Qt6::Gui Qt6::Test Qt6::Widgets)
add_test(NAME spiral_brush_patch COMMAND test_spiral_brush_patch)
vc_register_test_target(test_spiral_brush_patch)
vc_register_ctest(spiral_brush_patch test_spiral_brush_patch)
set_tests_properties(spiral_brush_patch PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

# Exercise the production controller publication paths after Qt/state cleanup.
# The VC3D app target only exists with VC_BUILD_APPS; CI runs these tests from
# the VC3D compile-and-smoke job.
if(TARGET VC3D)
    add_test(NAME object_store_vc3d_controller COMMAND VC3D --test-object-store)
    set_tests_properties(object_store_vc3d_controller PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 90)
    vc_register_test_target(VC3D)
    vc_register_ctest(object_store_vc3d_controller VC3D)
    add_test(NAME surface_panel_store COMMAND VC3D --test-surface-panel)
    set_tests_properties(surface_panel_store PROPERTIES
        ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 90)
    vc_register_ctest(surface_panel_store VC3D)
endif()

add_executable(test_surface_catalog test_surface_catalog.cpp ../SurfaceCatalog.cpp ../SurfaceCatalog.hpp
    ../segmentation/SurfaceStoreEdit.cpp ../segmentation/SurfaceStoreEdit.hpp
    ../segmentation/SurfaceCorrections.cpp ../segmentation/SurfaceCorrections.hpp
    ../OpenDataManifest.cpp ../VolumeRegistrations.cpp)
set_target_properties(test_surface_catalog PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_surface_catalog PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D ${PROJECT_SOURCE_DIR}/core/test)
target_link_libraries(test_surface_catalog PRIVATE vc_core Qt6::Core Qt6::Concurrent)
add_test(NAME surface_catalog COMMAND test_surface_catalog)
vc_register_test_target(test_surface_catalog)
vc_register_ctest(surface_catalog test_surface_catalog)
# Read-only 3D geometry adapters: source indices, frames, tags, and clipping.
add_executable(test_fiber_3d_data test_fiber_3d_data.cpp
    ../fibers3d/Fiber3DData.cpp)
set_target_properties(test_fiber_3d_data PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_fiber_3d_data PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_fiber_3d_data PRIVATE
    Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Test vc_core vc_fiber_view)
add_test(NAME fiber_3d_data COMMAND test_fiber_3d_data)
vc_register_test_target(test_fiber_3d_data)
vc_register_ctest(fiber_3d_data test_fiber_3d_data)

add_executable(test_fiber_3d_geometry test_fiber_3d_geometry.cpp)
set_target_properties(test_fiber_3d_geometry PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_fiber_3d_geometry PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_fiber_3d_geometry PRIVATE
    Qt6::Core Qt6::Gui Qt6::Test vc_core vc_fiber_view opencv_core)
add_test(NAME fiber_3d_geometry COMMAND test_fiber_3d_geometry)
vc_register_test_target(test_fiber_3d_geometry)
vc_register_ctest(fiber_3d_geometry test_fiber_3d_geometry)

add_executable(test_fiber_3d_ribbon test_fiber_3d_ribbon.cpp ../fibers3d/Fiber3DRibbon.cpp ../fibers3d/Fiber3DCT.cpp)
set_target_properties(test_fiber_3d_ribbon PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_fiber_3d_ribbon PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_fiber_3d_ribbon PRIVATE Qt6::Core Qt6::Gui Qt6::Test vc_core vc_fiber_view vc_lasagna)
add_test(NAME fiber_3d_ribbon COMMAND test_fiber_3d_ribbon)
vc_register_test_target(test_fiber_3d_ribbon)
vc_register_ctest(fiber_3d_ribbon test_fiber_3d_ribbon)

add_executable(test_fiber_3d_slices test_fiber_3d_slices.cpp ../fibers3d/Fiber3DSlices.cpp ../fibers3d/Fiber3DCT.cpp)
set_target_properties(test_fiber_3d_slices PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_fiber_3d_slices PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_fiber_3d_slices PRIVATE Qt6::Core Qt6::Gui Qt6::Test vc_core vc_fiber_view)
add_test(NAME fiber_3d_slices COMMAND test_fiber_3d_slices)
vc_register_test_target(test_fiber_3d_slices)
vc_register_ctest(fiber_3d_slices test_fiber_3d_slices)

add_executable(test_fiber_3d_volume test_fiber_3d_volume.cpp ../fibers3d/Fiber3DVolumeRenderer.cpp ../volume_viewers/VolumetricCompositor.cpp)
set_target_properties(test_fiber_3d_volume PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_fiber_3d_volume PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_fiber_3d_volume PRIVATE Qt6::Core Qt6::Gui Qt6::OpenGL Qt6::Test vc_core vc_fiber_view)
add_test(NAME fiber_3d_volume COMMAND test_fiber_3d_volume)
vc_register_test_target(test_fiber_3d_volume)
vc_register_ctest(fiber_3d_volume test_fiber_3d_volume)

add_executable(test_fiber_3d_surface test_fiber_3d_surface.cpp ../fibers3d/Fiber3DSurfaceMesh.cpp ../fibers3d/Fiber3DSurfaceRenderer.cpp ../fibers3d/Fiber3DSurfaceProximity.cpp)
set_target_properties(test_fiber_3d_surface PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_fiber_3d_surface PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_fiber_3d_surface PRIVATE Qt6::Core Qt6::Gui Qt6::OpenGL Qt6::Test vc_core vc_fiber_view)
add_test(NAME fiber_3d_surface COMMAND test_fiber_3d_surface)
vc_register_test_target(test_fiber_3d_surface)
vc_register_ctest(fiber_3d_surface test_fiber_3d_surface)

add_executable(test_fiber_3d_canvas test_fiber_3d_canvas.cpp
    ../fibers3d/Fiber3DCanvas.cpp
    ../fibers3d/Fiber3DSlices.cpp ../fibers3d/Fiber3DCT.cpp
    ../fibers3d/Fiber3DVolumeRenderer.cpp ../volume_viewers/VolumetricCompositor.cpp
    ../fibers3d/Fiber3DSurfaceRenderer.cpp ../fibers3d/Fiber3DSurfaceProximity.cpp)
set_target_properties(test_fiber_3d_canvas PROPERTIES AUTOMOC ON AUTOUIC OFF AUTORCC OFF)
target_include_directories(test_fiber_3d_canvas PRIVATE ${PROJECT_SOURCE_DIR}/apps/VC3D)
target_link_libraries(test_fiber_3d_canvas PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets
    Qt6::OpenGLWidgets Qt6::Concurrent Qt6::Test vc_core vc_fiber_view)
add_test(NAME fiber_3d_canvas COMMAND test_fiber_3d_canvas)
vc_register_test_target(test_fiber_3d_canvas)
vc_register_ctest(fiber_3d_canvas test_fiber_3d_canvas)
