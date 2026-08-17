# Standalone unit test for SeedingBatchTracker (apps/VC3D/SeedingBatchTracker.*),
# the honest run/expand batch-outcome aggregation extracted from SeedingWidget.
# Deterministic, no process launch, no volume data — safe for the default ctest
# run. Built only under VC_TESTING (wired from the top-level testing block).
#
# The tracker is a small QObject-free class using only QString/QStringList/QSet,
# so we compile its source straight into the test and link just Qt6::Core (for
# QString et al.) + Qt6::Test (the QTest harness). No other VC3D code is pulled
# in, keeping the test fast and self-contained.
find_package(Qt6 REQUIRED COMPONENTS Concurrent Core Test Widgets)

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
    test_fiber_save_batch_tracker.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/LineAnnotationFiberSaveJob.cpp)

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

add_executable(test_command_line_tool_runner
    test_command_line_tool_runner.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandLineToolRunner.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/CommandLineToolRunner.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/ConsoleOutputWidget.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/ConsoleOutputWidget.hpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/ProgressUtil.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/elements/ProgressUtil.hpp)

set_target_properties(test_command_line_tool_runner PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_command_line_tool_runner PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_command_line_tool_runner PRIVATE
    Qt6::Core
    Qt6::Test
    Qt6::Widgets)

add_test(NAME command_line_tool_runner COMMAND test_command_line_tool_runner)
vc_register_test_target(test_command_line_tool_runner)
vc_register_ctest(command_line_tool_runner test_command_line_tool_runner)
set_tests_properties(command_line_tool_runner PROPERTIES
    ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

add_executable(test_annotation_frame
    test_annotation_frame.cpp
    ${PROJECT_SOURCE_DIR}/apps/VC3D/AnnotationFrame.hpp)

set_target_properties(test_annotation_frame PROPERTIES
    AUTOMOC ON AUTOUIC OFF AUTORCC OFF)

target_include_directories(test_annotation_frame PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/VC3D)

target_link_libraries(test_annotation_frame PRIVATE
    Qt6::Core
    Qt6::Test)

add_test(NAME annotation_frame COMMAND test_annotation_frame)
vc_register_test_target(test_annotation_frame)
vc_register_ctest(annotation_frame test_annotation_frame)
