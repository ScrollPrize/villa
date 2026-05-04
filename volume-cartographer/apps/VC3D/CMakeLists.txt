set(CMAKE_AUTORCC ON)

add_definitions(-DCHUNK_CACHE_SIZE_GB=${CHUNK_CACHE_SIZE_GB})

set(srcs
    VCAppMain.cpp
    CState.cpp
    CState.hpp
    Keybinds.cpp
    Keybinds.hpp
    CWindow.cpp
    CWindowContextMenu.cpp
    FileWatcherService.cpp
    FileWatcherService.hpp
    AxisAlignedSliceController.cpp
    AxisAlignedSliceController.hpp
    SurfaceAffineTransformController.cpp
    SurfaceAffineTransformController.hpp
    SurfaceAreaCalculator.cpp
    SurfaceAreaCalculator.hpp
    SegmentationCommandHandler.cpp
    SegmentationCommandHandler.hpp
    volume_viewers/CVolumeViewerView.cpp
    VolumeViewerCmaps.cpp
    VolumeViewerCmaps.hpp
    WindowRangeWidget.cpp
    WindowRangeWidget.hpp
    elements/CollapsibleSettingsGroup.cpp
    elements/CollapsibleSettingsGroup.hpp
    elements/DropdownChecklistButton.cpp
    elements/DropdownChecklistButton.hpp
    elements/JsonProfileEditor.cpp
    elements/JsonProfileEditor.hpp
    elements/LabeledControlRow.cpp
    elements/LabeledControlRow.hpp
    elements/VolumeSelector.cpp
    elements/VolumeSelector.hpp
    elements/RangeSlider.cpp
    elements/RangeSlider.hpp
    elements/ViewerStatsBar.cpp
    elements/ViewerStatsBar.hpp
    SettingsDialog.cpp
    SurfaceTreeWidget.hpp
    SurfaceTreeWidget.cpp
    SeedingWidget.cpp
    SeedingWidget.hpp
    segmentation/SegmentationCommon.hpp
    segmentation/SegmentationWidget.cpp
    segmentation/SegmentationWidget.hpp
    segmentation/panels/SegmentationHeaderRow.cpp
    segmentation/panels/SegmentationHeaderRow.hpp
    segmentation/panels/SegmentationEditingPanel.cpp
    segmentation/panels/SegmentationEditingPanel.hpp
    segmentation/panels/SegmentationGrowthPanel.cpp
    segmentation/panels/SegmentationGrowthPanel.hpp
    segmentation/panels/SegmentationCorrectionsPanel.cpp
    segmentation/panels/SegmentationCorrectionsPanel.hpp
    segmentation/panels/SegmentationCustomParamsPanel.cpp
    segmentation/panels/SegmentationCustomParamsPanel.hpp
    segmentation/panels/SegmentationApprovalMaskPanel.cpp
    segmentation/panels/SegmentationApprovalMaskPanel.hpp
    segmentation/panels/SegmentationNeuralTracerPanel.cpp
    segmentation/panels/SegmentationNeuralTracerPanel.hpp
    segmentation/panels/SegmentationDirectionFieldPanel.cpp
    segmentation/panels/SegmentationDirectionFieldPanel.hpp
    segmentation/panels/SegmentationManualAddPanel.cpp
    segmentation/panels/SegmentationManualAddPanel.hpp
    segmentation/growth/SegmentationGrowth.cpp
    segmentation/growth/SegmentationGrower.cpp
    segmentation/growth/SegmentationGrower.hpp
    segmentation/tools/SegmentationEditManager.cpp
    segmentation/tools/SegmentationEditManager.hpp
    segmentation/tools/ApprovalMaskBrushTool.cpp
    segmentation/tools/ApprovalMaskBrushTool.hpp
    segmentation/tools/SurfaceMaskBrushTool.cpp
    segmentation/tools/SurfaceMaskBrushTool.hpp
    segmentation/tools/ManualAddTool.cpp
    segmentation/tools/ManualAddTool.hpp
    segmentation/tools/ThinPlateSpline3d.cpp
    segmentation/tools/ThinPlateSpline3d.hpp
    segmentation/tools/SegmentationLineTool.cpp
    segmentation/tools/SegmentationLineTool.hpp
    segmentation/tools/SegmentationPushPullTool.cpp
    segmentation/tools/SegmentationPushPullTool.hpp
    segmentation/tools/SegmentationTool.hpp
    segmentation/tools/SegmentationBrushTool.cpp
    segmentation/tools/SegmentationBrushTool.hpp
    segmentation/SegmentationModule_Falloff.cpp
    segmentation/SegmentationModule_Input.cpp
    segmentation/SegmentationModule_Session.cpp
    overlays/SegmentationOverlayController.cpp
    overlays/SegmentationOverlayController.hpp
    overlays/PointsOverlayController.cpp
    overlays/PointsOverlayController.hpp
    overlays/RawPointsOverlayController.cpp
    overlays/RawPointsOverlayController.hpp
    overlays/BBoxOverlayController.cpp
    overlays/BBoxOverlayController.hpp
    overlays/PathsOverlayController.cpp
    overlays/PathsOverlayController.hpp
    overlays/VectorOverlayController.cpp
    overlays/VectorOverlayController.hpp
    overlays/PlaneSlicingOverlayController.cpp
    overlays/PlaneSlicingOverlayController.hpp
    overlays/SurfaceRotationOverlayController.cpp
    overlays/SurfaceRotationOverlayController.hpp
    overlays/VolumeOverlayController.cpp
    overlays/VolumeOverlayController.hpp
    overlays/ViewerOverlayControllerBase.cpp
    overlays/ViewerOverlayControllerBase.hpp
    SurfacePanelController.cpp
    SurfacePanelController.hpp
    MenuActionController.cpp
    MenuActionController.hpp
    UnifiedBrowserDialog.cpp
    UnifiedBrowserDialog.hpp
    segmentation/SegmentationModule.cpp
    segmentation/SegmentationModule.hpp
    segmentation/growth/SegmentationCorrections.cpp
    segmentation/growth/SegmentationCorrections.hpp
    segmentation/SegmentationUndoHistory.hpp
    NeuralTraceServiceManager.cpp
    NeuralTraceServiceManager.hpp
    LasagnaServiceManager.cpp
    LasagnaServiceManager.hpp
    segmentation/panels/SegmentationLasagnaPanel.cpp
    segmentation/panels/SegmentationLasagnaPanel.hpp
    ViewerManager.cpp
    ViewerManager.hpp
    viewer_controls/ViewerControlsPanel.cpp
    viewer_controls/ViewerControlsPanel.hpp
    viewer_controls/panels/ViewerCompositePanel.cpp
    viewer_controls/panels/ViewerCompositePanel.hpp
    viewer_controls/panels/ViewerNavigationPanel.cpp
    viewer_controls/panels/ViewerNavigationPanel.hpp
    viewer_controls/panels/ViewerNormalVisualizationPanel.cpp
    viewer_controls/panels/ViewerNormalVisualizationPanel.hpp
    viewer_controls/panels/ViewerPostprocessingPanel.cpp
    viewer_controls/panels/ViewerPostprocessingPanel.hpp
    viewer_controls/panels/ViewerPreprocessingPanel.cpp
    viewer_controls/panels/ViewerPreprocessingPanel.hpp
    viewer_controls/panels/ViewerTransformsPanel.cpp
    viewer_controls/panels/ViewerTransformsPanel.hpp
    viewer_controls/panels/ViewerViewExtrasPanel.cpp
    viewer_controls/panels/ViewerViewExtrasPanel.hpp
    SegmentRenderThread.cpp
    SegmentRenderThread.hpp
    elements/ProgressUtil.cpp
    elements/ProgressUtil.hpp
    CommandLineToolRunner.cpp
    CommandLineToolRunner.hpp
    ToolDialogs.cpp
    ToolDialogs.hpp
    ConsoleOutputWidget.cpp
    ConsoleOutputWidget.hpp
    CPointCollectionWidget.cpp
    CFiberWidget.cpp
    CFiberWidget.hpp
    FiberAnnotationController.cpp
    FiberAnnotationController.hpp
    elements/COutlinedTextItem.hpp
    elements/COutlinedTextItem.cpp
    volume_viewers/CChunkedVolumeViewer.cpp
    volume_viewers/CChunkedVolumeViewer.hpp
)

if(VC_ENABLE_TSAN OR VC_ENABLE_LSAN OR VC_ENABLE_UBSAN OR VC_ENABLE_ASAN)
    list(APPEND srcs "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.c")
endif()


add_executable(VC3D ${srcs} resources.qrc)

target_include_directories(VC3D
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/core/src
        ${CMAKE_SOURCE_DIR}/libs/edt
        ${CMAKE_SOURCE_DIR}/libs/cc3d
        ${CMAKE_SOURCE_DIR}/libs/djikstra3d
)

# Avahi for mDNS service discovery (Linux only)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(AVAHI IMPORTED_TARGET avahi-client)
    endif()
endif()

if(AVAHI_FOUND)
    target_compile_definitions(VC3D PRIVATE VC_HAVE_AVAHI=1)
else()
    target_compile_definitions(VC3D PRIVATE VC_HAVE_AVAHI=0)
endif()

find_package(mimalloc QUIET)
if(mimalloc_FOUND)
    target_compile_definitions(VC3D PRIVATE VC_HAVE_MIMALLOC=1)
endif()

target_link_libraries(VC3D
    # mimalloc first so its symbols win malloc/free resolution.
    $<$<BOOL:${mimalloc_FOUND}>:mimalloc>
    vc_core
    vc_ui
    vc_tracer
    vc_flattening
    Ceres::ceres
    Qt6::Core
    Qt6::Concurrent
    Qt6::Gui
    Qt6::Widgets
    Qt6::Network
    Qt6::OpenGLWidgets
    opencv_ximgproc
    $<$<BOOL:${AVAHI_FOUND}>:PkgConfig::AVAHI>
    blosc_static
)

if(VC_USE_PCH)
    target_precompile_headers(VC3D REUSE_FROM vc_core)
endif()
