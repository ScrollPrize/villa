set(CMAKE_AUTORCC ON)

set(CHUNK_CACHE_SIZE_GB "10" CACHE STRING "size of the chunk cache in gigabytes")
add_definitions(-DCHUNK_CACHE_SIZE_GB=${CHUNK_CACHE_SIZE_GB})

set(srcs
    VCAppMain.cpp
    Keybinds.cpp
    Keybinds.hpp
    CWindow.cpp
    CWindowContextMenu.cpp
    CVolumeViewer.cpp
    CVolumeViewerIntersections.cpp
    CVolumeViewerRender.cpp
    CVolumeViewerOverlay.cpp
    CVolumeViewerView.cpp
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
    elements/VolumeSelector.cpp
    elements/VolumeSelector.hpp
    elements/RangeSlider.cpp
    elements/RangeSlider.hpp
    SettingsDialog.cpp
    CSurfaceCollection.cpp
    SurfaceTreeWidget.hpp
    SurfaceTreeWidget.cpp
    SeedingWidget.cpp
    SeedingWidget.hpp
    DrawingWidget.cpp
    DrawingWidget.hpp
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
    segmentation/panels/SegmentationCellReoptPanel.cpp
    segmentation/panels/SegmentationCellReoptPanel.hpp
    segmentation/panels/SegmentationNeuralTracerPanel.cpp
    segmentation/panels/SegmentationNeuralTracerPanel.hpp
    segmentation/panels/SegmentationDirectionFieldPanel.cpp
    segmentation/panels/SegmentationDirectionFieldPanel.hpp
    segmentation/growth/SegmentationGrowth.cpp
    segmentation/growth/ExtrapolationGrowth.cpp
    segmentation/growth/ExtrapolationGrowth.hpp
    segmentation/growth/SegmentationGrower.cpp
    segmentation/growth/SegmentationGrower.hpp
    segmentation/tools/SegmentationEditManager.cpp
    segmentation/tools/SegmentationEditManager.hpp
    segmentation/tools/ApprovalMaskBrushTool.cpp
    segmentation/tools/ApprovalMaskBrushTool.hpp
    segmentation/tools/CellReoptimizationTool.cpp
    segmentation/tools/CellReoptimizationTool.hpp
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
    overlays/VolumeOverlayController.cpp
    overlays/VolumeOverlayController.hpp
    overlays/ViewerOverlayControllerBase.cpp
    overlays/ViewerOverlayControllerBase.hpp
    SurfacePanelController.cpp
    SurfacePanelController.hpp
    MenuActionController.cpp
    MenuActionController.hpp
    segmentation/SegmentationModule.cpp
    segmentation/SegmentationModule.hpp
    segmentation/growth/SegmentationCorrections.cpp
    segmentation/growth/SegmentationCorrections.hpp
    segmentation/SegmentationUndoHistory.hpp
    NeuralTraceServiceManager.cpp
    NeuralTraceServiceManager.hpp
    ViewerManager.cpp
    ViewerManager.hpp
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
    elements/COutlinedTextItem.hpp
    elements/COutlinedTextItem.cpp
)

if(VC_ENABLE_TSAN OR VC_ENABLE_LSAN OR VC_ENABLE_UBSAN OR VC_ENABLE_ASAN)
    list(APPEND srcs "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.c")
endif()


add_executable(VC3D ${srcs} resources.qrc)

target_include_directories(VC3D
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/libs/edt
        ${CMAKE_SOURCE_DIR}/libs/cc3d
        ${CMAKE_SOURCE_DIR}/libs/djikstra3d
)

target_link_libraries(VC3D
    vc_core
    vc_ui
    vc_tracer
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    opencv_ximgproc
)

if (${VC_INSTALL_APPS})
    install(
        TARGETS VC3D
        BUNDLE DESTINATION . COMPONENT Programs
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Programs
    )

    install(
            PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/vc_sync.py
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT Programs
    )


    qt_generate_deploy_app_script(
        TARGET VC3D
        OUTPUT_SCRIPT vc_gui_deploy_script
        NO_UNSUPPORTED_PLATFORM_ERROR
    )
    install(SCRIPT ${vc_gui_deploy_script} COMPONENT Programs)
endif()
