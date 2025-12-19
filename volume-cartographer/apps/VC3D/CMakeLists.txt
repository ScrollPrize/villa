set(CMAKE_AUTORCC ON)

set(CHUNK_CACHE_SIZE_GB "10" CACHE STRING "size of the chunk cache in gigabytes")
add_definitions(-DCHUNK_CACHE_SIZE_GB=${CHUNK_CACHE_SIZE_GB})

set(srcs
    VCAppMain.cpp
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
    segmentation/SegmentationWidget.cpp
    segmentation/SegmentationWidget.hpp
    segmentation/SegmentationGrowth.cpp
    segmentation/SegmentationGrower.cpp
    segmentation/SegmentationGrower.hpp
    segmentation/SegmentationEditManager.cpp
    segmentation/SegmentationEditManager.hpp
    segmentation/SegmentationBrushTool.cpp
    segmentation/SegmentationBrushTool.hpp
    segmentation/ApprovalMaskBrushTool.cpp
    segmentation/ApprovalMaskBrushTool.hpp
    segmentation/SegmentationLineTool.cpp
    segmentation/SegmentationLineTool.hpp
    segmentation/SegmentationPushPullTool.cpp
    segmentation/SegmentationPushPullTool.hpp
    segmentation/SegmentationTool.hpp
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
    segmentation/SegmentationCorrections.cpp
    segmentation/SegmentationCorrections.hpp
    segmentation/SegmentationUndoHistory.hpp
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
)

target_link_libraries(VC3D
    vc_core
    vc_ui
    vc_tracer
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

if (${VC_INSTALL_APPS})
    install(
        TARGETS VC3D
        BUNDLE DESTINATION . COMPONENT Programs
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Programs
    )

    install(
            PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/vc_sync.py ${CMAKE_SOURCE_DIR}/scripts/flatboi.py
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
