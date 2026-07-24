set(CMAKE_AUTORCC ON)

set(CHUNK_CACHE_SIZE_GB "10" CACHE STRING "size of the chunk cache in gigabytes")
add_definitions(-DCHUNK_CACHE_SIZE_GB=${CHUNK_CACHE_SIZE_GB})

# Qt shim + Qt image helpers formerly in vc_ui; wrap the shim header for MOC.
qt6_wrap_cpp(vccollection_moc ${CMAKE_SOURCE_DIR}/core/include/vc/ui/VCCollection.hpp)

set(srcs
    ${CMAKE_SOURCE_DIR}/core/src/VCCollection.cpp
    ${vccollection_moc}
    VCAppMain.cpp
    CState.cpp
    CState.hpp
    Keybinds.cpp
    Keybinds.hpp
    CWindow.cpp
    CWindowContextMenu.cpp
    StatusDockPanelHost.cpp
    StatusDockPanelHost.hpp
    RenderBenchRecorder.cpp
    RenderBenchRecorder.hpp
    RenderBenchReplay.cpp
    RenderBenchReplay.hpp
    FileWatcherService.cpp
    FileWatcherService.hpp
    AxisAlignedSliceController.cpp
    AxisAlignedSliceController.hpp
    LineAnnotationFiberClassification.hpp
    LineAnnotationFiberSaveJob.cpp
    LineAnnotationFiberSaveJob.hpp
    FiberNameDisplay.cpp
    FiberNameDisplay.hpp
    FiberSliceGeometry.cpp
    FiberSliceGeometry.hpp
    LineAnnotationController.cpp
    LineAnnotationController.hpp
    LineAnnotationDialog.cpp
    LineAnnotationDialog.hpp
    LineAnnotationGeneratedViews.cpp
    LineAnnotationGeneratedViews.hpp
    OpenDataCatalogWindow.cpp
    OpenDataCatalogWindow.hpp
    OpenDataManifest.cpp
    OpenDataLasagna.cpp
    OpenDataLasagna.hpp
    OpenDataManifest.hpp
    OpenDataSampleProject.cpp
    OpenDataSampleProject.hpp
    OpenDataVolumePrefill.cpp
    OpenDataVolumePrefill.hpp
    OpenDataNormalGrids.cpp
    OpenDataNormalGrids.hpp
    OpenDataSegmentCache.cpp
    OpenDataSegmentCache.hpp
    OpenDataSegmentCacheIO.cpp
    OpenDataSegmentCacheIO.hpp
    SurfaceAffineTransformController.cpp
    SurfaceAffineTransformController.hpp
    SurfaceAreaCalculator.cpp
    SurfaceAreaCalculator.hpp
    AtlasControlPointsDock.cpp
    AtlasControlPointsDock.hpp
    AtlasControlPointsTypes.hpp
    CommandLaunchError.hpp
    FiberSaveBatchTracker.hpp
    SegmentationCommandHandler.cpp
    SegmentationCommandHandler.hpp
    volume_viewers/annotation_tools/SameWrapAnnotationTool.cpp
    volume_viewers/annotation_tools/SameWrapAnnotationTool.hpp
    volume_viewers/CVolumeViewerView.cpp
    VolumeViewerCmaps.cpp
    VolumeViewerCmaps.hpp
    WindowRangeWidget.cpp
    WindowRangeWidget.hpp
    WrapAnnotationWidget.cpp
    WrapAnnotationWidget.hpp
    elements/CollapsibleSettingsGroup.cpp
    elements/CollapsibleSettingsGroup.hpp
    elements/DropdownChecklistButton.cpp
    elements/DropdownChecklistButton.hpp
    elements/JsonProfileEditor.cpp
    elements/JsonProfileEditor.hpp
    elements/SpiralConfigProfileEditor.cpp
    elements/SpiralConfigProfileEditor.hpp
    elements/LabeledControlRow.cpp
    elements/LabeledControlRow.hpp
    elements/VolumeSelector.cpp
    elements/VolumeSelector.hpp
    elements/RangeSlider.cpp
    elements/RangeSlider.hpp
    elements/ViewerStatsBar.cpp
    elements/ViewerStatsBar.hpp
    elements/ViewerSplitGrid.cpp
    elements/ViewerSplitGrid.hpp
    RemoteVolumeCachePaths.cpp
    SettingsDialog.cpp
    SurfaceTreeWidget.hpp
    SurfaceTreeWidget.cpp
    SurfaceDisplayName.hpp
    SurfaceTimestamp.hpp
    SeedingBatchTracker.cpp
    SeedingBatchTracker.hpp
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
    overlays/AtlasOverlayController.cpp
    overlays/AtlasOverlayController.hpp
    overlays/AtlasControlPointsOverlayController.cpp
    overlays/AtlasControlPointsOverlayController.hpp
    overlays/FiberSliceOverlayController.cpp
    overlays/FiberSliceOverlayController.hpp
    overlays/FiberOverlayController.cpp
    overlays/FiberOverlayController.hpp
    overlays/InkDetectionOverlayController.cpp
    overlays/InkDetectionOverlayController.hpp
    overlays/VolumeOverlayController.cpp
    overlays/VolumeOverlayController.hpp
    overlays/ViewerOverlayControllerBase.cpp
    overlays/ViewerOverlayControllerBase.hpp
    overlays/SpiralOverlayController.cpp
    overlays/SpiralOverlayController.hpp
    SurfacePanelController.cpp
    SurfacePanelController.hpp
    MenuActionController.cpp
    MenuActionController.hpp
    VolumeAttachmentController.cpp
    VolumeAttachmentController.hpp
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
    SpiralArtifactCache.cpp
    SpiralArtifactCache.hpp
    SpiralServiceManager.cpp
    SpiralServiceManager.hpp
    SpiralServiceProfile.hpp
    SpiralSshTunnel.cpp
    SpiralSshTunnel.hpp
    SpiralPanel.cpp
    SpiralPanel.hpp
    SpiralWorkspace.cpp
    SpiralWorkspace.hpp
    LasagnaBatchWindow.cpp
    LasagnaBatchWindow.hpp
    segmentation/panels/SegmentationLasagnaPanel.cpp
    segmentation/panels/SegmentationLasagnaPanel.hpp
    ViewerManager.cpp
    ViewerManager.hpp
    viewer_controls/ViewerControlsPanel.cpp
    viewer_controls/ViewerControlsPanel.hpp
    viewer_controls/panels/ViewerCompositePanel.cpp
    viewer_controls/panels/ViewerCompositePanel.hpp
    viewer_controls/panels/ViewerInkDetectionPanel.cpp
    viewer_controls/panels/ViewerInkDetectionPanel.hpp
    viewer_controls/panels/ViewerNavigationPanel.cpp
    viewer_controls/panels/ViewerNavigationPanel.hpp
    viewer_controls/panels/ViewerNormalVisualizationPanel.cpp
    viewer_controls/panels/ViewerNormalVisualizationPanel.hpp
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
    agent_bridge/AgentBridgeServer.cpp
    agent_bridge/AgentBridgeServer.hpp
    agent_bridge/AgentBridgeMethod.cpp
    agent_bridge/AgentBridgeMethod.hpp
    agent_bridge/AgentBridgeMethods_atlas.cpp
    agent_bridge/AgentBridgeMethods_canvas.cpp
    agent_bridge/AgentBridgeMethods_editing.cpp
    agent_bridge/AgentBridgeMethods_fiber.cpp
    agent_bridge/AgentBridgeMethods_flatten.cpp
    agent_bridge/AgentBridgeMethods_jobs.cpp
    agent_bridge/AgentBridgeMethods_lasagna.cpp
    agent_bridge/AgentBridgeMethods_points.cpp
    agent_bridge/AgentBridgeMethods_seeding.cpp
    agent_bridge/AgentBridgeMethods_session.cpp
    agent_bridge/AgentBridgeMethods_surfaceops.cpp
    agent_bridge/AgentBridgeMethods_viewer.cpp
    agent_bridge/AgentBridgeMethods_wrap.cpp
    agent_bridge/AgentBridgeCommandError.hpp
    agent_bridge/AgentBridgeInternal.hpp
    agent_bridge/AgentBridgeJobs.cpp
    agent_bridge/AgentBridgeHandlers_catalog.cpp
    agent_bridge/AgentBridgeHandlers_session.cpp
    agent_bridge/AgentBridgeHandlers_editing.cpp
    agent_bridge/AgentBridgeHandlers_wrap.cpp
    agent_bridge/AgentBridgeHandlers_atlas.cpp
    agent_bridge/AgentBridgeHandlers_fiber.cpp
    agent_bridge/AgentBridgeHandlers_seeding.cpp
    agent_bridge/AgentBridgeHandlers_lasagna.cpp
    agent_bridge/AgentBridgeHandlers_flatten.cpp
    agent_bridge/AgentBridgeHandlers_points.cpp
    agent_bridge/AgentBridgeHandlers_review.cpp
    agent_bridge/AgentBridgeHandlers_viewer.cpp
    agent_bridge/AgentBridgeHandlers_surfaceops.cpp
)

if(VC_ENABLE_TSAN OR VC_ENABLE_LSAN OR VC_ENABLE_UBSAN OR VC_ENABLE_ASAN)
    list(APPEND srcs "${CMAKE_SOURCE_DIR}/misc/sanitizer_config.cpp")
endif()


add_executable(VC3D ${srcs} resources.qrc)

if(WIN32)
    # GUI subsystem so double-click launch doesn't open a console window.
    set_target_properties(VC3D PROPERTIES WIN32_EXECUTABLE ON)
    # Embed the app icon (windres compiles the .rc).
    target_sources(VC3D PRIVATE VC3D.rc)
endif()

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
    vc_atlas
    vc_lasagna
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
    Blosc::blosc
)

# Build-tree Qt plugin deployment (MSVC/vcpkg). vcpkg copies the Qt DLLs next to
# the executable but not the plugin tree, so a freshly built VC3D.exe cannot find
# platforms/qwindows.dll and aborts on startup. Copy the plugin folders next to
# VC3D.exe after each link so it runs straight from the build directory without a
# separate windeployqt step. (Packaging/install deployment is handled separately
# in cmake/WindowsPackaging.cmake.)
if(WIN32 AND MSVC)
    set(_vc_qwin_loc "")
    if(TARGET Qt6::QWindowsIntegrationPlugin)
        foreach(_cfg IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_RELWITHDEBINFO
                     IMPORTED_LOCATION_DEBUG IMPORTED_LOCATION)
            get_target_property(_vc_qwin_loc Qt6::QWindowsIntegrationPlugin ${_cfg})
            if(_vc_qwin_loc)
                break()
            endif()
        endforeach()
    endif()
    if(_vc_qwin_loc)
        # _vc_qwin_loc = <qt>/plugins/platforms/qwindows.dll; go up twice.
        get_filename_component(_vc_qt_plugins "${_vc_qwin_loc}" DIRECTORY)
        get_filename_component(_vc_qt_plugins "${_vc_qt_plugins}" DIRECTORY)
        foreach(_pd platforms styles imageformats iconengines tls generic
                    networkinformation)
            if(EXISTS "${_vc_qt_plugins}/${_pd}")
                add_custom_command(TARGET VC3D POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                        "${_vc_qt_plugins}/${_pd}"
                        "$<TARGET_FILE_DIR:VC3D>/${_pd}"
                    COMMENT "Deploying Qt ${_pd} plugins next to VC3D.exe"
                    VERBATIM)
            endif()
        endforeach()
    else()
        message(WARNING
            "Qt platform plugin not located; VC3D.exe may not launch from the "
            "build tree without a manual windeployqt/plugin copy.")
    endif()
endif()

if(VC_USE_PCH)
    # VC3D gets its own PCH = vc_core's common headers + the heavy Qt umbrella
    # headers. Qt stays out of vc_core's shared PCH so the Qt-free CLI apps that
    # REUSE_FROM it don't pay to parse Qt. (Can't REUSE_FROM vc_core and add Qt
    # — a target has one PCH — so build VC3D's directly from the same list.)
    target_precompile_headers(VC3D PRIVATE
        ${VC_PCH_HEADERS}
        "$<$<COMPILE_LANGUAGE:CXX>:<QtCore/QtCore>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<QtWidgets/QtWidgets>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<QtConcurrent/QtConcurrent>>")
endif()

# The manager checks this location after the development-tree and explicit
# SPIRAL_SERVICE_PATH candidates.  Install the complete module set because the
# service deliberately reuses the existing fitter rather than duplicating it.
install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/spiral/
    DESTINATION share/volume-cartographer/spiral
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "tests" EXCLUDE
        PATTERN "__pycache__" EXCLUDE)
