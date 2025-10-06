configure_file(src/Version.cpp.in Version.cpp)

add_library(vc_core
        src/DiskBasedObjectBaseClass.cpp
        src/GridStore.cpp
        src/NormalGridVolume.cpp
        src/normalgridtools.cpp
        src/Metadata.cpp
        src/Segmentation.cpp
        src/Volume.cpp
        src/VolumePkg.cpp
        src/ChunkedTensor.cpp
        src/Logging.cpp
        src/Slicing.cpp
        src/Surface.cpp
        src/Render.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp
)

target_link_libraries(vc_core
        PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
        opencv_calib3d
        opencv_video
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        z5
        OpenMP::OpenMP_CXX
        tiff
)

qt6_wrap_cpp(metrics_moc include/vc/ui/VCCollection.hpp)
add_library(vc_ui src/VCCollection.cpp src/UDataManipulateUtils.cpp src/surface_metrics.cpp ${metrics_moc})
target_link_libraries(vc_ui PUBLIC vc_core Qt6::Core Qt6::Gui Qt6::Widgets)

add_library(vc_tracer src/GrowPatch.cpp src/GrowSurface.cpp)
target_link_libraries(vc_tracer PUBLIC vc_core vc_ui Ceres::ceres Qt6::Core)


if(VC_INSTALL_LIBS)
    install(
            TARGETS vc_core
            COMPONENT "Libraries"
            EXPORT "${targets_export_name}"
            ARCHIVE DESTINATION "lib"
            LIBRARY DESTINATION "lib"
            INCLUDES DESTINATION "${include_install_dir}"
            RUNTIME DESTINATION "bin"
    )

    install(
            TARGETS vc_ui
            COMPONENT "Libraries"
            EXPORT "${targets_export_name}"
            ARCHIVE DESTINATION "lib"
            LIBRARY DESTINATION "lib"
            INCLUDES DESTINATION "${include_install_dir}"
            RUNTIME DESTINATION "bin"
    )

    install(
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
            DESTINATION "${include_install_dir}"
            COMPONENT "Libraries"
            FILES_MATCHING REGEX ".*\\.(h|hpp)$"
    )
endif()
