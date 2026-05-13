configure_file(src/Version.cpp.in Version.cpp)

add_library(vc_core
        src/GridStore.cpp
        src/NormalGridVolume.cpp
        src/normalgridtools.cpp
        src/Segmentation.cpp
        src/Volume.cpp
        src/VolumePkg.cpp
        src/SurfacePatchIndex.cpp
        src/PointIndex.cpp
        src/Logging.cpp
        src/CrashHandler.cpp
        src/VolpkgConvert.cpp
        src/SparseChunkSpool.cpp
        src/Slicing.cpp
        src/Compositing.cpp
        src/AffineTransform.cpp
        src/PlaneSurface.cpp
        src/Surface.cpp
        src/QuadSurface.cpp
        src/Render.cpp
        src/Tiff.cpp
        src/Zarr.cpp
        src/VcDataset.cpp
        src/LoadJson.cpp
        src/RemoteUrl.cpp
        src/HttpFetch.cpp
        src/PostProcess.cpp
        src/Geometry.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
        src/render/ChunkCache.cpp
        src/render/ChunkedPlaneSampler.cpp
        src/render/Colormaps.cpp
        src/render/PostProcess.cpp
        src/render/ZarrChunkFetcher.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp
)

target_link_libraries(vc_core
        PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_calib3d
        opencv_video
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        TIFF::TIFF
        OpenABF
        CURL::libcurl
)

# utils header-only library + vendored c3d codec.
# WHOLE_ARCHIVE is needed because ThinLTO + lld can internalize symbols from
# static archives before cross-archive references are resolved.
target_link_libraries(vc_core PUBLIC
    $<BUILD_INTERFACE:utils>
    $<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,utils_c3d_codec>>
)
target_link_libraries(vc_core PRIVATE
    Blosc::blosc
    ${VC_LIBBACKTRACE}
)



# Slicing.cpp is the hot sampling loop (87% of render time).
# Force O3 + fast-math for maximum vectorization regardless of build type.
set_source_files_properties(src/Slicing.cpp PROPERTIES
    COMPILE_OPTIONS "-O3;-ffast-math"
)

qt6_wrap_cpp(metrics_moc include/vc/ui/VCCollection.hpp)
add_library(vc_ui src/VCCollection.cpp src/UDataManipulateUtils.cpp src/surface_metrics.cpp ${metrics_moc})
target_link_libraries(vc_ui PUBLIC vc_core Qt6::Core Qt6::Gui Qt6::Widgets)

add_library(vc_flattening src/ABFFlattening.cpp)
target_link_libraries(vc_flattening PUBLIC vc_core OpenABF)

add_library(vc_tracer
    src/GrowPatch.cpp
    src/GrowSurface.cpp
    src/SurfTrackerData.cpp
    src/NeuralTracerConnection.cpp
    src/ChunkedTensor.cpp
    src/growth_strategies/CandidateOrdering.cpp
    src/growth_strategies/ComponentPruning.cpp
    src/growth_strategies/GrowthConfig.cpp
)
target_link_libraries(vc_tracer PUBLIC vc_core vc_ui vc_flattening Ceres::ceres Qt6::Core)
target_include_directories(vc_tracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../libs/edt)
