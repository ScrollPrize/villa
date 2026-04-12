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
        src/SparseChunkSpool.cpp
        src/Slicing.cpp
        src/Compositing.cpp
        src/PlaneSurface.cpp
        src/Surface.cpp
        src/QuadSurface.cpp
        src/Render.cpp
        src/Tiff.cpp
        src/Zarr.cpp
        src/VcDataset.cpp
        src/NetworkFilesystem.cpp
        src/cache/ChunkSource.cpp
        # DiskStore.cpp removed — cold tier now uses ZarrArray
        src/cache/HttpMetadataFetcher.cpp
        src/cache/BlockCache.cpp
        src/cache/IOPool.cpp
        src/cache/BlockPipeline.cpp
        src/cache/VcDecompressor.cpp
        src/cache/SimpleCacheFactory.cpp
        src/LoadJson.cpp
        src/RemoteUrl.cpp
        src/RemoteScroll.cpp
        src/PostProcess.cpp
        src/Geometry.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
        src/render/AdaptiveCamera.cpp
        src/render/Colormaps.cpp
        src/render/PostProcess.cpp
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

# utils header-only library + vendored video codec
# WHOLE_ARCHIVE is needed because ThinLTO + lld can internalize symbols from
# static archives before cross-archive references are resolved.
target_link_libraries(vc_core PUBLIC
    $<BUILD_INTERFACE:utils>
    $<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,utils_video_codec>>
)
target_link_libraries(vc_core PRIVATE
    $<BUILD_INTERFACE:blosc_static>
)



# Slicing.cpp is the hot sampling loop (87% of render time).
# Force O3 + fast-math for maximum vectorization regardless of build type.
set_source_files_properties(src/Slicing.cpp PROPERTIES
    COMPILE_OPTIONS "-O3;-ffast-math"
)

# Precompiled headers for faster builds
if(VC_USE_PCH)
target_precompile_headers(vc_core PRIVATE
    # STL - containers
    <array>
    <deque>
    <list>
    <map>
    <queue>
    <set>
    <stack>
    <unordered_map>
    <unordered_set>
    <vector>
    # STL - algorithms and utilities
    <algorithm>
    <chrono>
    <cmath>
    <cstdint>
    <cstdlib>
    <cstring>
    <functional>
    <iterator>
    <limits>
    <numeric>
    <tuple>
    <type_traits>
    <utility>
    # STL - I/O and strings
    <filesystem>
    <fstream>
    <iomanip>
    <iostream>
    <sstream>
    <string>
    <string_view>
    # STL - memory and smart pointers
    <memory>
    <new>
    # STL - concurrency
    <atomic>
    <condition_variable>
    <mutex>
    <shared_mutex>
    <thread>
    # STL - other
    <cassert>
    <optional>
    <stdexcept>
    <variant>
    # OpenCV
    <opencv2/core.hpp>
    <opencv2/core/mat.hpp>
    <opencv2/imgproc.hpp>
    <opencv2/imgcodecs.hpp>
    # Boost (used by CLI apps)
    <boost/program_options.hpp>
    # JSON (heavy templates, used in ~70 TUs)
    <nlohmann/json.hpp>
)
endif()

qt6_wrap_cpp(metrics_moc include/vc/ui/VCCollection.hpp)
add_library(vc_ui src/VCCollection.cpp src/UDataManipulateUtils.cpp src/surface_metrics.cpp ${metrics_moc})
target_link_libraries(vc_ui PUBLIC vc_core Qt6::Core Qt6::Gui Qt6::Widgets)
if(VC_USE_PCH)
    target_precompile_headers(vc_ui REUSE_FROM vc_core)
endif()

add_library(vc_flattening src/ABFFlattening.cpp)
target_link_libraries(vc_flattening PUBLIC vc_core OpenABF)

add_library(vc_tracer src/GrowPatch.cpp src/GrowSurface.cpp src/NeuralTracerConnection.cpp src/ChunkedTensor.cpp)
target_link_libraries(vc_tracer PUBLIC vc_core vc_ui Ceres::ceres Qt6::Core)
target_include_directories(vc_tracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../libs/edt)
if(VC_USE_PCH)
    target_precompile_headers(vc_tracer REUSE_FROM vc_core)
endif()
