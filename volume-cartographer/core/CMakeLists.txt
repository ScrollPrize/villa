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
        src/ChunkCache.cpp
        src/Slicing.cpp
        src/Compositing.cpp
        src/PlaneSurface.cpp
        src/QuadSurface.cpp
        src/Render.cpp
        src/Tiff.cpp
        src/Zarr.cpp
        src/VcDataset.cpp
        src/cache/ChunkSource.cpp
        src/cache/DiskStore.cpp
        src/cache/HttpMetadataFetcher.cpp
        src/cache/IOPool.cpp
        src/cache/TieredChunkCache.cpp
        src/cache/VcDecompressor.cpp
        src/cache/SimpleCacheFactory.cpp
        src/NetworkFilesystem.cpp
        src/RemoteUrl.cpp
        src/CoordGrid.cpp
        src/PostProcess.cpp
        src/Geometry.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
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
        z5
        OpenMP::OpenMP_CXX
        TIFF::TIFF
        CURL::libcurl
)

# Enable curl-backed HTTP cache code
target_compile_definitions(vc_core PUBLIC VC_USE_CURL)

# VcDataset dependencies
target_link_libraries(vc_core PRIVATE blosc_static utils)

# c-blosc internal headers (zstd, lz4, zlib) for VcDataset decompression
target_include_directories(vc_core PRIVATE
    ${c-blosc_SOURCE_DIR}/blosc
    ${c-blosc_SOURCE_DIR}/internal-complibs/zstd-1.5.5
    ${c-blosc_SOURCE_DIR}/internal-complibs/lz4-1.9.4
    ${c-blosc_SOURCE_DIR}/internal-complibs/zlib-1.2.13
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
    # z5/xtensor (zarr volume access - heavy templates)
    <xtensor/containers/xarray.hpp>
    <xtensor/containers/xadapt.hpp>
    <xtensor/views/xview.hpp>
    <xtensor/generators/xbuilder.hpp>
    <z5/factory.hxx>
    <z5/dataset.hxx>
    <z5/attributes.hxx>
    <z5/filesystem/handle.hxx>
    <z5/filesystem/dataset.hxx>
    <z5/multiarray/xtensor_access.hxx>
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

