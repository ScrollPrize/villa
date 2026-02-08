configure_file(src/Version.cpp.in Version.cpp)

add_library(vc_core
        src/GridStore.cpp
        src/NormalGridVolume.cpp
        src/normalgridtools.cpp
        src/Segmentation.cpp
        src/Volume.cpp
        src/VolumePkg.cpp
        src/ChunkedTensor.cpp
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
        src/Geometry.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
        src/ABFFlattening.cpp
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
        OpenABF
        #${CURL_LIBRARIES}
        #${OPENSSL_LIBRARIES}
        #${ZLIB_LIBRARIES}
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
    # Eigen (used by Ceres)
    <Eigen/Core>
    <Eigen/Dense>
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

add_library(vc_tracer src/GrowPatch.cpp src/GrowSurface.cpp src/NeuralTracerConnection.cpp)
target_link_libraries(vc_tracer PUBLIC vc_core vc_ui Ceres::ceres Qt6::Core)
target_include_directories(vc_tracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../libs/edt)
if(VC_USE_PCH)
    target_precompile_headers(vc_tracer REUSE_FROM vc_core)
endif()

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
