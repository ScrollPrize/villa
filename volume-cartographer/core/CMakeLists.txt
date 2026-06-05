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
        src/PointCollections.cpp
        src/surface_metrics.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
        src/render/ChunkCache.cpp
        src/render/ChunkedPlaneSampler.cpp
        src/render/Colormaps.cpp
        src/render/PostProcess.cpp
        src/render/ZarrChunkFetcher.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../apps/src/vc_merge_tifxyz_grid.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp
)

# Grid-resolution helpers (vc_merge_tifxyz_grid) live in apps/src but are
# compiled into vc_core so the unit tests link them without the binary.
target_include_directories(vc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../apps/src)

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
target_link_libraries(vc_core PUBLIC
    $<BUILD_INTERFACE:utils>
    $<BUILD_INTERFACE:utils_c3d_codec>
)
target_link_libraries(vc_core PRIVATE
    Blosc::blosc
    ${VC_LIBBACKTRACE}
)

# Neutral PCH carrier shared by vc_core, downstream vc_* libs, and CLI apps.
# Keep this separate from vc_core: shared-library targets add export macros
# such as vc_core_EXPORTS while compiling themselves, and GCC rejects a PCH
# built with those target-local definitions when consumers try to REUSE_FROM it.
if(VC_USE_PCH)
    set(VC_PCH_HEADERS
        "$<$<COMPILE_LANGUAGE:CXX>:<algorithm>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<array>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<atomic>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<chrono>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<cmath>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<cstdint>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<cstring>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<filesystem>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<fstream>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<iomanip>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<iostream>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<limits>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<memory>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<mutex>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<optional>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<sstream>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<stdexcept>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<string>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<unordered_map>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<unordered_set>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<random>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<utility>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<vector>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<functional>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<set>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<map>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<thread>>"
        # <format> is the heaviest std header in the build (~1.3k s of
        # instantiation across TUs; libstdc++ 15 + Qt pull it in widely).
        "$<$<COMPILE_LANGUAGE:CXX>:<format>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<string_view>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<opencv2/core.hpp>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<opencv2/imgproc.hpp>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<opencv2/imgcodecs.hpp>>"
        "$<$<COMPILE_LANGUAGE:CXX>:<nlohmann/json.hpp>>"
        CACHE INTERNAL "PCH header list shared by vc_* and downstream targets")

    set(_vc_common_pch_source "${CMAKE_CURRENT_BINARY_DIR}/vc_common_pch.cpp")
    file(WRITE "${_vc_common_pch_source}" "void vc_common_pch_anchor() {}\n")
    add_library(vc_common_pch STATIC "${_vc_common_pch_source}")
    set_target_properties(vc_common_pch PROPERTIES
        AUTOMOC OFF AUTOUIC OFF AUTORCC OFF
        POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(vc_common_pch PRIVATE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        TIFF::TIFF)
    target_precompile_headers(vc_common_pch PRIVATE ${VC_PCH_HEADERS})
    target_precompile_headers(vc_core REUSE_FROM vc_common_pch)
endif()

add_library(vc_flattening src/ABFFlattening.cpp)
target_link_libraries(vc_flattening PUBLIC vc_core OpenABF)

add_library(vc_inpaint src/InpaintSurface.cpp)
target_link_libraries(vc_inpaint PUBLIC vc_core PRIVATE Ceres::ceres)

add_library(vc_lasagna
    src/lasagna/Dataset.cpp
    src/lasagna/LasagnaNormalSampler.cpp
    src/lasagna/LineViewBuilder.cpp
    src/lasagna/LineOptimizer.cpp
    src/lasagna/Manifest.cpp
)
target_link_libraries(vc_lasagna PUBLIC vc_core PRIVATE Ceres::ceres)

add_library(vc_atlas
    src/Atlas.cpp
    src/FiberIntersections.cpp
)
target_link_libraries(vc_atlas PUBLIC vc_core vc_lasagna PRIVATE Ceres::ceres)

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
target_link_libraries(vc_tracer PUBLIC vc_core vc_flattening Ceres::ceres)
target_include_directories(vc_tracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../libs/edt)

if(VC_USE_PCH)
    foreach(_lib vc_flattening vc_inpaint vc_lasagna vc_atlas vc_tracer)
        target_precompile_headers(${_lib} REUSE_FROM vc_common_pch)
    endforeach()
endif()
