configure_file(src/Version.cpp.in Version.cpp)

add_library(vc_fiber_view STATIC src/FiberView.cpp)
target_include_directories(vc_fiber_view PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(vc_fiber_view PUBLIC cxx_std_20)
target_link_libraries(vc_fiber_view PUBLIC vc_object_store::vc_geometry)
set_target_properties(vc_fiber_view PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(vc_core
        src/PolylineObj.cpp
        src/AtomicFile.cpp
        src/TexturedMesh.cpp
        src/SurfaceTexture.cpp
        src/GridStore.cpp
        src/NormalGridVolume.cpp
        src/normalgridtools.cpp
        src/Volume.cpp
        src/VolumePkg.cpp
        src/ProjectDocument.cpp
        src/SurfacePatchIndex.cpp
        src/PointIndex.cpp
        src/Logging.cpp
        src/CrashHandler.cpp
        src/MemMap.cpp
        src/UnixSocket.cpp
        src/VoxelSizeMetadata.cpp
        src/SparseChunkSpool.cpp
        src/Compositing.cpp
        src/PlaneSurface.cpp
        src/Surface.cpp
        src/QuadSurface.cpp
        src/QuadSurfaceStore.cpp
        src/StoreSurfaces.cpp
        src/SurfaceMeshing.cpp
        src/Tiff.cpp
        src/Zarr.cpp
        src/VcDataset.cpp
        src/LoadJson.cpp
        src/RemoteUrl.cpp
        src/RemoteFileCache.cpp
        src/RemoteMetadataCache.cpp
        src/RemoteCacheSettings.cpp
        src/HttpFetch.cpp
        src/S3AuthFallback.cpp
        src/Geometry.cpp
        src/PointCollections.cpp
        src/StoredCollection.cpp
        src/Thinning.cpp
        src/Umbilicus.cpp
        src/render/ChunkCache.cpp
        src/render/ChunkRequestScheduler.cpp
        src/render/DecodedChunkCacheBudget.cpp
        src/render/PersistentZarrCacheBudget.cpp
        src/render/ChunkedPlaneSampler.cpp
        src/render/PinnedChunkRegion.cpp
        src/render/Colormaps.cpp
        src/render/PostProcess.cpp
        src/render/ZarrChunkFetcher.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/Version.cpp
)

target_link_libraries(vc_core
    PUBLIC vc_object_store::vc_objects vc_object_store::vc_object_formats vc_reader::catalog vc_reader::reader
        vc_reader::transforms
        PUBLIC
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        ${VC_OPENCV_CALIB_TARGETS}
        nlohmann_json::nlohmann_json
        OpenMP::OpenMP_CXX
        TIFF::TIFF
        CURL::libcurl
)

# utils header-only library + vendored c3d and volcomp codecs.
target_link_libraries(vc_core PUBLIC
    $<BUILD_INTERFACE:utils>
    $<BUILD_INTERFACE:utils_c3d_codec>
    vc_delta3d::vc_delta3d
    $<BUILD_INTERFACE:utils_volcomp_codec>
)
target_link_libraries(vc_core PRIVATE
    Blosc::blosc
    ${VC_LIBBACKTRACE}
)

# Winsock for the AF_UNIX compat layer (UnixSocket.cpp). PUBLIC because
# consumers (vc_tracer, VC3D) call send()/recv() directly.
if(WIN32)
    target_link_libraries(vc_core PUBLIC ws2_32)
endif()

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

add_library(vc_inpaint src/InpaintSurface.cpp)
target_link_libraries(vc_inpaint PUBLIC vc_core PRIVATE Ceres::ceres)

add_library(vc_lasagna
    src/lasagna/ChannelSampler.cpp
    src/lasagna/Dataset.cpp
    src/lasagna/PredictionPooling.cpp
    src/lasagna/LaplaceAmgx.cpp
    src/lasagna/LaplaceRank.cpp
    src/lasagna/LasagnaNormalSampler.cpp
    src/lasagna/LineViewBuilder.cpp
    src/lasagna/LineOptimizer.cpp
    src/lasagna/LineSpline.cpp
    src/lasagna/Manifest.cpp
    src/lasagna/MaxflowGraph.cpp
    src/lasagna/ProjectVolumes.cpp
)
if(VC_ENABLE_AMGX)
    target_compile_definitions(vc_lasagna PUBLIC VC_ENABLE_AMGX=1)
    target_link_libraries(vc_lasagna PRIVATE AMGX::amgx)
endif()
target_link_libraries(vc_lasagna PUBLIC vc_core PRIVATE Ceres::ceres)
find_package(LAPACK QUIET)
target_link_libraries(vc_lasagna PRIVATE Eigen3::Eigen)
if(LAPACK_FOUND AND UNIX AND NOT APPLE)
    target_link_libraries(vc_lasagna PRIVATE ${LAPACK_LIBRARIES})
    target_compile_definitions(vc_lasagna PRIVATE VC_POOLING_LAPACK)
endif()

add_library(vc_fiber_tracer
    src/fiber_tracer/FiberReferenceReplayBenchmark.cpp
    src/fiber_tracer/BinaryBeliefPropagation.cpp
    src/fiber_tracer/FiberTraceSeed.cpp
    src/fiber_tracer/FiberTraceBeliefPropagation.cpp
    src/fiber_tracer/FiberTraceConstraints.cpp
    src/fiber_tracer/FiberTraceWindingBeliefPropagation.cpp
    src/fiber_tracer/FiberTraceWindingLeastSquares.cpp
    src/fiber_tracer/FiberTraceWindingOrderedCuts.cpp
    src/fiber_tracer/LasagnaNormalAlignment.cpp
    src/fiber_tracer/FiberAnchors.cpp
    src/fiber_tracer/FiberAnchorObjectives.cpp
    src/fiber_tracer/FiberAnchorFinalEvaluation.cpp
    src/fiber_tracer/FiberAxisTensor.cpp
    src/fiber_tracer/FiberGraph.cpp
    src/fiber_tracer/FiberletChunkGraph.cpp
    src/fiber_tracer/FiberletCropTrace.cpp
    src/fiber_tracer/FiberLocalScoring.cpp
    src/fiber_tracer/FiberPaths.cpp
    src/fiber_tracer/FiberReplayMetric.cpp
    src/fiber_tracer/FiberReverseConsistency.cpp
    src/fiber_tracer/FiberReferenceReplayCore.cpp
    src/fiber_tracer/FiberChunkTrace.cpp
    src/fiber_tracer/FiberChunkFilter.cpp
    src/fiber_tracer/FiberChunkGpuBridge.cpp
    src/fiber_tracer/FiberChunkRoute.cpp
    src/fiber_tracer/FiberLiveBridge.cpp
    src/fiber_tracer/FiberEvaluation.cpp
    src/fiber_tracer/FiberReplayVisualization.cpp
    src/fiber_tracer/FiberletCropVisualization.cpp
    src/fiber_tracer/FiberCaseQueue.cpp
    src/fiber_tracer/PolylineGeometry.cpp
    src/fiber_tracer/FiberletStorage.cpp
    src/fiber_tracer/FiberletDataset.cpp
    src/fiber_tracer/FiberletQuantization.cpp
    src/fiber_tracer/FiberletOnDemand.cpp
    src/fiber_tracer/FiberletPipeline.cpp
    src/fiber_tracer/FiberTrace.cpp
    src/fiber_tracer/FiberSideStripIntersections.cpp
)
target_link_libraries(vc_fiber_tracer PUBLIC vc_core vc_lasagna PRIVATE Eigen3::Eigen Ceres::ceres)
target_link_libraries(vc_fiber_tracer PRIVATE ${CMAKE_DL_LIBS})

# Optional backend; CPU builds do not need a CUDA toolkit.
option(VC_BUILD_FIBER_CHUNK_CUDA "Build resident CUDA chunk tracing backend" OFF)
if(VC_BUILD_FIBER_CHUNK_CUDA)
    enable_language(CUDA)
    add_library(vc_fiber_chunk_cuda SHARED src/fiber_tracer/FiberChunkGpu.cu)
    target_compile_definitions(vc_fiber_chunk_cuda PRIVATE VC_CHUNK_GPU_LIBRARY)
    add_executable(benchmark_fiber_chunk_cuda src/fiber_tracer/FiberChunkGpu.cu)
    foreach(target vc_fiber_chunk_cuda benchmark_fiber_chunk_cuda)
        target_include_directories(${target} PRIVATE include)
        target_link_libraries(${target} PRIVATE nlohmann_json::nlohmann_json)
        set_target_properties(${target} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--expt-relaxed-constexpr>)
    endforeach()
endif()
if(VC_TESTING)
    target_compile_definitions(vc_fiber_tracer PRIVATE VC_TESTING)
endif()

add_library(vc_tracer
    src/GrowPatch.cpp
    src/GrowSurface.cpp
    src/SurfTrackerData.cpp
    src/NeuralTracerConnection.cpp
    src/growth_strategies/CandidateOrdering.cpp
    src/growth_strategies/GrowthConfig.cpp
)
target_link_libraries(vc_tracer PUBLIC vc_core Ceres::ceres)

if(VC_USE_PCH)
    foreach(_lib vc_inpaint vc_lasagna vc_fiber_tracer vc_tracer)
        target_precompile_headers(${_lib} REUSE_FROM vc_common_pch)
    endforeach()
endif()
