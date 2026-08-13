#pragma once

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/Sampling.hpp"

#include <opencv2/core/mat.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace vc::render {

class ChunkedPlaneSampler {
public:
    using TrilinearCornerPointVisitor = void (*)(
        void* context,
        size_t pointIndex,
        const cv::Vec3f& fractionXYZ,
        bool valid,
        std::span<const std::array<uint8_t, 8>> volumeCorners);

    struct Options {
        Options()
            : sampling(vc::Sampling::Nearest)
            , tileSize(32)
            , queueMisses(true)
            , queuedFallbackLevels(-1)
        {
        }
        Options(vc::Sampling sampling_, int tileSize_)
            : sampling(sampling_)
            , tileSize(tileSize_)
            , queueMisses(true)
            , queuedFallbackLevels(-1)
        {
        }

        vc::Sampling sampling;
        int tileSize;
        // When false, sample only decoded chunks already resident in memory.
        // Resident-only reads also leave the shared cache's LRU order alone.
        bool queueMisses;
        // Fine-to-coarse sampling always queues the requested start level.
        // Limit how many following fallback levels may also queue misses:
        // -1 preserves the general-purpose "all levels" behavior, while 0
        // makes every fallback lookup resident-only.
        int queuedFallbackLevels;
        ChunkRequestContext request;
    };

    struct Stats {
        int coveredPixels = 0;
        int requestedChunks = 0;
        int errorChunks = 0;
        int missingChunks = 0;
        int fallbackLevels = 0;
        bool requestedLevelOnly = false;
        double cornerPrepareSeconds = 0.0;
        double cornerLayoutSeconds = 0.0;
        double cornerPinSeconds = 0.0;
        double cornerGatherSeconds = 0.0;
        uint64_t cornerLayoutChunkRuns = 0;
        uint64_t cornerBoundaryPoints = 0;
        uint64_t cornerDependencies = 0;
        uint64_t cornerPointCount = 0;
        uint64_t cornerUniqueVoxelCubes = 0;
        uint64_t cornerWorkerTasks = 0;
        uint64_t cornerMaxCandidatesPerCube = 0;
        std::array<uint64_t, 65> cornerCubeOccupancyHistogram{};
        std::vector<uint64_t> cornerDependencyIds;
    };

    struct ChunkPixelLookupLevel {
        int level = 0;
        // Pixel IDs are one-based indices into this table; zero is invalid.
        std::vector<ChunkKey> chunks;
        cv::Mat_<uint16_t> pixelIds;
        bool overflowed = false;
    };

    // Queue chunk dependencies for pixels not already covered. The viewer can
    // call these before sampling a frame so cache misses start resolving early.
    static Stats requestPlaneDependencies(IChunkedArray& array,
                                          int level,
                                          const cv::Vec3f& origin,
                                          const cv::Vec3f& vxStep,
                                          const cv::Vec3f& vyStep,
                                          const cv::Mat_<uint8_t>& coverage,
                                          const Options& options = Options());

    static Stats requestCoordsDependencies(IChunkedArray& array,
                                           int level,
                                           const cv::Mat_<cv::Vec3f>& coords,
                                           const cv::Mat_<uint8_t>& coverage,
                                           const Options& options = Options());

    static std::vector<ChunkKey> collectPlaneDependencies(IChunkedArray& array,
                                                          int level,
                                                          const cv::Vec3f& origin,
                                                          const cv::Vec3f& vxStep,
                                                          const cv::Vec3f& vyStep,
                                                          const cv::Mat_<uint8_t>& coverage,
                                                          const Options& options = Options());

    static std::vector<ChunkKey> collectCoordsDependencies(IChunkedArray& array,
                                                           int level,
                                                           const cv::Mat_<cv::Vec3f>& coords,
                                                           const cv::Mat_<uint8_t>& coverage,
                                                           const Options& options = Options());

    // Collect candidate chunk occurrences for sparse logical-level-0 XYZ
    // samples. `viewportPositions` uses framebuffer pixel coordinates and must
    // match `coords`. No cache state is read or changed; the atomic demand
    // publisher filters chunks that resolved while this local traversal ran.
    // Nearby occurrences of one chunk are deduplicated in 2-D.
    static std::vector<ChunkViewportSample> collectViewportDependencies(
        IChunkedArray& array,
        int startLevel,
        const std::vector<cv::Vec3f>& coords,
        const std::vector<std::array<float, 2>>& viewportPositions,
        const Options& options = Options(),
        float dedupRadiusPixels = 8.0f);

    // Build a debug/diagnostic mapping without reading or queueing cache data.
    // Each pixel names the containing chunk at every included level through a
    // compact level-local uint16 ID. Coordinates are logical level-0 XYZ.
    static std::vector<ChunkPixelLookupLevel> buildChunkPixelLookup(
        IChunkedArray& array,
        VolumeSourceId sourceId,
        int startLevel,
        int fallbackLevels,
        const cv::Mat_<cv::Vec3f>& coords,
        vc::Sampling sampling = vc::Sampling::Nearest,
        bool zeroIsSentinel = true);

    // Selects how far a GUI viewport should queue coarse fallback data. When
    // pixelsPerLevel0VolumeVoxel is available, stops early once one average
    // chunk edge spans the larger viewport edge in level-0 volume voxels.
    // Parameterized surfaces must pass nullopt: their screen scale is pixels
    // per surface unit, not pixels per volume voxel, so the bounded full range
    // is required unless they provide an explicit volume-space conversion.
    static int fallbackLevelCountForViewport(
        IChunkedArray& array,
        int startLevel,
        int viewportWidth,
        int viewportHeight,
        std::optional<float> pixelsPerLevel0VolumeVoxel,
        int maximumFallbackLevels = 5);

    // Samples one pyramid level into `out` for pixels not already marked in
    // `coverage`. Coordinates are logical level-0 XYZ voxel coordinates.
    static Stats samplePlaneLevel(IChunkedArray& array,
                                  int level,
                                  const cv::Vec3f& origin,
                                  const cv::Vec3f& vxStep,
                                  const cv::Vec3f& vyStep,
                                  cv::Mat_<uint8_t>& out,
                                  cv::Mat_<uint8_t>& coverage,
                                  const Options& options = Options());

    static Stats sampleCoordsLevel(IChunkedArray& array,
                                   int level,
                                   const cv::Mat_<cv::Vec3f>& coords,
                                   cv::Mat_<uint8_t>& out,
                                   cv::Mat_<uint8_t>& coverage,
                                   const Options& options = Options());

    // Blocking requested-level sampling for batch/CLI paths. This samples only
    // the requested level: required chunks are fetched and pinned before
    // sampling starts, scale fallback is disabled, and chunk I/O errors throw.
    // Known-missing requested-level chunks render as black covered pixels.
    static Stats sampleCoordsLevelBlockingRequestedLevel(IChunkedArray& array,
                                                         int level,
                                                         const cv::Mat_<cv::Vec3f>& coords,
                                                         cv::Mat_<uint8_t>& out,
                                                         cv::Mat_<uint8_t>& coverage,
                                                         const Options& options = Options());

    // Batch primitive for callers that need to interpolate structured values
    // themselves. Input coordinates are XYZ voxel coordinates in the requested
    // level (not logical level-0 coordinates). Geometry/dependencies are built
    // once, every array must have the same requested-level shape and may use
    // its own chunk grid. One dependency layout is reused per distinct chunk
    // shape. values[volume][point] contains corners in dz/dy/dx order.
    static Stats sampleTrilinearCornersLevelBlockingRequestedLevel(
        const std::vector<IChunkedArray*>& arrays,
        int level,
        const std::vector<cv::Vec3f>& levelCoords,
        std::vector<std::vector<std::array<uint8_t, 8>>>& values,
        std::vector<cv::Vec3f>& fractionsXYZ,
        std::vector<uint8_t>& valid,
        int parallelThreads = 0);

    // Equivalent requested-level corner access without materializing
    // candidate-sized output arrays. The visitor is invoked once per point;
    // its corner span follows the input array order and is valid only for the
    // duration of the call.
    static Stats visitTrilinearCornersLevelBlockingRequestedLevel(
        const std::vector<IChunkedArray*>& arrays,
        int level,
        const std::vector<cv::Vec3f>& levelCoords,
        void* visitorContext,
        TrilinearCornerPointVisitor visitor,
        int parallelThreads = 0,
        bool collectLocalityStats = false);

    // Fine-to-coarse fallback. Finer covered pixels are never overwritten by
    // coarser levels.
    static Stats samplePlaneFineToCoarse(IChunkedArray& array,
                                         int startLevel,
                                         const cv::Vec3f& origin,
                                         const cv::Vec3f& vxStep,
                                         const cv::Vec3f& vyStep,
                                         cv::Mat_<uint8_t>& out,
                                         cv::Mat_<uint8_t>& coverage,
                                         const Options& options = Options());

    static Stats sampleCoordsFineToCoarse(IChunkedArray& array,
                                          int startLevel,
                                          const cv::Mat_<cv::Vec3f>& coords,
                                          cv::Mat_<uint8_t>& out,
                                          cv::Mat_<uint8_t>& coverage,
                                          const Options& options = Options());

    // Coarse-to-fine progressive render. Coarser levels provide the first
    // available image; ready finer levels overwrite those pixels.
    static Stats samplePlaneCoarseToFine(IChunkedArray& array,
                                         int finestLevel,
                                         const cv::Vec3f& origin,
                                         const cv::Vec3f& vxStep,
                                         const cv::Vec3f& vyStep,
                                         cv::Mat_<uint8_t>& out,
                                         cv::Mat_<uint8_t>& coverage,
                                         const Options& options = Options());

    static Stats sampleCoordsCoarseToFine(IChunkedArray& array,
                                          int finestLevel,
                                          const cv::Mat_<cv::Vec3f>& coords,
                                          cv::Mat_<uint8_t>& out,
                                          cv::Mat_<uint8_t>& coverage,
                                          const Options& options = Options());
};

} // namespace vc::render
