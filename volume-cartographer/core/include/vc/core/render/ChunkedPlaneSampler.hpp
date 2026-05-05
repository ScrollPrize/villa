#pragma once

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/Sampling.hpp"

#include <opencv2/core.hpp>

#include <cstdint>

namespace vc::render {

class ChunkedPlaneSampler {
public:
    struct Options {
        Options()
            : sampling(vc::Sampling::Nearest)
            , tileSize(32)
        {
        }
        Options(vc::Sampling sampling_, int tileSize_)
            : sampling(sampling_)
            , tileSize(tileSize_)
        {
        }

        vc::Sampling sampling;
        int tileSize;
    };

    struct Stats {
        int coveredPixels = 0;
        int requestedChunks = 0;
        int errorChunks = 0;
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
