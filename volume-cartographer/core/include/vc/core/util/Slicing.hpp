#pragma once

#include <opencv2/core.hpp>
#include <string>

#include <vc/core/cache/BlockPipeline.hpp>
#include <vc/core/util/Compositing.hpp>
#include <vc/core/types/Sampling.hpp>
#include <vc/core/types/Array3D.hpp>

// Forward declaration
namespace vc { class VcDataset; }

// Read interpolated 3D data from a zarr dataset via BlockPipeline
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::cache::BlockPipeline* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::cache::BlockPipeline* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);

// Overloads accepting vc::Sampling enum (supports Nearest, Trilinear, Tricubic)
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::cache::BlockPipeline* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::cache::BlockPipeline* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);

// Read a 3D area from a zarr dataset via BlockPipeline
void readArea3D(Array3D<uint8_t> &out, const cv::Vec3i& offset, vc::cache::BlockPipeline* cache, int level);
void readArea3D(Array3D<uint16_t> &out, const cv::Vec3i& offset, vc::cache::BlockPipeline* cache, int level);

// Composite rendering with configurable interpolation.
void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    vc::cache::BlockPipeline* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    vc::Sampling method = vc::Sampling::Nearest
);

// Bulk multi-slice read with trilinear interpolation.
void readMultiSlice(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::cache::BlockPipeline* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::BlockPipeline* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Single-threaded per-tile multi-slice sampler (called from within OMP thread).
void sampleTileSlices(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::cache::BlockPipeline* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::BlockPipeline* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Fused plane coordinate generation + sampling (eliminates intermediate coords Mat).
// origin, vx_step, vy_step define the affine plane in level-scaled coordinates.
// coord(i,j) = origin + vx_step * i + vy_step * j
void samplePlane(cv::Mat_<uint8_t>& out, vc::cache::BlockPipeline* cache, int level,
                 const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
                 int width, int height, vc::Sampling method);

// Unified composite-capable adaptive sampler with per-pixel level fallback.
// One entry point for plane/coords and composite/non-composite rendering:
//   - Plane mode: pass coords=nullptr and origin/vx_step/vy_step/planeNormal.
//   - Coords mode (QuadSurface): pass coords, leave origin/vx_step/vy_step/planeNormal null.
//   - Composite: pass normals (or planeNormal) with numLayers>1. numLayers=1
//     and any normal (or zero) collapses the loop to a single sample.
//   - compositeMethod: "max", "min", "mean", "median", "minabs", "alpha".
// Non-blocking: missing blocks render as lut[0] after exhausting fallback levels.
void sampleAdaptiveARGB32(
    uint32_t* outBuf, int outStride,
    vc::cache::BlockPipeline* cache,
    int desiredLevel, int numLevels,
    const cv::Mat_<cv::Vec3f>* coords,
    const cv::Vec3f* origin, const cv::Vec3f* vx_step, const cv::Vec3f* vy_step,
    const cv::Mat_<cv::Vec3f>* normals,
    const cv::Vec3f* planeNormal,
    int numLayers, int zStart, float zStep,
    int width, int height,
    const std::string& compositeMethod,
    const uint32_t lut[256],
    vc::Sampling method = vc::Sampling::Nearest,
    // Optional lighting: when lightParams->lightingEnabled is true the
    // composite output is modulated by a Lambertian diffuse factor.
    // Normal source (mesh vs. volume gradient) is selected via the params.
    const CompositeParams* lightParams = nullptr,
    // Optional per-pixel fallback-level output. When non-null, each pixel is
    // tagged with the *coarsest* pyramid-level offset used while sampling
    // (0 = desired level, 1..N = fallback depth). Stride is in bytes.
    uint8_t* levelOut = nullptr,
    int levelStride = 0,
    // When true, skip the per-frame chunk enumeration + sort + fetchInteractive
    // that the kernel normally runs before dispatching tiles. Intended for
    // callers that can prove the coords haven't changed since the last
    // render (e.g. QuadSurface gen cache hit) — the prior frame already
    // queued the needed blocks, so rerunning the enumeration is pure
    // overhead. No correctness impact on block residency: the per-sample
    // adaptive fallback still handles any block not yet loaded.
    bool skipPrefetch = false);

// Fused plane composite: inline coords + nearest-neighbor per layer + composite + LUT → ARGB32.
// No coord matrix allocation. For PlaneSurface composite rendering.
void samplePlaneCompositeARGB32(
    uint32_t* outBuf, int outStride,
    vc::cache::BlockPipeline* cache, int level,
    const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
    const cv::Vec3f& normal, float zStep, int zStart, int numLayers,
    int width, int height,
    const std::string& compositeMethod,
    const uint32_t lut[256]);

