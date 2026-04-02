#pragma once

#include <opencv2/core.hpp>
#include <string>

#include <vc/core/cache/TieredChunkCache.hpp>
#include <vc/core/util/Compositing.hpp>
#include <vc/core/types/Sampling.hpp>
#include <vc/core/types/Array3D.hpp>

// Forward declaration
namespace vc { class VcDataset; }

// Read interpolated 3D data from a zarr dataset via TieredChunkCache
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);

// Overloads accepting vc::Sampling enum (supports Nearest, Trilinear, Tricubic)
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::cache::TieredChunkCache* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);

// Read a 3D area from a zarr dataset via TieredChunkCache
void readArea3D(Array3D<uint8_t> &out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level);
void readArea3D(Array3D<uint16_t> &out, const cv::Vec3i& offset, vc::cache::TieredChunkCache* cache, int level);

// Composite rendering with configurable interpolation.
void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    vc::cache::TieredChunkCache* cache,
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
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Single-threaded per-tile multi-slice sampler (called from within OMP thread).
void sampleTileSlices(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::cache::TieredChunkCache* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Fused plane coordinate generation + sampling (eliminates intermediate coords Mat).
// origin, vx_step, vy_step define the affine plane in level-scaled coordinates.
// coord(i,j) = origin + vx_step * i + vy_step * j
void samplePlane(cv::Mat_<uint8_t>& out, vc::cache::TieredChunkCache* cache, int level,
                 const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
                 int width, int height, vc::Sampling method);

// Fused plane sampling + LUT: samples uint8 voxels and writes ARGB32 directly
// via lut[voxelValue], eliminating the intermediate cv::Mat and second pass.
// outBuf must point to width*height uint32_t pixels (row-major, stride in uint32_t units).
void samplePlaneARGB32(uint32_t* outBuf, int outStride,
                       vc::cache::TieredChunkCache* cache, int level,
                       const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
                       int width, int height, vc::Sampling method,
                       const uint32_t lut[256]);

