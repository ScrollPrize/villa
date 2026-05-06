#pragma once

#include <opencv2/core.hpp>
#include <string>

#include <vc/core/render/IChunkedArray.hpp>
#include <vc/core/util/Compositing.hpp>
#include <vc/core/types/Sampling.hpp>

// Read interpolated 3D data from a chunked zarr source.
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::render::IChunkedArray* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::render::IChunkedArray* cache, int level, const cv::Mat_<cv::Vec3f> &coords, bool nearest_neighbor=false);

// Overloads accepting vc::Sampling enum (supports Nearest, Trilinear, Tricubic)
void readInterpolated3D(cv::Mat_<uint8_t> &out, vc::render::IChunkedArray* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);
void readInterpolated3D(cv::Mat_<uint16_t> &out, vc::render::IChunkedArray* cache, int level, const cv::Mat_<cv::Vec3f> &coords, vc::Sampling method);

// Composite rendering with configurable interpolation.
void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    vc::render::IChunkedArray* cache,
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
    vc::render::IChunkedArray* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::render::IChunkedArray* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Single-threaded per-tile multi-slice sampler (called from within OMP thread).
void sampleTileSlices(
    std::vector<cv::Mat_<uint8_t>>& out,
    vc::render::IChunkedArray* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    vc::render::IChunkedArray* cache,
    int level,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Fused plane coordinate generation + sampling (eliminates intermediate coords Mat).
// origin, vx_step, vy_step define the affine plane in level-scaled coordinates.
// coord(i,j) = origin + vx_step * i + vy_step * j
void samplePlane(cv::Mat_<uint8_t>& out, vc::render::IChunkedArray* cache, int level,
                 const cv::Vec3f& origin, const cv::Vec3f& vx_step, const cv::Vec3f& vy_step,
                 int width, int height, vc::Sampling method);
