#pragma once

#include <xtensor/containers/xarray.hpp>
#include <opencv2/core.hpp>
#include <string>

#include <vc/core/util/ChunkCache.hpp>
#include <vc/core/util/Compositing.hpp>

// Forward declaration
namespace z5 { class Dataset; }

// Read interpolated 3D data from a z5 dataset
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache, bool nearest_neighbor=false);

// Read a 3D area from a z5 dataset
void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<uint8_t> *cache);
void readArea3D(xt::xtensor<uint16_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<uint16_t> *cache);

// Fast composite rendering - nearest neighbor only, uses ChunkCache directly
void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache
);

// Fast composite rendering with constant normal (optimized for plane surfaces)
void readCompositeFastConstantNormal(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Vec3f& normal,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    ChunkCache<uint8_t>& cache
);

// Bulk multi-slice read with trilinear interpolation.
// Samples basePoints + offsets[i] * stepDirs for each offset, returning one Mat per offset.
// Does a single prefetch pass covering all slices, then samples in parallel.
// basePoints/stepDirs use (X,Y,Z) in Vec3f[0],[1],[2] (same convention as readInterpolated3D coords).
void readMultiSlice(
    std::vector<cv::Mat_<uint8_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint8_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void readMultiSlice(
    std::vector<cv::Mat_<uint16_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint16_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

// Single-threaded per-tile multi-slice sampler (called from within OMP thread).
// Same trilinear math as readMultiSlice but no internal OMP parallelism.
void sampleTileSlices(
    std::vector<cv::Mat_<uint8_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint8_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);

void sampleTileSlices(
    std::vector<cv::Mat_<uint16_t>>& out,
    z5::Dataset* ds,
    ChunkCache<uint16_t>* cache,
    const cv::Mat_<cv::Vec3f>& basePoints,
    const cv::Mat_<cv::Vec3f>& stepDirs,
    const std::vector<float>& offsets
);
