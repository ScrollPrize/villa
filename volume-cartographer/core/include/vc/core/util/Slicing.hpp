#pragma once

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include <opencv2/core.hpp>
#include <string>

#include <vc/core/util/ChunkCache.hpp>
#include <vc/core/util/Compositing.hpp>

// Forward declaration
namespace z5 { class Dataset; }

//NOTE depending on request this might load a lot (the whole array) into RAM
// void readInterpolated3D(xt::xarray<uint8_t> &out, z5::Dataset *ds, const xt::xarray<float> &coords, ChunkCache *cache = nullptr);
void readInterpolated3D(cv::Mat_<uint8_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint8_t> *cache = nullptr, bool nearest_neighbor=false);
void readInterpolated3D(cv::Mat_<uint16_t> &out, z5::Dataset *ds, const cv::Mat_<cv::Vec3f> &coords, ChunkCache<uint16_t> *cache = nullptr, bool nearest_neighbor=false);
//template <typename T>
//void readArea3D(xt::xtensor<T,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<T> *cache) { throw std::runtime_error("missing implementation"); }
void readArea3D(xt::xtensor<uint8_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<uint8_t> *cache);
void readArea3D(xt::xtensor<uint16_t,3,xt::layout_type::column_major> &out, const cv::Vec3i& offset, z5::Dataset *ds, ChunkCache<uint16_t> *cache);

// Fast composite rendering cache - holds chunks needed for composite rendering
// without mutex overhead. Designed for single-threaded composite rendering.
class FastCompositeCache {
public:
    FastCompositeCache() = default;
    ~FastCompositeCache() = default;

    // Clear the cache
    void clear();

    // Set the dataset this cache is for
    void setDataset(z5::Dataset* ds);

    // Get a chunk, loading it if necessary. Returns nullptr if out of bounds.
    // No mutex - assumes single-threaded access during composite rendering.
    const xt::xarray<uint8_t>* getChunk(int ix, int iy, int iz);

    // Get chunk dimensions
    int chunkSizeX() const { return _cw; }
    int chunkSizeY() const { return _ch; }
    int chunkSizeZ() const { return _cd; }

    // Get dataset dimensions
    int datasetSizeX() const { return _sx; }
    int datasetSizeY() const { return _sy; }
    int datasetSizeZ() const { return _sz; }

private:
    z5::Dataset* _ds = nullptr;
    int _cw = 0, _ch = 0, _cd = 0;  // Chunk dimensions
    int _sx = 0, _sy = 0, _sz = 0;  // Dataset dimensions
    int _chunksX = 0, _chunksY = 0, _chunksZ = 0;  // Number of chunks

    // Simple map for chunk storage - no shared_ptr overhead
    std::unordered_map<uint64_t, std::unique_ptr<xt::xarray<uint8_t>>> _chunks;

    uint64_t chunkKey(int ix, int iy, int iz) const {
        return (uint64_t(ix) << 40) | (uint64_t(iy) << 20) | uint64_t(iz);
    }
};

// Fast composite rendering - nearest neighbor only, no mutex, inline caching
// Returns directly into output matrix
void readCompositeFast(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Mat_<cv::Vec3f>& normals,
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    FastCompositeCache& cache
);

// Fast composite rendering with constant normal (optimized for plane surfaces)
// Avoids per-pixel normal lookup and uses pre-computed layer offsets
void readCompositeFastConstantNormal(
    cv::Mat_<uint8_t>& out,
    z5::Dataset* ds,
    const cv::Mat_<cv::Vec3f>& baseCoords,
    const cv::Vec3f& normal,  // Single constant normal for all pixels
    float zStep,
    int zStart, int zEnd,
    const CompositeParams& params,
    FastCompositeCache& cache
);
