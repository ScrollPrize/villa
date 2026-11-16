#pragma once
// Zarr volumes read through vc_reader (local, http(s)://, s3://; an optional disk cache): the levels of an
// OME-Zarr pyramid, or the one level of a plain array, and regions of one level.
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vc_reader/array.hpp"
#include "vc_reader/zarr.hpp"
#include "vc_tomography/common.hpp"

namespace vc_tomography::io {

inline vcr::zarr::OpenOptions volume_open_options(const std::string& cache_dir, bool offline = false) {
    vcr::zarr::OpenOptions o;
    if (!cache_dir.empty()) o.cacheDirs = {cache_dir};
    o.offline = offline;
    return o;
}

// open(options): online; where that fails and there is a cache, from the cache alone (a remote volume read before,
// its server out of reach), and the first error if that fails too
template <class Open>
auto open_volume(const std::string& cache_dir, Open&& open) {
    try {
        return open(volume_open_options(cache_dir));
    } catch (const std::exception& online) {
        if (cache_dir.empty()) throw;
        try {
            return open(volume_open_options(cache_dir, true));
        } catch (const std::exception&) {
            throw IoError(online.what());
        }
    }
}

// One level: voxel i lies at scale * i + offset in level-0 voxels, per axis (z, y, x) (vcr::zarr::levelPlacement:
// a level that declares no translation is block-reduced, offset (scale - 1) / 2).
struct VolumeLevel {
    int level = 0;
    std::string location;   // of the level's array
    std::array<double, 3> scale{1, 1, 1}, offset{0, 0, 0};
    std::array<std::int64_t, 3> shape{0, 0, 0};
    std::string dtype;      // numpy typestr
};

inline std::vector<VolumeLevel> volume_levels(const std::string& location, const std::string& cache_dir = {}) {
    std::vector<VolumeLevel> out;
    const auto levels = open_volume(cache_dir, [&](const vcr::zarr::OpenOptions& o) { return vcr::zarr::pyramidLevels(location, o); });
    for (const vcr::zarr::PyramidLevelInfo& l : levels) {
        if (l.meta.shape.size() != 3) throw Error("zarr: " + location + " is not a 3-D volume");
        VolumeLevel v;
        v.level = l.level;
        v.location = l.path.empty() ? location : location + "/" + l.path;
        v.scale = l.placement.scale, v.offset = l.placement.offset;
        v.shape = {l.meta.shape[0], l.meta.shape[1], l.meta.shape[2]};
        v.dtype = l.meta.dtype;
        out.push_back(std::move(v));
    }
    if (out.empty()) throw Error("zarr: no array at " + location);
    return out;
}

// One 3-D array; read() fills [start, start + shape) in C order (0 where nothing is stored). Thread-safe.
class VolumeArray {
  public:
    VolumeArray(const std::string& location, const std::string& cache_dir = {})
        : opened_(open_volume(cache_dir, [&](const vcr::zarr::OpenOptions& o) { return vcr::zarr::openArray(location, o); })) {
        if (opened_.meta.shape.size() != 3) throw Error("zarr: " + location + " is not a 3-D volume");
    }
    std::array<std::int64_t, 3> shape() const { return {opened_.meta.shape[0], opened_.meta.shape[1], opened_.meta.shape[2]}; }
    const std::string& dtype() const { return opened_.meta.dtype; }
    void read(const std::array<std::int64_t, 3>& start, const std::array<std::int64_t, 3>& shape, void* out) const {
        opened_.array->read_many({vcr::Index{start[0], start[1], start[2]}}, {vcr::Index{shape[0], shape[1], shape[2]}},
                                 {static_cast<std::uint8_t*>(out)});
    }

  private:
    vcr::zarr::OpenedArray opened_;
};

}  // namespace vc_tomography::io
