#include "helirec/reference.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include <vc_reader/array.hpp>
#include <vc_reader/zarr.hpp>

namespace helirec {

struct ReferenceMask::Impl {
    std::string location;
    int scale = 0;
    vcr::zarr::OpenedArray level;
    std::array<int, 3> f{1, 1, 1}, shift{0, 0, 0};
    std::array<std::int64_t, 3> n{0, 0, 0};          // the level's shape
    std::map<int, std::vector<std::uint8_t>> buffers;
};

ReferenceMask::ReferenceMask(const std::string& location, int scale, const std::array<std::int64_t, 3>& shape,
                             const std::string& cache_dir)
    : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.location = location;
    I.scale = scale;
    vcr::zarr::OpenOptions options;
    options.access = vcr::zarr::resolveAccess(location);
    if (!cache_dir.empty()) options.cacheDirs = {cache_dir};
    const vcr::zarr::PyramidLevelInfo* full = nullptr;
    const vcr::zarr::PyramidLevelInfo* chosen = nullptr;
    const auto levels = vcr::zarr::pyramidLevels(location, options);
    for (const auto& l : levels) {
        if (l.level == 0) full = &l;
        if (l.level == scale) chosen = &l;
    }
    if (!full) throw Error("reference volume " + location + ": no level 0");
    if (!chosen) throw Error("reference volume " + location + ": no pyramid level " + std::to_string(scale));
    auto text = [](const auto& s) {
        std::string t = "(";
        for (std::size_t i = 0; i < s.size(); ++i) t += (i ? ", " : "") + std::to_string(s[i]);
        return t + ")";
    };
    if (full->meta.shape.size() != 3 || !std::equal(shape.begin(), shape.end(), full->meta.shape.begin()))
        throw Error("reference volume " + location + " has shape " + text(full->meta.shape) + ", the reconstruction " +
                    text(shape) + " (z, y, x): its mask can only be reused on the same grid");
    if (chosen->meta.dtype != "|u1") throw Error("reference volume " + location + " is " + chosen->meta.dtype + ", not uint8");
    for (int a = 0; a < 3; ++a) {
        // level0 = scale * index + offset: volume voxel v belongs to the level voxel nearest (v - offset) / scale,
        // which is (v + shift) / f for a whole factor f
        const double s = chosen->placement.scale[a];
        I.f[a] = static_cast<int>(std::llround(s));
        if (I.f[a] < 1 || std::abs(s - I.f[a]) > 1e-6 * s)
            throw Error("reference volume " + location + ": level " + std::to_string(scale) + " is not a whole number of "
                        "level-0 voxels per voxel");
        I.shift[a] = static_cast<int>(std::floor((I.f[a] - 1) / 2.0 - chosen->placement.offset[a] + 1e-9));
        I.n[a] = chosen->meta.shape[a];
    }
    I.level = vcr::zarr::openPyramidLevel(location, scale, options);
}

ReferenceMask::~ReferenceMask() = default;

const std::string& ReferenceMask::location() const { return impl_->location; }
int ReferenceMask::scale() const { return impl_->scale; }
std::array<int, 3> ReferenceMask::factors() const { return impl_->f; }
std::array<std::int64_t, 3> ReferenceMask::level_shape() const { return impl_->n; }

OutputMask ReferenceMask::read(int z0, int z1, const SliceRoi& roi, int slot) {
    auto& I = *impl_;
    const int lo[3] = {z0, roi.y0, roi.x0}, hi[3] = {z1, roi.y1, roi.x1};
    std::int64_t c0[3], cn[3];
    int off[3];
    for (int a = 0; a < 3; ++a) {
        if (hi[a] <= lo[a]) throw Error("ReferenceMask::read: empty region");
        auto index = [&](int v) { return std::min<std::int64_t>(std::max(v + I.shift[a], 0) / I.f[a], I.n[a] - 1); };
        c0[a] = index(lo[a]);
        cn[a] = index(hi[a] - 1) - c0[a] + 1;
        off[a] = lo[a] + I.shift[a] - static_cast<int>(c0[a]) * I.f[a];
    }
    auto& buf = I.buffers[slot];
    buf.resize(static_cast<std::size_t>(cn[0] * cn[1] * cn[2]));
    I.level.array->read_many({vcr::Index{c0[0], c0[1], c0[2]}}, {vcr::Index{cn[0], cn[1], cn[2]}}, {buf.data()});
    OutputMask m;
    m.data = buf.data();
    m.nz = static_cast<int>(cn[0]); m.ny = static_cast<int>(cn[1]); m.nx = static_cast<int>(cn[2]);
    m.fz = I.f[0]; m.fy = I.f[1]; m.fx = I.f[2];
    m.oz = off[0]; m.oy = off[1]; m.ox = off[2];
    return m;
}

}  // namespace helirec
