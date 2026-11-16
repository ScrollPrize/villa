#include "helirec/reference.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "helirec/block_spectra.hpp"
#include "helirec/zarr.hpp"

namespace helirec {

struct ReferenceMask::Impl {
    std::string location;
    int scale = 0;
    std::unique_ptr<ZarrArray> level;
    std::array<int, 3> f{1, 1, 1}, shift{0, 0, 0};
    std::array<std::int64_t, 3> n{0, 0, 0};          // the level's shape
    std::map<int, std::vector<std::uint8_t>> buffers;
    // a reference with another number of slices: its level-0 slices per slice of the volume, and the level's place
    // along z (level 0 = z_scale * index + z_offset)
    double slice_ratio = 1.0, z_scale = 1.0, z_offset = 0.0;
};

ReferenceMask::ReferenceMask(const std::string& location, int scale, const std::array<std::int64_t, 3>& shape,
                             const std::string& cache_dir)
    : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.location = location;
    I.scale = scale;
    const auto levels = ome_zarr_levels(location, cache_dir);
    const OmeZarrLevel* chosen = nullptr;
    for (const auto& l : levels)
        if (l.level == scale) chosen = &l;
    if (!chosen) throw Error("reference volume " + location + ": no pyramid level " + std::to_string(scale));
    auto text = [](const auto& s) {
        std::string t = "(";
        for (std::size_t i = 0; i < s.size(); ++i) t += (i ? ", " : "") + std::to_string(s[i]);
        return t + ")";
    };
    const ZarrArray full = ZarrArray::open(levels.front().location, cache_dir);
    const ZarrIndex fs = full.meta().shape;
    const bool slices_close = shape[0] > 0 && std::abs(static_cast<double>(fs[0] - shape[0])) <= kMaxSliceMismatch * static_cast<double>(shape[0]);
    if (fs[1] != shape[1] || fs[2] != shape[2] || !slices_close)
        throw Error("reference volume " + location + " has shape " + text(fs) + ", the reconstruction " + text(shape) +
                    " (z, y, x): its mask can only be reused on the same grid (the number of slices may differ by " +
                    std::to_string(static_cast<int>(kMaxSliceMismatch * 100)) + " %)");
    if (fs[0] != shape[0]) I.slice_ratio = static_cast<double>(fs[0]) / static_cast<double>(shape[0]);
    I.z_scale = chosen->scale[0];
    I.z_offset = chosen->offset[0];
    I.level = std::make_unique<ZarrArray>(scale == 0 ? full : ZarrArray::open(chosen->location, cache_dir));
    if (I.level->meta().dtype != "|u1") throw Error("reference volume " + location + " is " + I.level->meta().dtype + ", not uint8");
    for (int a = 0; a < 3; ++a) {
        // level0 = scale * index + offset: volume voxel v belongs to the level voxel nearest (v - offset) / scale,
        // which is (v + shift) / f for a whole factor f
        const double s = chosen->scale[a];
        I.f[a] = static_cast<int>(std::llround(s));
        if (I.f[a] < 1 || std::abs(s - I.f[a]) > 1e-6 * s)
            throw Error("reference volume " + location + ": level " + std::to_string(scale) + " is not a whole number of "
                        "level-0 voxels per voxel");
        I.shift[a] = static_cast<int>(std::floor((I.f[a] - 1) / 2.0 - chosen->offset[a] + 1e-9));
        I.n[a] = I.level->meta().shape[a];
    }
}

ReferenceMask::~ReferenceMask() = default;

const std::string& ReferenceMask::location() const { return impl_->location; }
int ReferenceMask::scale() const { return impl_->scale; }
std::array<int, 3> ReferenceMask::factors() const { return impl_->f; }
std::array<std::int64_t, 3> ReferenceMask::level_shape() const { return impl_->n; }
double ReferenceMask::slice_ratio() const { return impl_->slice_ratio; }

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
    if (I.slice_ratio != 1.0) {
        // Another number of slices: one mask row per f[0] slices of the volume, each the reference slice nearest
        // where the middle of those slices falls when the reference's slices are spread evenly over the volume's.
        const int f = I.f[0], za = (std::max(z0, 0) / f) * f, rows = (z1 - 1 - za) / f + 1;
        std::vector<std::int64_t> src(rows);
        for (int k = 0; k < rows; ++k) {
            const double middle = za + k * f + (f - 1) / 2.0;
            const double level0 = (middle + 0.5) * I.slice_ratio - 0.5;
            src[k] = std::clamp<std::int64_t>(static_cast<std::int64_t>(std::floor((level0 - I.z_offset) / I.z_scale + 0.5)),
                                              0, I.n[0] - 1);
        }
        const std::int64_t first = src.front(), count = src.back() - first + 1;   // (the map is monotonic)
        const std::size_t plane = static_cast<std::size_t>(cn[1] * cn[2]);
        std::vector<std::uint8_t> slab(static_cast<std::size_t>(count) * plane);
        I.level->read({first, c0[1], c0[2]}, {count, cn[1], cn[2]}, slab.data());
        buf.resize(static_cast<std::size_t>(rows) * plane);
        for (int k = 0; k < rows; ++k)
            std::copy_n(slab.data() + static_cast<std::size_t>(src[k] - first) * plane, plane, buf.data() + static_cast<std::size_t>(k) * plane);
        cn[0] = rows;
        off[0] = z0 - za;
    } else {
        buf.resize(static_cast<std::size_t>(cn[0] * cn[1] * cn[2]));
        I.level->read({c0[0], c0[1], c0[2]}, {cn[0], cn[1], cn[2]}, buf.data());
    }
    OutputMask m;
    m.data = buf.data();
    m.nz = static_cast<int>(cn[0]); m.ny = static_cast<int>(cn[1]); m.nx = static_cast<int>(cn[2]);
    m.fz = I.f[0]; m.fy = I.f[1]; m.fx = I.f[2];
    m.oz = off[0]; m.oy = off[1]; m.ox = off[2];
    return m;
}

// ------------------------------------------------------------------------------------------ RegisteredVolume
struct RegisteredVolume::Impl {
    std::string location;
    int level = 0;
    std::unique_ptr<ZarrArray> array;
    std::array<double, 12> to_level{};   // reconstruction voxel index (z, y, x, 1) -> level voxel index
    bool u8 = true;
};

RegisteredVolume::RegisteredVolume(const std::string& location, int level, const std::array<double, 12>& affine,
                                   const std::string& cache_dir)
    : impl_(std::make_unique<Impl>()) {
    auto& I = *impl_;
    I.location = location;
    I.level = level;
    const auto levels = ome_zarr_levels(location, cache_dir);
    const OmeZarrLevel* chosen = nullptr;
    for (const auto& l : levels)
        if (l.level == level) chosen = &l;
    if (!chosen) throw Error("reference volume " + location + ": no pyramid level " + std::to_string(level));
    I.array = std::make_unique<ZarrArray>(ZarrArray::open(chosen->location, cache_dir));
    I.u8 = I.array->meta().dtype == "|u1";
    // level 0 index = scale * level index + offset
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            I.to_level[r * 4 + c] = (affine[r * 4 + c] - (c == 3 ? chosen->offset[r] : 0.0)) / chosen->scale[r];
}

RegisteredVolume::~RegisteredVolume() = default;

const std::string& RegisteredVolume::location() const { return impl_->location; }
int RegisteredVolume::level() const { return impl_->level; }
std::array<std::int64_t, 3> RegisteredVolume::level_shape() const { return impl_->array->meta().shape; }

std::array<double, 12> RegisteredVolume::block_affine(const std::array<double, 3>& origin, double step) const {
    const auto& t = impl_->to_level;
    std::array<double, 12> m{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) m[r * 4 + c] = t[r * 4 + c] * step;
        m[r * 4 + 3] = t[r * 4] * origin[0] + t[r * 4 + 1] * origin[1] + t[r * 4 + 2] * origin[2] + t[r * 4 + 3];
    }
    return m;
}

double RegisteredVolume::read(const std::array<double, 3>& origin, double step, const std::array<int, 3>& shape,
                              float* out, int interp, Device device) const {
    const auto& I = *impl_;
    if (shape[0] < 1 || shape[1] < 1 || shape[2] < 1) throw Error("RegisteredVolume::read: empty block");
    std::array<double, 12> m = block_affine(origin, step);
    // the level's voxels the block's samples need: the bounds of its corners, plus the interpolation's reach
    const ZarrIndex full = I.array->meta().shape;
    ZarrIndex lo{}, n{};
    bool empty = false;
    for (int r = 0; r < 3; ++r) {
        double a = 1e300, b = -1e300;
        for (int corner = 0; corner < 8; ++corner) {
            double v = m[r * 4 + 3];
            for (int c = 0; c < 3; ++c)
                if (corner >> c & 1) v += m[r * 4 + c] * (shape[c] - 1);
            a = std::min(a, v);
            b = std::max(b, v);
        }
        const std::int64_t first = std::max<std::int64_t>(0, static_cast<std::int64_t>(std::floor(a)) - 3);
        const std::int64_t last = std::min<std::int64_t>(full[r] - 1, static_cast<std::int64_t>(std::ceil(b)) + 3);
        lo[r] = first;
        n[r] = last - first + 1;
        empty = empty || n[r] < 1;
        m[r * 4 + 3] -= static_cast<double>(first);
    }
    if (empty) {   // nothing of the reference there: one voxel, taken as 0
        lo = {0, 0, 0};
        n = {1, 1, 1};
    }
    if (n[0] > INT32_MAX || n[1] > INT32_MAX || n[2] > INT32_MAX) throw Error("RegisteredVolume::read: block too large");
    const std::size_t voxels = static_cast<std::size_t>(n[0] * n[1] * n[2]);
    std::vector<float> src(voxels, 0.f);
    std::size_t nonzero = 0;
    if (empty) {
    } else if (I.u8) {
        std::vector<std::uint8_t> raw(voxels);
        I.array->read(lo, n, raw.data());
#pragma omp parallel for schedule(static) reduction(+ : nonzero)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(voxels); ++i) {
            src[i] = static_cast<float>(raw[i]);
            nonzero += raw[i] != 0;
        }
    } else {
        I.array->read(lo, n, src.data());
#pragma omp parallel for schedule(static) reduction(+ : nonzero)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(voxels); ++i) nonzero += src[i] != 0.f;
    }
    const int sz = static_cast<int>(n[0]), sy = static_cast<int>(n[1]), sx = static_cast<int>(n[2]);
    if (device == Device::CPU) {
        cpu::resample_affine(src.data(), sz, sy, sx, m.data(), out, shape[0], shape[1], shape[2], interp);
    } else {
#ifdef HELIREC_WITH_CUDA
        cuda::resample_affine_host(src.data(), sz, sy, sx, m.data(), out, shape[0], shape[1], shape[2], interp);
#else
        throw Error("built without CUDA");
#endif
    }
    return static_cast<double>(nonzero) / static_cast<double>(voxels);
}

}  // namespace helirec
