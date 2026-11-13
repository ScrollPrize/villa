#include "vc/core/util/SheetNormalField.hpp"

#include <vc_reader/array.hpp>

// Header-only CPU implementation shared with vc_reader's Python sheet_normals.
#include <vc_reader/native/sheet_normal/normal.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{
constexpr int kWidth = vcr::sheet_normal::width; // 109 voxel patch
constexpr int kHalf = 54;                         // patch start = lattice point - 54
constexpr std::size_t kPatchCacheBytes = 512ULL * 1024 * 1024;
// CpuPatch owns six gradient arrays, each with `count` floats.
constexpr std::size_t kPatchBytes = sizeof(vcr::sheet_normal::CpuPatch) +
                                    6ULL * vcr::sheet_normal::count * sizeof(float);
constexpr std::size_t kPatchCacheCapacity = kPatchCacheBytes / kPatchBytes;
static_assert(kPatchCacheCapacity > 0, "The normal cache must hold at least one patch");

} // namespace

SheetNormalField::SheetNormalField(std::shared_ptr<vcr::Array> volume)
    : _volume(std::move(volume))
{
    if (_volume && (_volume->ndim() != 3 || _volume->itemsize() != 1))
        throw std::invalid_argument("sheet normals read a three-dimensional uint8 volume");
}

std::optional<cv::Vec3f> SheetNormalField::normalAt(const cv::Vec3f& world, bool continuous)
{
    if (!_volume || !std::isfinite(world[0]) || !std::isfinite(world[1]) || !std::isfinite(world[2])) {
        return std::nullopt;
    }
    const std::array<int, 3> lattice{static_cast<int>(std::lround(world[0] / 3.0f)) * 3,
                                     static_cast<int>(std::lround(world[1] / 3.0f)) * 3,
                                     static_cast<int>(std::lround(world[2] / 3.0f)) * 3};
    if (!continuous) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (auto it = _cache.find(lattice); it != _cache.end()) {
            return it->second;
        }
    }

    std::shared_ptr<const vcr::sheet_normal::CpuPatch> patch;
    if (continuous) {
        std::lock_guard lock(_mutex);
        if (auto it = _patchCache.find(lattice); it != _patchCache.end()) {
            patch = it->second;
        }
    }
    std::optional<cv::Vec3f> result;
    if (!patch) {
        // The 109 voxel cube around the lattice point, ZYX with x fastest; outside
        // the volume, and where no chunk is stored, it reads as the fill value.
        std::vector<uint8_t> image(static_cast<size_t>(kWidth) * kWidth * kWidth);
        const vcr::Index start{lattice[2] - kHalf, lattice[1] - kHalf, lattice[0] - kHalf};
        _volume->read_many({start}, {{kWidth, kWidth, kWidth}}, {image.data()});
        auto prepared = std::make_shared<vcr::sheet_normal::CpuPatch>();
        vcr::sheet_normal::CpuWorkspace workspace;
        workspace.prepare(image.data(), *prepared);
        patch = std::move(prepared);
        if (continuous) {
            std::lock_guard lock(_mutex);
            // Two workers may prepare the same patch concurrently. Keep
            // only one copy in the bounded cache; their values are identical.
            if (_patchCache.emplace(lattice, patch).second) {
                _patchOrder.push_back(lattice);
                if (_patchOrder.size() > kPatchCacheCapacity) {
                    _patchCache.erase(_patchOrder.front());
                    _patchOrder.pop_front();
                }
            }
        }
    }
    if (patch) {
        // Patch voxel 54 is estimator coordinate 48 (gradient grid starts at 6).
        const float center[4] = {
            48.0f + (continuous ? world[0] - lattice[0] : 0.0f),
            48.0f + (continuous ? world[1] - lattice[1] : 0.0f),
            48.0f + (continuous ? world[2] - lattice[2] : 0.0f), 1.0f};
        float out[4] = {};
        patch->evaluate(center, out);
        if (out[3] > 0.0f) result = cv::Vec3f(out[0], out[1], out[2]);
    }

    if (!continuous) {
        std::lock_guard<std::mutex> lock(_mutex);
        _cache[lattice] = result;
    }
    return result;
}
