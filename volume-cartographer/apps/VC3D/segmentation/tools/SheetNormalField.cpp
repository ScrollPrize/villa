#include "SheetNormalField.hpp"

#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"

// Header-only CPU implementation shared with vc_reader's Python sheet_normals.
#include "sheet_normal/normal.hpp"

#include <cmath>
#include <vector>

namespace
{
constexpr int kWidth = vcr::sheet_normal::width; // 109 voxel patch
constexpr int kHalf = 54;                         // patch start = lattice point - 54
} // namespace

SheetNormalField::SheetNormalField(std::shared_ptr<Volume> volume)
    : _volume(std::move(volume))
{
}

std::optional<cv::Vec3f> SheetNormalField::normalAt(const cv::Vec3f& world)
{
    if (!_volume || !std::isfinite(world[0]) || !std::isfinite(world[1]) || !std::isfinite(world[2])) {
        return std::nullopt;
    }
    const std::array<int, 3> lattice{static_cast<int>(std::lround(world[0] / 3.0f)) * 3,
                                     static_cast<int>(std::lround(world[1] / 3.0f)) * 3,
                                     static_cast<int>(std::lround(world[2] / 3.0f)) * 3};
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (auto it = _cache.find(lattice); it != _cache.end()) {
            return it->second;
        }
    }

    Array3D<uint8_t> patch({static_cast<size_t>(kWidth), static_cast<size_t>(kWidth), static_cast<size_t>(kWidth)});
    std::optional<cv::Vec3f> result;
    if (_volume->readZYX(patch, {lattice[2] - kHalf, lattice[1] - kHalf, lattice[0] - kHalf}, 0)) {
        // Array3D is column-major (z fastest); the estimator wants ZYX with x fastest.
        std::vector<uint8_t> image(static_cast<size_t>(kWidth) * kWidth * kWidth);
        for (int z = 0; z < kWidth; ++z) {
            for (int y = 0; y < kWidth; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    image[(static_cast<size_t>(z) * kWidth + y) * kWidth + x] = patch(z, y, x);
                }
            }
        }
        // The lattice point is voxel 54 of the patch, which is 48 in the
        // estimator's site coordinates (its grid starts 6 voxels in).
        const float center[4] = {48.0f, 48.0f, 48.0f, 1.0f};
        float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        vcr::sheet_normal::cpu<uint8_t>(image.data(), center, out, 1);
        if (out[3] > 0.0f) {
            result = cv::Vec3f(out[0], out[1], out[2]);
        }
    }

    std::lock_guard<std::mutex> lock(_mutex);
    _cache[lattice] = result;
    return result;
}
