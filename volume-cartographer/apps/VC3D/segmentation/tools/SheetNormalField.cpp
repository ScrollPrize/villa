#include "SheetNormalField.hpp"

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"

// Header-only CPU implementation shared with vc_reader's Python sheet_normals.
#include "sheet_normal/normal.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int kWidth = vcr::sheet_normal::width; // 109 voxel patch
constexpr int kHalf = 54;                         // patch start = lattice point - 54

// Callers walk many lines at once on their own threads. Volume::readZYX opens
// an OpenMP team as wide as the machine, so each walk thread would start its
// own team for one 109 voxel read: threads times threads on the cores. Reads
// here use one OpenMP thread; the voxels read are the same.
class OneOmpThread
{
public:
    OneOmpThread() : _previous(std::max(1, omp_get_max_threads())) { omp_set_num_threads(1); }
    ~OneOmpThread() { omp_set_num_threads(_previous); }
    OneOmpThread(const OneOmpThread&) = delete;
    OneOmpThread& operator=(const OneOmpThread&) = delete;

private:
    int _previous;
};

// The 109 voxel cube at `start` (ZYX) of level 0 into `image`, ZYX with x
// fastest, as Volume::readZYX would give it: zero outside the volume, the
// fill value where a chunk holds none. Each chunk the cube touches is looked
// up once and its rows are copied whole, which takes the chunk cache's lock
// a few times per cube instead of once or twice per voxel row. False when the
// volume is not plain uint8 level 0 chunks; the caller then uses readZYX.
bool readCube(Volume& volume, const std::array<int, 3>& start, std::vector<uint8_t>& image)
{
    using vc::render::ChunkStatus;
    auto* array = volume.chunkedCache();
    if (!array || !volume.hasScaleLevel(0) || array->dtype() != vc::render::ChunkDtype::UInt8) {
        return false;
    }
    const auto shape = array->shape(0);
    const auto chunk = array->chunkShape(0);
    if (chunk[0] <= 0 || chunk[1] <= 0 || chunk[2] <= 0) {
        return false;
    }
    std::fill(image.begin(), image.end(), uint8_t{0});
    int lo[3], hi[3];
    for (int a = 0; a < 3; ++a) {
        lo[a] = std::max(0, start[a]);
        hi[a] = std::min(shape[a] - 1, start[a] + kWidth - 1);
        if (lo[a] > hi[a]) {
            return true;
        }
    }
    const uint8_t fill = static_cast<uint8_t>(std::clamp(array->fillValue(), 0.0, 255.0));
    const size_t strideY = static_cast<size_t>(chunk[2]);
    const size_t strideZ = static_cast<size_t>(chunk[1]) * strideY;
    for (int cz = lo[0] / chunk[0]; cz <= hi[0] / chunk[0]; ++cz) {
        for (int cy = lo[1] / chunk[1]; cy <= hi[1] / chunk[1]; ++cy) {
            for (int cx = lo[2] / chunk[2]; cx <= hi[2] / chunk[2]; ++cx) {
                const auto result = array->getChunkBlocking(0, cz, cy, cx);
                if (result.status == ChunkStatus::Error) {
                    throw std::runtime_error("Volume::read failed fetching chunk 0/" + std::to_string(cz) + "/" +
                                             std::to_string(cy) + "/" + std::to_string(cx) + ": " +
                                             (result.error.empty() ? "chunk fetch failed" : result.error));
                }
                const uint8_t* data = (result.status == ChunkStatus::Data && result.bytes)
                    ? reinterpret_cast<const uint8_t*>(result.bytes->data())
                    : nullptr;
                const int z0 = std::max(lo[0], cz * chunk[0]), z1 = std::min(hi[0], (cz + 1) * chunk[0] - 1);
                const int y0 = std::max(lo[1], cy * chunk[1]), y1 = std::min(hi[1], (cy + 1) * chunk[1] - 1);
                const int x0 = std::max(lo[2], cx * chunk[2]), x1 = std::min(hi[2], (cx + 1) * chunk[2] - 1);
                const size_t run = static_cast<size_t>(x1 - x0 + 1);
                for (int z = z0; z <= z1; ++z) {
                    for (int y = y0; y <= y1; ++y) {
                        uint8_t* dst = image.data() +
                                       (static_cast<size_t>(z - start[0]) * kWidth + (y - start[1])) * kWidth +
                                       (x0 - start[2]);
                        if (!data) {
                            std::memset(dst, fill, run);
                            continue;
                        }
                        const uint8_t* src = data + static_cast<size_t>(z - cz * chunk[0]) * strideZ +
                                             static_cast<size_t>(y - cy * chunk[1]) * strideY +
                                             static_cast<size_t>(x0 - cx * chunk[2]);
                        std::memcpy(dst, src, run);
                    }
                }
            }
        }
    }
    return true;
}
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

    std::vector<uint8_t> image(static_cast<size_t>(kWidth) * kWidth * kWidth);
    std::optional<cv::Vec3f> result;
    const std::array<int, 3> start{lattice[2] - kHalf, lattice[1] - kHalf, lattice[0] - kHalf};
    bool read = readCube(*_volume, start, image);
    if (!read) {
        Array3D<uint8_t> patch({static_cast<size_t>(kWidth), static_cast<size_t>(kWidth), static_cast<size_t>(kWidth)});
        {
            const OneOmpThread oneThread;
            read = _volume->readZYX(patch, start, 0);
        }
        if (read) {
            // Array3D is column-major (z fastest); the estimator wants ZYX with x fastest.
            for (int z = 0; z < kWidth; ++z) {
                for (int y = 0; y < kWidth; ++y) {
                    for (int x = 0; x < kWidth; ++x) {
                        image[(static_cast<size_t>(z) * kWidth + y) * kWidth + x] = patch(z, y, x);
                    }
                }
            }
        }
    }
    if (read) {
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
