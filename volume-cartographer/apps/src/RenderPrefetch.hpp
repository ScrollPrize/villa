#pragma once

#include "vc/core/render/IChunkedArray.hpp"
#include "vc/core/util/Slicing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace vc::render::prefetch {

// readMultiSlice/sampleTileSlices use Trilinear; readCompositeFast uses Nearest.
inline vc::Sampling samplingForRender(bool composite)
{
    return composite ? vc::Sampling::Nearest : vc::Sampling::Trilinear;
}

inline void insertExactChunksForSamples(
    const cv::Mat_<cv::Vec3f>& base,
    const cv::Mat_<cv::Vec3f>& dirs,
    const std::vector<float>& offsets,
    IChunkedArray* ds,
    int level,
    vc::Sampling method,
    std::unordered_set<ChunkKey, ChunkKeyHash>& uniq)
{
    if (!ds || base.empty() || offsets.empty()) return;
    const auto chunkShape = ds->chunkShape(level);
    const auto shape = ds->shape(level);
    for (int axis = 0; axis < 3; ++axis)
        if (chunkShape[axis] <= 0 || shape[axis] <= 0) return;

    // Coordinates are x,y,z; array metadata is z,y,x.
    const double extent[3] = {double(shape[2]), double(shape[1]), double(shape[0])};
    const int dim[3] = {chunkShape[2], chunkShape[1], chunkShape[0]};
    const int maxC[3] = {(shape[2] - 1) / dim[0],
                         (shape[1] - 1) / dim[1],
                         (shape[0] - 1) / dim[2]};
    auto chunkOfVoxel = [](double voxel, int width, int maxChunk) {
        // Clamp before narrowing, including for extreme finite coordinates.
        return int(std::clamp(std::floor(voxel / width), 0.0, double(maxChunk)));
    };

    auto visitSpans = [&](auto&& visit) {
        for (int row = 0; row < base.rows; ++row) {
            for (int col = 0; col < base.cols; ++col) {
                const auto& b = base(row, col);
                const auto& d = dirs(row, col);
                for (float off : offsets) {
                    // Match Slicing.cpp's float arithmetic before computing
                    // voxel/chunk indices. Double arithmetic can fall on the
                    // other side of a rounded voxel or chunk boundary.
                    const float p[3] = {b[0] + d[0] * off,
                                         b[1] + d[1] * off,
                                         b[2] + d[2] * off};
                    if (!(p[0] >= 0 && p[0] < float(shape[2]) &&
                          p[1] >= 0 && p[1] < float(shape[1]) &&
                          p[2] >= 0 && p[2] < float(shape[0]))) continue;
                    int first[3], last[3];
                    for (int axis = 0; axis < 3; ++axis) {
                        double lo, hi;
                        switch (method) {
                            case vc::Sampling::Trilinear:
                                lo = std::floor(p[axis]); hi = lo + 1; break;
                            case vc::Sampling::Tricubic:
                                lo = std::floor(p[axis]) - 1; hi = lo + 3; break;
                            default:
                                // sampleNearest adds 0.5f before conversion.
                                lo = hi = std::min(std::floor(double(p[axis] + 0.5f)),
                                                   extent[axis] - 1);
                                break;
                        }
                        first[axis] = chunkOfVoxel(lo, dim[axis], maxC[axis]);
                        last[axis] = chunkOfVoxel(hi, dim[axis], maxC[axis]);
                    }
                    visit(first, last);
                }
            }
        }
    };

    int lo[3] = {maxC[0], maxC[1], maxC[2]}, hi[3] = {-1, -1, -1};
    visitSpans([&](const int first[3], const int last[3]) {
        for (int axis = 0; axis < 3; ++axis) {
            lo[axis] = std::min(lo[axis], first[axis]);
            hi[axis] = std::max(hi[axis], last[axis]);
        }
    });
    if (hi[0] < 0) return;

    const std::size_t w = std::size_t(hi[0]) - std::size_t(lo[0]) + 1;
    const std::size_t h = std::size_t(hi[1]) - std::size_t(lo[1]) + 1;
    const std::size_t depth = std::size_t(hi[2]) - std::size_t(lo[2]) + 1;
    // A few widely separated samples must not allocate or scan a bitmap for
    // the entire intervening volume. Divide before multiplying to avoid overflow.
    constexpr std::size_t maxBitmapBytes = 8 * 1024 * 1024;
    const bool useBitmap = w <= maxBitmapBytes / h && w * h <= maxBitmapBytes / depth;
    std::vector<uint8_t> bits(useBitmap ? w * h * depth : 0, 0);
    auto index = [&](int cx, int cy, int cz) {
        return ((std::size_t(cz - lo[2]) * h) + std::size_t(cy - lo[1])) * w +
               std::size_t(cx - lo[0]);
    };
    visitSpans([&](const int first[3], const int last[3]) {
        for (int cz = first[2]; cz <= last[2]; ++cz)
            for (int cy = first[1]; cy <= last[1]; ++cy)
                for (int cx = first[0]; cx <= last[0]; ++cx) {
                    if (useBitmap) bits[index(cx, cy, cz)] = 1;
                    else uniq.insert(ChunkKey{level, cz, cy, cx});
                }
    });
    if (useBitmap) {
        for (int cz = lo[2]; cz <= hi[2]; ++cz)
            for (int cy = lo[1]; cy <= hi[1]; ++cy)
                for (int cx = lo[0]; cx <= hi[0]; ++cx)
                    if (bits[index(cx, cy, cz)]) uniq.insert(ChunkKey{level, cz, cy, cx});
    }
}

} // namespace vc::render::prefetch
