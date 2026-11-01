#pragma once
#include "common.hpp"
#include "../aligned/plan.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace vcr::projection {
// One surface sample that landed in a crop: the voxel (index in the crop), its UV, the
// distance from the voxel's centre to its surface point, and its id.
struct Sample {
    std::int64_t voxel;
    float uv[2];
    float distance;
    std::int64_t id;
};
// The samples of one crop binned by voxel for raster(): offsets per voxel, then uv,
// distance and ids in voxel order.
struct Plan {
    Raster grid;
    std::vector<float> uv, distance;
    std::vector<std::int64_t> ids, offsets;
    Plan(const Sample* samples, std::size_t n, aligned::Triple shape, std::size_t budget)
    {
        grid.count = 1;
        for (int a = 0; a < 3; ++a) {
            if (shape[a] < 1 || shape[a] > 256) throw std::invalid_argument("invalid raster block");
            grid.count *= std::size_t(shape[a]);
        }
        if (grid.count > 32768) throw std::invalid_argument("raster blocks are limited to 32768 voxels");
        const auto bytes = (grid.count + 1) * 8 + n * (3 * 4 + 8);
        if (bytes > budget) throw std::length_error("projection sample batch exceeds sampling_work_bytes");
        offsets.assign(grid.count + 1, 0);
        for (std::size_t i = 0; i < n; ++i) {
            if (samples[i].voxel < 0 || std::size_t(samples[i].voxel) >= grid.count) throw std::invalid_argument("sample outside its raster block");
            ++offsets[std::size_t(samples[i].voxel) + 1];
        }
        for (std::size_t i = 0; i < grid.count; ++i) offsets[i + 1] += offsets[i];
        uv.resize(n * 2);
        distance.resize(n);
        ids.resize(n);
        auto next = offsets;
        for (std::size_t i = 0; i < n; ++i) {
            const auto at = std::size_t(next[std::size_t(samples[i].voxel)]++);
            uv[at * 2] = samples[i].uv[0];
            uv[at * 2 + 1] = samples[i].uv[1];
            distance[at] = samples[i].distance;
            ids[at] = samples[i].id;
        }
    }
    [[nodiscard]] std::size_t bytes() const { return uv.size() * 4 + distance.size() * 4 + ids.size() * 8 + offsets.size() * 8; }
};
#ifdef VC_READER_CUDA
// raster() over every voxel on the device: the plan's arrays are uploaded, `label` and the
// outputs are device storage. Returns when the voxels are written.
void launch(const Plan&, Raster, const void* label, ElementType, void* out, bool* coverage, bool* conflict, float* best, std::int64_t* winner, float* uv, bool* geometry, int device);
#endif
}  // namespace vcr::projection
