// The CUDA sampling of vcr::grid rasters: shared by the kernel and its host side.
#pragma once
#include "vc_reader/composite.hpp"
#include "../reader_cuda/types.hpp"
#include "level_map.hpp"
#include "points.hpp"

#include <cstddef>
#include <cstdint>

namespace vcr::grid::cuda_detail {

// The bricks a blocking pass lacked, each noted once: an open-addressed set of brick
// keys (empty slots hold kEmpty) and the keys in the order they were first noted.
// Sized by the frame, not the volume: a level's brick grid can hold billions of bricks.
struct LackingSet {
    static constexpr unsigned long long kEmpty = ~0ull;
    unsigned long long* table = nullptr;  // capacity slots
    long long* keys = nullptr;            // capacity entries
    unsigned* count = nullptr;            // keys noted; above capacity when the set overflowed
    unsigned capacity = 0;                // a power of two
};

// One level on the device: its resident bricks and where its indices sit.
struct DeviceLevel {
    reader_cuda::Source source;
    LevelMap map;
};

struct Params {
    int width = 0, height = 0;
    unsigned layers = 0;
    const float* offsets = nullptr;
    // Points: a plane, or a raster over a surface's grid (grid::point on the device).
    bool plane = false, prepared = false;
    const float* xyz = nullptr;
    const float* normals = nullptr;
    float origin[3]{}, dx[3]{}, dy[3]{}, normal[3]{};
    double x0 = 0, y0 = 0, sx = 1, sy = 1, kx = 0, ky = 0;
    PointMap map{};
    GridView grid{};
    int order = 1;
    const std::uint8_t* pixels = nullptr;  // optional mask
    bool reduce = false;
    composite::Options composite{};
    const DeviceLevel* levels = nullptr;
    int levelCount = 0;
    float* values = nullptr;        // pixels, or layers * pixels (layer-major)
    std::int8_t* level = nullptr;   // per pixel
    // Bricks of the first level a sample lacked (blocking requests).
    LackingSet lacking{};
};

// Samples every pixel on the current stream; returns when the launch is enqueued.
void launch(const Params& params);

}  // namespace vcr::grid::cuda_detail
