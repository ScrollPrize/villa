#pragma once
#include "Fiber3DScene.hpp"
#include <atomic>

class Volume;

namespace vc3d::fibers3d::ct
{
struct Tile {
    std::vector<Point> left, right;
    int width = 0, height = 0;
    // Interleaved grayscale / validity, independent of display window/level.
    std::vector<uint8_t> pixels;
};

// Row endpoints and all sample coordinates use active level-0 CT voxels.
// Bounded, cancellable trilinear sampling; no slab, MIP or extra disk cache.
void sampleTile(Tile&, Volume&, int level, const std::atomic_bool* cancelled = nullptr);
}
