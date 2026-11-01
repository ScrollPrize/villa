#pragma once
#include "fibers3d/Fiber3DSlices.hpp"

class Volume;
namespace vc3d::scene3d
{
// A bounded, resident preview in native CT coordinates. Camera changes reuse
// this texture; no mesh construction or whole-volume allocation is required.
struct VolumeData {
    Bounds bounds;
    fibers3d::slices::Images images;
    int requestedLevel = 0;
    bool resampled = false;
};
// Prefer the requested available Zarr level, then coarser levels. If even the
// coarsest is too large, resample it into <=256 voxels per axis. CPU + GPU
// texture storage is capped at 64 MiB; sampling uses small cancellable batches.
std::shared_ptr<VolumeData> sampleVolume(Volume&, int requestedLevel, size_t maxBytes,
                                       const std::atomic_bool& cancelled);
}
