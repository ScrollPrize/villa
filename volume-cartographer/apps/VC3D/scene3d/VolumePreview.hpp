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
// coarsest is too large, resample it into <=maxEdge voxels per axis. CPU + GPU
// texture storage and sampling scratch fit maxBytes. Callers clamp maxEdge to
// their GPU's 3D texture limit; 256 keeps the default preview small and portable.
std::shared_ptr<VolumeData> sampleVolume(Volume&, int requestedLevel, size_t maxBytes,
                                       const std::atomic_bool& cancelled, int maxEdge = 256);
}
