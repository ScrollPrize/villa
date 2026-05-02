#pragma once

class ViewerManager;
class CState;

namespace vc3d::ramstats {

// One-line dump to stderr summarizing process + per-subsystem memory.
// Snapshots: /proc/self/status, mallinfo2, SurfacePatchIndex
// (patchCount/surfaceCount), surface LRU resident count.
void dumpOnce(ViewerManager* viewerManager, CState* state);

}  // namespace vc3d::ramstats
