#pragma once

// Pyramid level selection for a sparse overlay pyramid. Pure so the viewer
// and its tests share one definition.
namespace vc3d::overlay_level
{

// The first level at or coarser than `level` for which `present(level)` is
// true, i.e. the level actually stores data. When no level at or coarser than
// the request is present, the coarsest level is returned so the caller still
// addresses a valid index (it renders nothing rather than an invalid level).
template <class Present>
int presentLevelAtOrCoarser(int level, int numLevels, Present&& present)
{
    if (numLevels <= 0) {
        return 0;
    }
    if (level < 0) {
        level = 0;
    }
    const int last = numLevels - 1;
    if (level > last) {
        level = last;
    }
    while (level < last && !present(level)) {
        ++level;
    }
    return level;
}

}  // namespace vc3d::overlay_level
