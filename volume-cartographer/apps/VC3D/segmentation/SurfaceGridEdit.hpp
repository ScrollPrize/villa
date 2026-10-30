#pragma once

#include <opencv2/core/types.hpp>

#include <vector>

class QuadSurface;

namespace segmentation
{

// Grows (positive) or shrinks (negative) a surface's vertex grid on each side.
// New vertices are invalid (-1, -1, -1); every channel, including ones not yet
// loaded from disk, is padded or cropped with it so it stays aligned with the
// points. Channels stored at an integer multiple of the grid resolution keep
// that resolution. Returns false and leaves the surface unchanged if a crop
// would remove the whole grid.
bool resizeSurfaceGrid(QuadSurface& surface, int top, int bottom, int left, int right);

// Sets the "mask" channel to valid at the given grid cells (x = column,
// y = row). The mask hides vertices when a segment loads, so cells that were
// invalid when it was written must be marked before new geometry there is
// saved, or the geometry is dropped on the next load. Returns true if a mask
// exists and was changed.
bool markMaskCellsValid(QuadSurface& surface, const std::vector<cv::Point>& cells);

} // namespace segmentation
