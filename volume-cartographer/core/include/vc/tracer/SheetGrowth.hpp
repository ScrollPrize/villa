#pragma once

#include <opencv2/core.hpp>
#include "utils/Json.hpp"

class Volume;
class QuadSurface;

// Seed-only projection/streamline branch of GrowPatch; uses raw level-0 CT.
QuadSurface* growSheetPatch(Volume& volume, const cv::Vec3f& seed,
                            const utils::Json& params, float voxelSize,
                            const utils::Json& metadata);
