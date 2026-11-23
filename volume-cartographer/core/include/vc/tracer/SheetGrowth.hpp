#pragma once

#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

namespace vcr { class Array; }
class QuadSurface;

// Seed-only projection/streamline branch of GrowPatch; uses raw level-0 CT.
QuadSurface* growSheetPatch(vcr::Array& volume, const cv::Vec3f& seed,
                            const nlohmann::json& params, float voxelSize,
                            const nlohmann::json& metadata);
