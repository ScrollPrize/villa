#pragma once

#include "vc/core/util/SheetWalk.hpp"

namespace vc::sheet {
enum class SeedStage { SeedNormal, CentralColumn, RowDirections, Rows };
// Stage-local counts. Calls are serialized, but may originate on workers.
using SeedProgress = std::function<void(SeedStage, std::size_t completed, std::size_t total)>;

struct SeedOptions
{
    int generations{20}; // Cells on each side of the fixed center seed.
    double spacing{20.0}; // Output mesh spacing in level-0 voxels.
    unsigned workers{1};
    SheetWalkOptions walk;
};

// Construct a coordinate patch: walk a central column in both directions,
// then advance and redistribute a front on each side of that column.
// The seed remains the center vertex. Normal-query failures propagate; callers
// decide whether a missing normal permits straight-line continuation.
cv::Mat_<cv::Vec3f> growSheetSeed(const cv::Vec3f& seed,
                                const SheetNormalFn& normal,
                                const SeedOptions& options);
cv::Mat_<cv::Vec3f> growSheetSeed(const cv::Vec3f& seed,
                                const SheetNormalFn& normal,
                                const SeedOptions& options,
                                const SeedProgress& progress);
} // namespace vc::sheet
