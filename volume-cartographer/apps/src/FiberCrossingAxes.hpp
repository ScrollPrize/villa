#pragma once
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>

// Geometry of every perpendicular crossing of a fiber, extracted once from a
// crossing search store (the extractor's own graph, normal-free) so normal
// solves need neither the partner fibers nor a new search.
struct FiberCrossingAxis {
    double arc = 0.;                // own arc, base voxels
    cv::Vec3d axis{0., 0., 0.};     // unit own tangent x partner tangent (sign-free)
    double gap = 0.;                // |offset|
    cv::Vec3d offset{0., 0., 0.};   // partner point - own point
    cv::Vec3d ownTangent{0., 0., 0.};
};

// One entry list per fiber of `ids`; `lines` are the fibers' polylines (all
// fibers referenced by the store must be among them). Writes a binary file.
void writeFiberCrossingAxes(const std::filesystem::path& output, const std::string& crossingStore,
    const std::vector<std::string>& ids, const std::vector<std::vector<cv::Vec3d>>& lines, size_t workers);

// Entries per fiber of `ids` (empty for fibers the file does not list).
std::vector<std::vector<FiberCrossingAxis>> readFiberCrossingAxes(const std::filesystem::path& path,
    const std::vector<std::string>& ids, size_t& unknownFibers);
