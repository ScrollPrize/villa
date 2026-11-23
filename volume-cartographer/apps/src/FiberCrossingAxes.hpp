#pragma once
#include "vc/fiber_tracer/FiberCrossingAxes.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>

// Geometry of every perpendicular crossing of a fiber, extracted once from a
// crossing search store (the extractor's own graph, normal-free) so normal
// solves need neither the partner fibers nor a new search.
using FiberCrossingAxis = vc::fiber_tracer::FiberCrossingAxis;

// One entry list per fiber of `ids`; `lines` are the fibers' polylines (all
// fibers referenced by the store must be among them). Writes a binary file.
void writeFiberCrossingAxes(const std::filesystem::path& output, const std::string& crossingStore,
    const std::vector<std::string>& ids, const std::vector<std::vector<cv::Vec3d>>& lines, size_t workers);

// Entries per fiber of `ids` (empty for fibers the file does not list).
std::vector<std::vector<FiberCrossingAxis>> readFiberCrossingAxes(const std::filesystem::path& path,
    const std::vector<std::string>& ids, size_t& unknownFibers);
