#pragma once

#include <filesystem>
#include <string>

#include <opencv2/core/types.hpp>

namespace vcr::zarr {
struct OpenedArray;
}

namespace vc::fiber_tracer
{

void writeFiberletCropBoxVisualization(
    const vcr::zarr::OpenedArray& ct,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ,
    const std::filesystem::path& outputStem,
    int maximumTextureDimension = 4096);

}  // namespace vc::fiber_tracer
