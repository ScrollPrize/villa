#pragma once

#include <filesystem>
#include <string>

#include <opencv2/core/types.hpp>

class Volume;

namespace vc::fiber_tracer
{

void writeFiberletCropBoxVisualization(
    ::Volume& volume,
    const std::string& sourceLocator,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ,
    const std::filesystem::path& outputStem,
    int maximumTextureDimension = 4096);

}  // namespace vc::fiber_tracer
