#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::core::io
{

struct NamedPolyline {
    std::string name;
    std::vector<cv::Vec3d> points;
};

[[nodiscard]] std::string objElementName(std::string name);

void writePolylinesObj(
    const std::vector<NamedPolyline>& lines, const std::filesystem::path& outputPath, std::string_view comment = "VC3D polylines");

}  // namespace vc::core::io
