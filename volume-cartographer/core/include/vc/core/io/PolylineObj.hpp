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

struct ObjVertexColor {
    double red;
    double green;
    double blue;
};

[[nodiscard]] std::string objElementName(std::string name);

void writePolylinesObj(
    const std::vector<NamedPolyline>& lines, const std::filesystem::path& outputPath, std::string_view comment = "VC3D polylines");

void writePolylinesObj(
    const std::vector<NamedPolyline>& lines,
    const std::filesystem::path& outputPath,
    std::string_view comment,
    ObjVertexColor color);

}  // namespace vc::core::io
