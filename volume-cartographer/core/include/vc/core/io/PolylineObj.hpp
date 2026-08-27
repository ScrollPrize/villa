#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core/types.hpp>

#include "vc/core/io/ObjMaterial.hpp"

namespace vc::core::io
{

struct NamedPolyline {
    std::string name;
    std::vector<cv::Vec3d> points;
};

struct MaterialPolylineObjPaths {
    std::filesystem::path obj;
    std::filesystem::path material;
};

[[nodiscard]] std::string objElementName(std::string name);

void writePolylinesObj(
    const std::vector<NamedPolyline>& lines, const std::filesystem::path& outputPath, std::string_view comment = "VC3D polylines");

[[nodiscard]] MaterialPolylineObjPaths writePolylinesObjWithMaterial(
    const std::vector<NamedPolyline>& lines,
    const std::filesystem::path& outputPath,
    std::string_view comment,
    const ObjMaterial& material);

}  // namespace vc::core::io
