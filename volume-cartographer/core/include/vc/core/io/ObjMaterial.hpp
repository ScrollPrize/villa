#pragma once

#include <string>
#include <string_view>

#include <opencv2/core/types.hpp>

namespace vc::core::io
{

struct ObjMaterial {
    cv::Vec3d ambient{1.0, 1.0, 1.0};
    cv::Vec3d diffuse{1.0, 1.0, 1.0};
    cv::Vec3d specular{0.0, 0.0, 0.0};
    double opacity = 1.0;
    int illuminationModel = 1;
};

void requireObjToken(std::string_view value, const char* label);

[[nodiscard]] std::string objMaterialReference(
    std::string_view materialLibrary,
    std::string_view materialName);

[[nodiscard]] std::string objMaterialLibraryReference(
    std::string_view materialLibrary);

[[nodiscard]] std::string objUseMaterial(std::string_view materialName);

[[nodiscard]] std::string objMaterialMtl(
    std::string_view materialName,
    const ObjMaterial& material,
    std::string_view diffuseTexture = {});

}  // namespace vc::core::io
