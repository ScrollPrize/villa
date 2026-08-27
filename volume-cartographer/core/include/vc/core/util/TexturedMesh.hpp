#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core/types.hpp>

#include "vc/core/io/ObjMaterial.hpp"

namespace vc::core::util
{

struct TexturedQuad {
    std::array<size_t, 4> vertexIndices{};
    std::array<size_t, 4> textureCoordinateIndices{};
};

struct TexturedMesh {
    std::vector<cv::Vec3d> vertices;
    std::vector<cv::Vec2d> textureCoordinates;
    std::vector<TexturedQuad> quads;
};

[[nodiscard]] std::string texturedMeshObj(
    const TexturedMesh& mesh, std::string_view comment, std::string_view materialLibrary, std::string_view materialName, std::string_view objectName = {});

[[nodiscard]] std::string textureMaterialMtl(std::string_view materialName, std::string_view textureFile);

}  // namespace vc::core::util
