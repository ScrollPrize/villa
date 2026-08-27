#include "vc/core/util/TexturedMesh.hpp"

#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace vc::core::util
{
std::string texturedMeshObj(const TexturedMesh& mesh, std::string_view comment, std::string_view materialLibrary, std::string_view materialName, std::string_view objectName)
{
    vc::core::io::requireObjToken(materialLibrary, "textured mesh material library");
    vc::core::io::requireObjToken(materialName, "textured mesh material name");
    if (!objectName.empty())
        vc::core::io::requireObjToken(objectName, "textured mesh object name");
    if (mesh.vertices.empty() || mesh.textureCoordinates.empty())
        throw std::invalid_argument("textured mesh must contain vertices and texture coordinates");

    std::ostringstream output;
    output.imbue(std::locale::classic());
    if (!comment.empty())
        output << "# " << comment << '\n';
    output << vc::core::io::objMaterialLibraryReference(materialLibrary);
    if (!objectName.empty())
        output << "o " << objectName << '\n';
    output << vc::core::io::objUseMaterial(materialName);
    for (const auto& vertex : mesh.vertices) {
        if (!std::isfinite(vertex[0]) || !std::isfinite(vertex[1]) || !std::isfinite(vertex[2]))
            throw std::invalid_argument("textured mesh vertex must be finite");
        output << "v " << vertex[0] << ' ' << vertex[1] << ' ' << vertex[2] << '\n';
    }
    for (const auto& coordinate : mesh.textureCoordinates) {
        if (!std::isfinite(coordinate[0]) || !std::isfinite(coordinate[1]))
            throw std::invalid_argument("textured mesh coordinate must be finite");
        output << "vt " << coordinate[0] << ' ' << coordinate[1] << '\n';
    }
    for (const auto& quad : mesh.quads) {
        output << "f";
        for (size_t corner = 0; corner < 4; ++corner) {
            if (quad.vertexIndices[corner] >= mesh.vertices.size() || quad.textureCoordinateIndices[corner] >= mesh.textureCoordinates.size()) {
                throw std::invalid_argument("textured mesh quad index is out of range");
            }
            output << ' ' << quad.vertexIndices[corner] + 1 << '/' << quad.textureCoordinateIndices[corner] + 1;
        }
        output << '\n';
    }
    return output.str();
}

std::string textureMaterialMtl(std::string_view materialName, std::string_view textureFile)
{
    return vc::core::io::objMaterialMtl(
        materialName, {}, textureFile);
}

}  // namespace vc::core::util
