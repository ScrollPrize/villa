#include "vc/core/util/TexturedMesh.hpp"

#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace vc::core::util
{
namespace
{

void requireToken(std::string_view value, const char* label)
{
    if (value.empty() || value.find_first_of(" \t\r\n") != std::string_view::npos)
        throw std::invalid_argument(std::string(label) + " must be a non-empty OBJ token");
}

}  // namespace

std::string texturedMeshObj(const TexturedMesh& mesh, std::string_view comment, std::string_view materialLibrary, std::string_view materialName, std::string_view objectName)
{
    requireToken(materialLibrary, "textured mesh material library");
    requireToken(materialName, "textured mesh material name");
    if (!objectName.empty())
        requireToken(objectName, "textured mesh object name");
    if (mesh.vertices.empty() || mesh.textureCoordinates.empty())
        throw std::invalid_argument("textured mesh must contain vertices and texture coordinates");

    std::ostringstream output;
    output.imbue(std::locale::classic());
    if (!comment.empty())
        output << "# " << comment << '\n';
    output << "mtllib " << materialLibrary << '\n';
    if (!objectName.empty())
        output << "o " << objectName << '\n';
    output << "usemtl " << materialName << '\n';
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
    requireToken(materialName, "texture material name");
    requireToken(textureFile, "texture material file");
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "newmtl " << materialName << '\n'
           << "Ka 1 1 1\n"
           << "Kd 1 1 1\n"
           << "Ks 0 0 0\n"
           << "d 1\n"
           << "illum 1\n"
           << "map_Kd " << textureFile << '\n';
    return output.str();
}

}  // namespace vc::core::util
