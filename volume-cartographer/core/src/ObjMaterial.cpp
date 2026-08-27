#include "vc/core/io/ObjMaterial.hpp"

#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace vc::core::io
{
namespace
{

void requireColor(const cv::Vec3d& color, const char* label)
{
    for (int index = 0; index < 3; ++index) {
        const double component = color[index];
        if (!std::isfinite(component) || component < 0.0 || component > 1.0) {
            throw std::invalid_argument(
                std::string(label) +
                " components must be finite and between zero and one");
        }
    }
}

}  // namespace

void requireObjToken(std::string_view value, const char* label)
{
    if (value.empty() ||
        value.find_first_of(" \t\r\n") != std::string_view::npos) {
        throw std::invalid_argument(
            std::string(label) + " must be a non-empty OBJ token");
    }
}

std::string objMaterialReference(
    std::string_view materialLibrary,
    std::string_view materialName)
{
    return objMaterialLibraryReference(materialLibrary) +
        objUseMaterial(materialName);
}

std::string objMaterialLibraryReference(std::string_view materialLibrary)
{
    requireObjToken(materialLibrary, "OBJ material library");
    return "mtllib " + std::string(materialLibrary) + "\n";
}

std::string objUseMaterial(std::string_view materialName)
{
    requireObjToken(materialName, "OBJ material name");
    return "usemtl " + std::string(materialName) + "\n";
}

std::string objMaterialMtl(
    std::string_view materialName,
    const ObjMaterial& material,
    std::string_view diffuseTexture)
{
    requireObjToken(materialName, "OBJ material name");
    if (!diffuseTexture.empty())
        requireObjToken(diffuseTexture, "OBJ material texture");
    requireColor(material.ambient, "OBJ ambient color");
    requireColor(material.diffuse, "OBJ diffuse color");
    requireColor(material.specular, "OBJ specular color");
    if (!std::isfinite(material.opacity) ||
        material.opacity < 0.0 || material.opacity > 1.0) {
        throw std::invalid_argument(
            "OBJ material opacity must be finite and between zero and one");
    }
    if (material.illuminationModel < 0 || material.illuminationModel > 10)
        throw std::invalid_argument("OBJ illumination model must be between zero and ten");

    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "newmtl " << materialName << '\n'
           << "Ka " << material.ambient[0] << ' ' << material.ambient[1]
           << ' ' << material.ambient[2] << '\n'
           << "Kd " << material.diffuse[0] << ' ' << material.diffuse[1]
           << ' ' << material.diffuse[2] << '\n'
           << "Ks " << material.specular[0] << ' ' << material.specular[1]
           << ' ' << material.specular[2] << '\n'
           << "d " << material.opacity << '\n'
           << "illum " << material.illuminationModel << '\n';
    if (!diffuseTexture.empty())
        output << "map_Kd " << diffuseTexture << '\n';
    return output.str();
}

}  // namespace vc::core::io
