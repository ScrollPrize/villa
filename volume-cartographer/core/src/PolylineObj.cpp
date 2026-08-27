#include "vc/core/io/PolylineObj.hpp"

#include "vc/core/util/AtomicFile.hpp"

#include <cctype>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace vc::core::io
{

std::string objElementName(std::string name)
{
    if (name.empty())
        return "polyline";
    for (char& character : name) {
        const auto value = static_cast<unsigned char>(character);
        if (!std::isalnum(value) && character != '_' && character != '-')
            character = '_';
    }
    return name;
}

namespace
{

std::string polylinesObj(
    const std::vector<NamedPolyline>& lines,
    std::string_view comment,
    std::string_view materialReference)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "# " << comment << '\n';
    output << materialReference;
    std::size_t nextVertex = 1;
    for (const auto& line : lines) {
        output << "o " << objElementName(line.name) << '\n';
        const std::size_t firstVertex = nextVertex;
        for (const auto& point : line.points) {
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2]
                   << '\n';
            ++nextVertex;
        }
        if (line.points.size() >= 2) {
            for (std::size_t index = 1; index < line.points.size(); ++index) {
                output << "l " << firstVertex + index - 1 << ' ' << firstVertex + index << '\n';
            }
        } else if (line.points.size() == 1) {
            output << "p " << firstVertex << '\n';
        }
    }
    return output.str();
}

}  // namespace

void writePolylinesObj(
    const std::vector<NamedPolyline>& lines,
    const std::filesystem::path& outputPath,
    std::string_view comment)
{
    std::ofstream output(outputPath);
    if (!output)
        throw std::runtime_error("could not open OBJ output: " + outputPath.string());
    output.imbue(std::locale::classic());
    output << polylinesObj(lines, comment, {});
}

MaterialPolylineObjPaths writePolylinesObjWithMaterial(
    const std::vector<NamedPolyline>& lines,
    const std::filesystem::path& outputPath,
    std::string_view comment,
    const ObjMaterial& material)
{
    const std::string stem = outputPath.stem().string();
    requireObjToken(stem, "polyline OBJ output stem");
    const std::string materialName = stem + "_lines";
    const auto materialPath = outputPath.parent_path() / (stem + ".mtl");
    const std::string reference = objMaterialReference(
        materialPath.filename().string(), materialName);
    const std::string mtl = objMaterialMtl(materialName, material);
    const std::string obj = polylinesObj(lines, comment, reference);

    vc::core::util::atomicWriteString(materialPath, mtl);
    vc::core::util::atomicWriteString(outputPath, obj);
    return {outputPath, materialPath};
}

}  // namespace vc::core::io
