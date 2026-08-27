#include "vc/core/io/PolylineObj.hpp"

#include <cctype>
#include <cmath>
#include <fstream>
#include <locale>
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

void writePolylinesObjImpl(
    const std::vector<NamedPolyline>& lines,
    const std::filesystem::path& outputPath,
    std::string_view comment,
    const ObjVertexColor* color)
{
    if (color != nullptr) {
        const auto valid = [](double component) {
            return std::isfinite(component) && component >= 0.0 && component <= 1.0;
        };
        if (!valid(color->red) || !valid(color->green) || !valid(color->blue))
            throw std::invalid_argument("OBJ vertex color components must be finite and between zero and one");
    }
    std::ofstream output(outputPath);
    if (!output)
        throw std::runtime_error("could not open OBJ output: " + outputPath.string());
    output.imbue(std::locale::classic());
    output << "# " << comment << '\n';
    std::size_t nextVertex = 1;
    for (const auto& line : lines) {
        output << "o " << objElementName(line.name) << '\n';
        const std::size_t firstVertex = nextVertex;
        for (const auto& point : line.points) {
            output << "v " << point[0] << ' ' << point[1] << ' ' << point[2];
            if (color != nullptr)
                output << ' ' << color->red << ' ' << color->green << ' ' << color->blue;
            output << '\n';
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
}

}  // namespace

void writePolylinesObj(
    const std::vector<NamedPolyline>& lines,
    const std::filesystem::path& outputPath,
    std::string_view comment)
{
    writePolylinesObjImpl(lines, outputPath, comment, nullptr);
}

void writePolylinesObj(
    const std::vector<NamedPolyline>& lines,
    const std::filesystem::path& outputPath,
    std::string_view comment,
    ObjVertexColor color)
{
    writePolylinesObjImpl(lines, outputPath, comment, &color);
}

}  // namespace vc::core::io
