#include "vc/fiber_tracer/FiberletCropVisualization.hpp"

#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/SurfaceTexture.hpp"
#include "vc/core/util/TexturedMesh.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/fiber_tracer/FiberReplay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vc::fiber_tracer
{
namespace
{

struct BoxFace {
    const char* name;
    cv::Vec3d origin;
    cv::Vec3d uExtent;
    cv::Vec3d vExtent;
    int uAxis = 0;
    int vAxis = 1;
};

std::vector<BoxFace> boxFaces(const cv::Vec3d& minimum, const cv::Vec3d& maximum)
{
    const cv::Vec3d span = maximum - minimum;
    return {
        {"x_min", minimum, {0, span[1], 0}, {0, 0, span[2]}, 1, 2},
        {"x_max", {maximum[0], minimum[1], minimum[2]}, {0, span[1], 0}, {0, 0, span[2]}, 1, 2},
        {"y_min", minimum, {span[0], 0, 0}, {0, 0, span[2]}, 0, 2},
        {"y_max", {minimum[0], maximum[1], minimum[2]}, {span[0], 0, 0}, {0, 0, span[2]}, 0, 2},
        {"z_min", minimum, {span[0], 0, 0}, {0, span[1], 0}, 0, 1},
        {"z_max", {minimum[0], minimum[1], maximum[2]}, {span[0], 0, 0}, {0, span[1], 0}, 0, 1},
    };
}

int textureDimension(double span, double samplesPerBase, int maximum)
{
    return std::max(2, static_cast<int>(std::min(std::ceil(std::abs(span) * samplesPerBase) + 1.0, static_cast<double>(maximum))));
}

cv::Mat_<cv::Vec3f> coordinateGrid(const BoxFace& face, int columns, int rows)
{
    cv::Mat_<cv::Vec3f> result(rows, columns);
    for (int row = 0; row < rows; ++row) {
        const double v = static_cast<double>(row) / static_cast<double>(rows - 1);
        for (int column = 0; column < columns; ++column) {
            const double u = static_cast<double>(column) / static_cast<double>(columns - 1);
            const cv::Vec3d point = face.origin + face.uExtent * u + face.vExtent * v;
            result(row, column) = cv::Vec3f(point);
        }
    }
    return result;
}

vc::core::util::TexturedMesh faceMesh(const BoxFace& face)
{
    vc::core::util::TexturedMesh result;
    result.vertices = {
        face.origin,
        face.origin + face.uExtent,
        face.origin + face.uExtent + face.vExtent,
        face.origin + face.vExtent,
    };
    result.textureCoordinates = {{0, 1}, {1, 1}, {1, 0}, {0, 0}};
    result.quads.push_back({{0, 1, 2, 3}, {0, 1, 2, 3}});
    return result;
}

}  // namespace

void writeFiberletCropBoxVisualization(
    ::Volume& volume, const std::string& sourceLocator, const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ, const std::filesystem::path& outputStem, int maximumTextureDimension)
{
    if (maximumTextureDimension < 2)
        throw std::invalid_argument("Fiberlet crop texture dimension must be at least two");
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(minimumBaseXYZ[axis]) || !std::isfinite(maximumBaseXYZ[axis]) || !(maximumBaseXYZ[axis] > minimumBaseXYZ[axis])) {
            throw std::invalid_argument("Fiberlet crop visualization bounds are invalid");
        }
    }
    const auto source = validateFiberReplayStripCtVolume(volume, sourceLocator);
    const auto directory = outputStem.parent_path();
    const auto stem = outputStem.stem().string();
    const auto mtlName = stem + ".mtl";
    std::string mtl;
    for (const auto& face : boxFaces(minimumBaseXYZ, maximumBaseXYZ)) {
        const int columns = textureDimension(cv::norm(face.uExtent), source.scaleFromBaseXYZ[face.uAxis], maximumTextureDimension);
        const int rows = textureDimension(cv::norm(face.vExtent), source.scaleFromBaseXYZ[face.vAxis], maximumTextureDimension);
        auto coordinates = coordinateGrid(face, columns, rows);
        for (auto& point : coordinates) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                point[axis] = static_cast<float>(static_cast<double>(point[axis]) * source.scaleFromBaseXYZ[axis] + source.offsetFromBaseXYZ[axis]);
            }
        }
        const std::string textureName = stem + "_" + face.name + ".tif";
        const std::string materialName = "fiberlet_crop_" + std::string(face.name);
        vc::core::util::writeUncompressedTextureTiff(
            directory / textureName,
            vc::core::util::renderCoordsTextureFineToCoarse(coordinates, volume, 0, 1, "Fiberlet crop texture sampling"));
        mtl += vc::core::util::textureMaterialMtl(materialName, textureName);
        vc::core::util::atomicWriteString(
            directory / (stem + "_" + face.name + ".obj"),
            vc::core::util::texturedMeshObj(faceMesh(face), "VC3D Fiberlet crop bbox face", mtlName, materialName, face.name));
    }
    vc::core::util::atomicWriteString(directory / mtlName, mtl);
}

}  // namespace vc::fiber_tracer
