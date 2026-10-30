#pragma once

// An empty local OME-Zarr volume for tests (uncompressed, zarr v2, opened as a
// Volume). Tests write the voxels they need with vcr::zarr::Writer.

#include "vc/core/types/Volume.hpp"
#include "utils/Json.hpp"

#include <vc_reader/zarr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace vc::test {

struct ZarrVolumeSpec {
    std::array<size_t, 3> shapeZYX{};
    std::array<size_t, 3> chunkShapeZYX{64, 64, 64};
    std::string dtype = "|u1";  // numpy typestr: |u1 or <u2
    // Each level halves every axis (rounding up) from the one before.
    size_t numLevels = 1;
    double fillValue = 0.0;
    double voxelSize = 1.0;
    std::string uuid;
    std::string name;
};

namespace detail {

inline void writeText(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path, std::ios::binary);
    out << text;
    if (!out) throw std::runtime_error("failed to write " + path.string());
}

inline std::array<size_t, 3> levelShape(const ZarrVolumeSpec& spec, size_t level)
{
    auto shape = spec.shapeZYX;
    for (size_t i = 0; i < level; ++i)
        for (auto& d : shape) d = std::max<size_t>(1, (d + 1) / 2);
    return shape;
}

} // namespace detail

// Writes the volume (.zgroup, one array per level, .zattrs multiscales,
// meta.json) at `path`, replacing anything there, and opens it.
inline std::shared_ptr<Volume> createZarrVolume(const std::filesystem::path& path, const ZarrVolumeSpec& spec)
{
    if (spec.numLevels == 0 || spec.shapeZYX[0] == 0 || spec.shapeZYX[1] == 0 || spec.shapeZYX[2] == 0)
        throw std::invalid_argument("test zarr volume needs a shape and a level");
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    detail::writeText(path / ".zgroup", R"({"zarr_format": 2})");
    utils::Json datasets = utils::Json::array();
    for (size_t level = 0; level < spec.numLevels; ++level) {
        const auto shape = detail::levelShape(spec, level);
        vcr::zarr::CreateOptions options;
        options.shape = {std::int64_t(shape[0]), std::int64_t(shape[1]), std::int64_t(shape[2])};
        for (size_t d = 0; d < 3; ++d) options.chunks.push_back(std::int64_t(std::min(spec.chunkShapeZYX[d], shape[d])));
        options.dtype = spec.dtype;
        options.compressor = "";
        options.fillValue = spec.fillValue;
        vcr::zarr::Writer::create(path / std::to_string(level), options);
        const double scale = spec.voxelSize * double(size_t(1) << level);
        utils::Json scales = utils::Json::array();
        for (int axis = 0; axis < 3; ++axis) scales.push_back(scale);
        utils::Json transforms = utils::Json::array();
        transforms.push_back(utils::Json{{"type", "scale"}, {"scale", scales}});
        datasets.push_back(utils::Json{{"path", std::to_string(level)}, {"coordinateTransformations", transforms}});
    }
    const std::string base = path.filename().string();
    const std::string name = spec.name.empty() ? base : spec.name;
    utils::Json axes = utils::Json::array();
    for (const char* axis : {"z", "y", "x"}) axes.push_back(utils::Json{{"name", axis}, {"type", "space"}});
    utils::Json multiscale{{"version", "0.4"}, {"name", name}, {"axes", axes}, {"datasets", datasets}};
    utils::Json multiscales = utils::Json::array();
    multiscales.push_back(multiscale);
    detail::writeText(path / ".zattrs", utils::Json{{"multiscales", multiscales}}.dump(2));
    detail::writeText(path / "meta.json",
                      utils::Json{{"type", "vol"}, {"uuid", spec.uuid.empty() ? base : spec.uuid}, {"name", name},
                                  {"format", "zarr"}, {"width", int(spec.shapeZYX[2])}, {"height", int(spec.shapeZYX[1])},
                                  {"slices", int(spec.shapeZYX[0])}, {"voxelsize", spec.voxelSize}, {"min", 0.0},
                                  {"max", spec.dtype == "|u1" ? 255.0 : 65535.0}}
                          .dump(2));
    return Volume::New(path);
}

} // namespace vc::test
