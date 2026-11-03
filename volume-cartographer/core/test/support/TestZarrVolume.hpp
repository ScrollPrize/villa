#pragma once

// Local OME-Zarr volumes for tests: VC only reads volumes, so fixtures are
// written straight through utils::ZarrArray (uncompressed, zarr v2).

#include "vc/core/render/ChunkFetch.hpp"
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"
#include "utils/Json.hpp"
#include "utils/zarr.hpp"

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
    vc::render::ChunkDtype dtype = vc::render::ChunkDtype::UInt8;
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

inline utils::ZarrArray openLevel(const Volume& volume, int level)
{
    return utils::ZarrArray::open(volume.path() / std::to_string(level));
}

template <typename T>
Array3D<T> readLevel(const utils::ZarrArray& array)
{
    const auto& meta = array.metadata();
    const std::array<size_t, 3> shape{meta.shape[0], meta.shape[1], meta.shape[2]};
    const std::array<size_t, 3> chunks{meta.chunks[0], meta.chunks[1], meta.chunks[2]};
    Array3D<T> out(shape, static_cast<T>(meta.fill_value.value_or(0.0)));
    for (size_t cz = 0; cz * chunks[0] < shape[0]; ++cz)
        for (size_t cy = 0; cy * chunks[1] < shape[1]; ++cy)
            for (size_t cx = 0; cx * chunks[2] < shape[2]; ++cx) {
                const std::array<size_t, 3> index{cz, cy, cx};
                const auto bytes = array.read_chunk(index);
                if (!bytes) continue;
                const auto* values = reinterpret_cast<const T*>(bytes->data());
                for (size_t z = 0; z < chunks[0] && cz * chunks[0] + z < shape[0]; ++z)
                    for (size_t y = 0; y < chunks[1] && cy * chunks[1] + y < shape[1]; ++y)
                        for (size_t x = 0; x < chunks[2] && cx * chunks[2] + x < shape[2]; ++x)
                            out(cz * chunks[0] + z, cy * chunks[1] + y, cx * chunks[2] + x) =
                                values[(z * chunks[1] + y) * chunks[2] + x];
            }
    return out;
}

template <typename T>
void writeRegion(utils::ZarrArray& array, const Array3D<T>& data, const std::array<size_t, 3>& offset)
{
    const auto& meta = array.metadata();
    const std::array<size_t, 3> shape{meta.shape[0], meta.shape[1], meta.shape[2]};
    const std::array<size_t, 3> chunks{meta.chunks[0], meta.chunks[1], meta.chunks[2]};
    const auto region = data.shape();
    for (size_t d = 0; d < 3; ++d)
        if (offset[d] + region[d] > shape[d]) throw std::out_of_range("test zarr write exceeds the array");
    if (region[0] == 0 || region[1] == 0 || region[2] == 0) return;
    const T fill = static_cast<T>(meta.fill_value.value_or(0.0));
    std::vector<T> chunk(chunks[0] * chunks[1] * chunks[2]);
    for (size_t cz = offset[0] / chunks[0]; cz * chunks[0] < offset[0] + region[0]; ++cz)
        for (size_t cy = offset[1] / chunks[1]; cy * chunks[1] < offset[1] + region[1]; ++cy)
            for (size_t cx = offset[2] / chunks[2]; cx * chunks[2] < offset[2] + region[2]; ++cx) {
                const std::array<size_t, 3> index{cz, cy, cx};
                if (auto existing = array.read_chunk(index)) std::memcpy(chunk.data(), existing->data(), chunk.size() * sizeof(T));
                else std::fill(chunk.begin(), chunk.end(), fill);
                for (size_t z = 0; z < chunks[0]; ++z)
                    for (size_t y = 0; y < chunks[1]; ++y)
                        for (size_t x = 0; x < chunks[2]; ++x) {
                            const size_t gz = cz * chunks[0] + z, gy = cy * chunks[1] + y, gx = cx * chunks[2] + x;
                            if (gz < offset[0] || gy < offset[1] || gx < offset[2]) continue;
                            if (gz >= offset[0] + region[0] || gy >= offset[1] + region[1] || gx >= offset[2] + region[2]) continue;
                            chunk[(z * chunks[1] + y) * chunks[2] + x] = data(gz - offset[0], gy - offset[1], gx - offset[2]);
                        }
                array.write_chunk(index, std::as_bytes(std::span<const T>(chunk)));
            }
}

// Rounded mean over the source voxels each coarse voxel covers.
template <typename T>
Array3D<T> downsampleMean(const Array3D<T>& src, const std::array<size_t, 3>& dstShape)
{
    const auto srcShape = src.shape();
    Array3D<T> out(dstShape);
    std::array<double, 3> scale{};
    for (size_t d = 0; d < 3; ++d) scale[d] = std::max(1.0, double(srcShape[d]) / double(std::max<size_t>(1, dstShape[d])));
    for (size_t z = 0; z < dstShape[0]; ++z)
        for (size_t y = 0; y < dstShape[1]; ++y)
            for (size_t x = 0; x < dstShape[2]; ++x) {
                const std::array<size_t, 3> at{z, y, x};
                std::array<size_t, 3> begin{}, end{};
                for (size_t d = 0; d < 3; ++d) {
                    begin[d] = std::min(srcShape[d], size_t(std::floor(double(at[d]) * scale[d])));
                    end[d] = std::min(srcShape[d], size_t(std::ceil(double(at[d] + 1) * scale[d])));
                    if (end[d] <= begin[d] && begin[d] < srcShape[d]) end[d] = begin[d] + 1;
                }
                std::uint64_t sum = 0, count = 0;
                for (size_t sz = begin[0]; sz < end[0]; ++sz)
                    for (size_t sy = begin[1]; sy < end[1]; ++sy)
                        for (size_t sx = begin[2]; sx < end[2]; ++sx) { sum += src(sz, sy, sx); ++count; }
                out(z, y, x) = count ? static_cast<T>((sum + count / 2) / count) : T{};
            }
    return out;
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
        utils::ZarrMetadata meta;
        meta.version = utils::ZarrVersion::v2;
        meta.shape = {shape[0], shape[1], shape[2]};
        meta.chunks = {std::min(spec.chunkShapeZYX[0], shape[0]), std::min(spec.chunkShapeZYX[1], shape[1]),
                       std::min(spec.chunkShapeZYX[2], shape[2])};
        meta.dtype = spec.dtype == vc::render::ChunkDtype::UInt16 ? utils::ZarrDtype::uint16 : utils::ZarrDtype::uint8;
        meta.fill_value = spec.fillValue;
        meta.dimension_separator = ".";
        utils::ZarrArray::create(path / std::to_string(level), std::move(meta));
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
                                  {"max", spec.dtype == vc::render::ChunkDtype::UInt8 ? 255.0 : 65535.0}}
                          .dump(2));
    return Volume::New(path);
}

// Writes `data` ([z, y, x]) at `offsetZYX` into `level`, rebuilds every
// coarser level from it, and drops the volume's cached chunks.
template <typename T>
void writeZYX(Volume& volume, const Array3D<T>& data, const std::array<int, 3>& offsetZYX, int level = 0)
{
    auto array = detail::openLevel(volume, level);
    detail::writeRegion(array, data, {size_t(offsetZYX[0]), size_t(offsetZYX[1]), size_t(offsetZYX[2])});
    for (int dst = level + 1; dst < int(volume.numScales()) && volume.hasScaleLevel(dst); ++dst) {
        const auto src = detail::readLevel<T>(detail::openLevel(volume, dst - 1));
        auto dstArray = detail::openLevel(volume, dst);
        const auto& meta = dstArray.metadata();
        detail::writeRegion(dstArray, detail::downsampleMean(src, {meta.shape[0], meta.shape[1], meta.shape[2]}), {0, 0, 0});
    }
    volume.invalidateCache();
}

// Stores one raw chunk ([z, y, x] chunk indices) of `level`.
inline void writeChunk(Volume& volume, int level, const std::array<size_t, 3>& chunkZYX, std::span<const std::byte> data)
{
    detail::openLevel(volume, level).write_chunk(chunkZYX, data);
    volume.invalidateCache();
}

} // namespace vc::test
