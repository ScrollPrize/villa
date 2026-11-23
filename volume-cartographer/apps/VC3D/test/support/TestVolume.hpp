#pragma once
// Fixture CT volumes for VC3D tests, written with vc_reader.
#include "vc/core/types/Volume.hpp"

#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>

#include <array>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace vc3d::test {

struct VolumeSpec {
    std::array<int, 3> shapeZYX{32, 32, 32};
    std::array<int, 3> chunkShapeZYX{32, 32, 32};
    int numLevels = 1;
    std::string dtype = "|u1";
};

// C-order values to write.
template <class T>
struct Grid {
    std::array<int, 3> shape;
    std::vector<T> values;
    explicit Grid(std::array<int, 3> shapeZYX, T value = T{})
        : shape(shapeZYX), values(std::size_t(shapeZYX[0]) * shapeZYX[1] * shapeZYX[2], value) {}
    T& operator()(int z, int y, int x) { return values[(std::size_t(z) * shape[1] + y) * shape[2] + x]; }
    void fill(T value) { std::fill(values.begin(), values.end(), value); }
};

// An empty volume: level i halves level i-1 on every axis, rounding up.
inline std::shared_ptr<Volume> createVolume(const std::filesystem::path& root, const VolumeSpec& spec)
{
    for (int level = 0; level < spec.numLevels; ++level) {
        vcr::Index shape, chunks;
        for (int axis = 0; axis < 3; ++axis) {
            shape.push_back(std::max(1, (spec.shapeZYX[axis] + (1 << level) - 1) >> level));
            chunks.push_back(spec.chunkShapeZYX[axis]);
        }
        vcr::zarr::CreateOptions options;
        options.shape = shape;
        options.chunks = chunks;
        options.dtype = spec.dtype;
        options.compressor = "";
        options.overwrite = true;
        vcr::zarr::Writer::create(root / std::to_string(level), options);
    }
    std::ofstream(root / "meta.json") << nlohmann::json{
        {"type", "vol"}, {"uuid", root.filename().string()}, {"name", root.filename().string()}, {"format", "zarr"},
        {"width", spec.shapeZYX[2]}, {"height", spec.shapeZYX[1]}, {"slices", spec.shapeZYX[0]}, {"voxelsize", 1.0}}.dump();
    return Volume::New(root);
}

template <class T>
void writeZYX(const Volume& volume, const Grid<T>& data, const std::array<int, 3>& offset, int level = 0)
{
    auto writer = vcr::zarr::Writer::open(std::filesystem::path(volume.sourceLocation()) / std::to_string(level));
    writer.writeRegion({offset[0], offset[1], offset[2]}, {data.shape[0], data.shape[1], data.shape[2]},
                       reinterpret_cast<const std::uint8_t*>(data.values.data()));
}

}  // namespace vc3d::test
