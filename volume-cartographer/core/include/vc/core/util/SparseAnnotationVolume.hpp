#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "utils/Json.hpp"

namespace vc {

enum class AnnotationCoordinateOrder {
    XYZ,
    ZYX,
};

enum class AnnotationGeometryMode {
    Points,
    OrderedPolyline,
};

// Geometry for an optional pyramid level. scaleZYX and translationZYX map a
// voxel coordinate in this level to the level-0 coordinate system:
//
//     level0 = level * scaleZYX + translationZYX
//
// The same transform is emitted in the OME-Zarr multiscales metadata.
struct SparseAnnotationPyramidLevel {
    std::array<std::size_t, 3> shapeZYX{};
    std::array<double, 3> scaleZYX{2.0, 2.0, 2.0};
    std::array<double, 3> translationZYX{0.0, 0.0, 0.0};
};

struct SparseAnnotationVolumeSpec {
    // Level-0 shape in storage order.
    std::array<std::size_t, 3> shapeZYX{};
    std::vector<SparseAnnotationPyramidLevel> pyramidLevels;
    std::array<std::size_t, 3> chunkShapeZYX{64, 64, 64};
    std::string compressor{"zstd"};
    int compressionLevel{3};
    std::uint16_t fillValue{0};
    // Merged into root .zattrs. Reserved generic keys may not be replaced.
    utils::Json rootAttributes{utils::Json::object()};
};

struct AnnotationPointBatch {
    std::uint16_t label{0};
    std::vector<std::array<double, 3>> coordinates;
    AnnotationCoordinateOrder coordinateOrder{AnnotationCoordinateOrder::XYZ};
    AnnotationGeometryMode geometryMode{AnnotationGeometryMode::Points};
    double radius{0.0};
};

class SparseAnnotationVolumeWriter {
public:
    explicit SparseAnnotationVolumeWriter(SparseAnnotationVolumeSpec spec);
    ~SparseAnnotationVolumeWriter();

    SparseAnnotationVolumeWriter(SparseAnnotationVolumeWriter&&) noexcept;
    SparseAnnotationVolumeWriter& operator=(SparseAnnotationVolumeWriter&&) noexcept;
    SparseAnnotationVolumeWriter(const SparseAnnotationVolumeWriter&) = delete;
    SparseAnnotationVolumeWriter& operator=(const SparseAnnotationVolumeWriter&) = delete;

    void addBatch(const AnnotationPointBatch& batch);
    void finish(const std::filesystem::path& destination);

    [[nodiscard]] const SparseAnnotationVolumeSpec& spec() const noexcept;
    [[nodiscard]] std::size_t batchCount() const noexcept;
    [[nodiscard]] std::size_t occupiedChunkCount(std::size_t level = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void writeSparseAnnotationVolume(
    const SparseAnnotationVolumeSpec& spec,
    const std::vector<AnnotationPointBatch>& batches,
    const std::filesystem::path& destination);

} // namespace vc
