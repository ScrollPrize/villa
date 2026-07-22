#include "vc/core/util/SparseAnnotationVolume.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "vc/core/types/VcDataset.hpp"

namespace vc {
namespace {

struct ChunkKey {
    std::size_t z{};
    std::size_t y{};
    std::size_t x{};

    bool operator==(const ChunkKey&) const = default;
};

struct ChunkKeyHash {
    std::size_t operator()(const ChunkKey& key) const noexcept
    {
        std::size_t seed = key.z;
        seed ^= key.y + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        seed ^= key.x + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

using Chunk = std::vector<std::uint16_t>;
using ChunkMap = std::unordered_map<ChunkKey, Chunk, ChunkKeyHash>;

struct LevelGeometry {
    std::array<std::size_t, 3> shape;
    std::array<std::size_t, 3> chunkShape;
    std::array<double, 3> scale;
    std::array<double, 3> translation;
};

void requirePositiveShape(const std::array<std::size_t, 3>& shape, const char* what)
{
    if (std::ranges::any_of(shape, [](std::size_t value) { return value == 0; }))
        throw std::invalid_argument(std::string(what) + " must contain three positive values");
}

void validateSpec(const SparseAnnotationVolumeSpec& spec)
{
    requirePositiveShape(spec.shapeZYX, "shapeZYX");
    requirePositiveShape(spec.chunkShapeZYX, "chunkShapeZYX");
    if (spec.compressionLevel < 0)
        throw std::invalid_argument("compressionLevel must be nonnegative");
    if (!spec.rootAttributes.is_object())
        throw std::invalid_argument("rootAttributes must be a JSON object");
    if (spec.rootAttributes.contains("vc_annotation_volume") ||
        spec.rootAttributes.contains("multiscales")) {
        throw std::invalid_argument(
            "rootAttributes may not replace reserved vc_annotation_volume or multiscales keys");
    }
    for (const auto& level : spec.pyramidLevels) {
        requirePositiveShape(level.shapeZYX, "pyramid level shapeZYX");
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(level.scaleZYX[axis]) || level.scaleZYX[axis] <= 0.0)
                throw std::invalid_argument("pyramid level scaleZYX must be finite and positive");
            if (!std::isfinite(level.translationZYX[axis]))
                throw std::invalid_argument("pyramid level translationZYX must be finite");
        }
    }
}

std::array<double, 3> toZyx(const std::array<double, 3>& point,
                            AnnotationCoordinateOrder order)
{
    if (order == AnnotationCoordinateOrder::ZYX)
        return point;
    return {point[2], point[1], point[0]};
}

std::size_t checkedChunkElementCount(const std::array<std::size_t, 3>& shape)
{
    std::size_t result = 1;
    for (const auto value : shape) {
        if (value > std::numeric_limits<std::size_t>::max() / result)
            throw std::overflow_error("chunkShapeZYX is too large");
        result *= value;
    }
    if (result > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))
        throw std::overflow_error("chunkShapeZYX byte size is too large");
    return result;
}

utils::Json transformJson(const LevelGeometry& level)
{
    auto array3 = [](const std::array<double, 3>& values) {
        utils::Json result = utils::Json::array();
        for (const double value : values)
            result.push_back(value);
        return result;
    };
    utils::Json transforms = utils::Json::array();
    transforms.push_back(utils::Json{{"type", "scale"}, {"scale", array3(level.scale)}});
    if (std::ranges::any_of(level.translation, [](double value) { return value != 0.0; })) {
        transforms.push_back(
            utils::Json{{"type", "translation"}, {"translation", array3(level.translation)}});
    }
    return transforms;
}

void writeMetadataFile(const std::filesystem::path& destination,
                       const SparseAnnotationVolumeSpec& spec)
{
    utils::Json metadata{
        {"type", "vol"},
        {"uuid", destination.filename().string()},
        {"name", destination.filename().string()},
        {"format", "zarr"},
        {"width", static_cast<std::uint64_t>(spec.shapeZYX[2])},
        {"height", static_cast<std::uint64_t>(spec.shapeZYX[1])},
        {"slices", static_cast<std::uint64_t>(spec.shapeZYX[0])},
        {"min", 0},
        {"max", 65535},
    };
    std::ofstream out(destination / "meta.json");
    if (!out)
        throw std::runtime_error("failed to write annotation volume meta.json");
    out << metadata.dump(2) << '\n';
}

} // namespace

struct SparseAnnotationVolumeWriter::Impl {
    explicit Impl(SparseAnnotationVolumeSpec inputSpec)
        : spec(std::move(inputSpec))
    {
        validateSpec(spec);
        auto makeLevel = [this](const std::array<std::size_t, 3>& shape,
                                const std::array<double, 3>& scale,
                                const std::array<double, 3>& translation) {
            std::array<std::size_t, 3> chunkShape{};
            for (std::size_t axis = 0; axis < 3; ++axis)
                chunkShape[axis] = std::min(spec.chunkShapeZYX[axis], shape[axis]);
            checkedChunkElementCount(chunkShape);
            return LevelGeometry{shape, chunkShape, scale, translation};
        };
        levels.push_back(makeLevel(spec.shapeZYX, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}));
        for (const auto& level : spec.pyramidLevels) {
            levels.push_back(
                makeLevel(level.shapeZYX, level.scaleZYX, level.translationZYX));
        }
        chunks.resize(levels.size());
    }

    SparseAnnotationVolumeSpec spec;
    std::vector<LevelGeometry> levels;
    std::vector<ChunkMap> chunks;
    std::size_t batches{};
    bool finished{false};

    void stamp(std::size_t levelIndex,
               const std::array<double, 3>& levelPoint,
               double baseRadius,
               std::uint16_t label)
    {
        const auto& level = levels[levelIndex];
        const auto& chunkShape = level.chunkShape;
        const std::size_t chunkElements = checkedChunkElementCount(chunkShape);

        std::array<long long, 3> center{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (levelPoint[axis] < static_cast<double>(std::numeric_limits<long long>::min()) ||
                levelPoint[axis] > static_cast<double>(std::numeric_limits<long long>::max()))
                return;
            center[axis] = std::llround(levelPoint[axis]);
        }

        std::array<long long, 3> extent{};
        for (std::size_t axis = 0; axis < 3; ++axis)
            extent[axis] = static_cast<long long>(std::ceil(baseRadius / level.scale[axis]));

        for (long long dz = -extent[0]; dz <= extent[0]; ++dz) {
            for (long long dy = -extent[1]; dy <= extent[1]; ++dy) {
                for (long long dx = -extent[2]; dx <= extent[2]; ++dx) {
                    const std::array<long long, 3> delta{dz, dy, dx};
                    double distanceSquared = 0.0;
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        const double baseDelta = static_cast<double>(delta[axis]) * level.scale[axis];
                        distanceSquared += baseDelta * baseDelta;
                    }
                    if (distanceSquared > baseRadius * baseRadius)
                        continue;

                    const std::array<long long, 3> voxel{
                        center[0] + dz, center[1] + dy, center[2] + dx};
                    bool inside = true;
                    for (std::size_t axis = 0; axis < 3; ++axis) {
                        inside = inside && voxel[axis] >= 0 &&
                                 static_cast<unsigned long long>(voxel[axis]) < level.shape[axis];
                    }
                    if (!inside)
                        continue;

                    const ChunkKey key{
                        static_cast<std::size_t>(voxel[0]) / chunkShape[0],
                        static_cast<std::size_t>(voxel[1]) / chunkShape[1],
                        static_cast<std::size_t>(voxel[2]) / chunkShape[2]};
                    auto [it, inserted] = chunks[levelIndex].try_emplace(key);
                    if (inserted)
                        it->second.assign(chunkElements, spec.fillValue);
                    const std::size_t lz = static_cast<std::size_t>(voxel[0]) % chunkShape[0];
                    const std::size_t ly = static_cast<std::size_t>(voxel[1]) % chunkShape[1];
                    const std::size_t lx = static_cast<std::size_t>(voxel[2]) % chunkShape[2];
                    it->second[(lz * chunkShape[1] + ly) * chunkShape[2] + lx] = label;
                }
            }
        }
    }

    void rasterizeAtLevel(std::size_t levelIndex,
                          const AnnotationPointBatch& batch,
                          const std::vector<std::array<double, 3>>& basePoints)
    {
        const auto& level = levels[levelIndex];
        std::vector<std::array<double, 3>> points;
        points.reserve(basePoints.size());
        for (const auto& point : basePoints) {
            std::array<double, 3> transformed{};
            for (std::size_t axis = 0; axis < 3; ++axis)
                transformed[axis] = (point[axis] - level.translation[axis]) / level.scale[axis];
            points.push_back(transformed);
        }

        if (batch.geometryMode == AnnotationGeometryMode::Points || points.size() < 2) {
            for (const auto& point : points)
                stamp(levelIndex, point, batch.radius, batch.label);
            return;
        }

        for (std::size_t segment = 1; segment < points.size(); ++segment) {
            const auto& start = points[segment - 1];
            const auto& end = points[segment];
            const double maxDelta = std::max({std::abs(end[0] - start[0]),
                                              std::abs(end[1] - start[1]),
                                              std::abs(end[2] - start[2])});
            const std::size_t steps = static_cast<std::size_t>(std::ceil(maxDelta));
            if (steps == 0) {
                stamp(levelIndex, start, batch.radius, batch.label);
                continue;
            }
            for (std::size_t step = 0; step <= steps; ++step) {
                const double t = static_cast<double>(step) / static_cast<double>(steps);
                std::array<double, 3> point{};
                for (std::size_t axis = 0; axis < 3; ++axis)
                    point[axis] = start[axis] + t * (end[axis] - start[axis]);
                stamp(levelIndex, point, batch.radius, batch.label);
            }
        }
    }
};

SparseAnnotationVolumeWriter::SparseAnnotationVolumeWriter(SparseAnnotationVolumeSpec spec)
    : impl_(std::make_unique<Impl>(std::move(spec)))
{
}

SparseAnnotationVolumeWriter::~SparseAnnotationVolumeWriter() = default;
SparseAnnotationVolumeWriter::SparseAnnotationVolumeWriter(SparseAnnotationVolumeWriter&&) noexcept = default;
SparseAnnotationVolumeWriter& SparseAnnotationVolumeWriter::operator=(SparseAnnotationVolumeWriter&&) noexcept = default;

void SparseAnnotationVolumeWriter::addBatch(const AnnotationPointBatch& batch)
{
    if (!impl_)
        throw std::logic_error("cannot use a moved-from SparseAnnotationVolumeWriter");
    if (impl_->finished)
        throw std::logic_error("cannot add a batch after finish");
    if (batch.label == 0)
        throw std::invalid_argument("annotation label 0 is reserved for background");
    if (!std::isfinite(batch.radius) || batch.radius < 0.0)
        throw std::invalid_argument("annotation radius must be finite and nonnegative");

    std::vector<std::array<double, 3>> points;
    points.reserve(batch.coordinates.size());
    for (const auto& input : batch.coordinates) {
        if (std::ranges::any_of(input, [](double value) { return !std::isfinite(value); }))
            throw std::invalid_argument("annotation coordinates must be finite");
        points.push_back(toZyx(input, batch.coordinateOrder));
    }
    for (std::size_t level = 0; level < impl_->levels.size(); ++level)
        impl_->rasterizeAtLevel(level, batch, points);
    ++impl_->batches;
}

void SparseAnnotationVolumeWriter::finish(const std::filesystem::path& destination)
{
    if (!impl_)
        throw std::logic_error("cannot use a moved-from SparseAnnotationVolumeWriter");
    if (impl_->finished)
        throw std::logic_error("SparseAnnotationVolumeWriter::finish may only be called once");
    if (destination.empty())
        throw std::invalid_argument("annotation volume destination must not be empty");
    if (std::filesystem::exists(destination) && !std::filesystem::is_empty(destination))
        throw std::invalid_argument("annotation volume destination already exists and is not empty");

    std::filesystem::create_directories(destination);
    for (std::size_t levelIndex = 0; levelIndex < impl_->levels.size(); ++levelIndex) {
        const auto& level = impl_->levels[levelIndex];
        auto dataset = createZarrDataset(destination,
                                         std::to_string(levelIndex),
                                         {level.shape[0], level.shape[1], level.shape[2]},
                                         {level.chunkShape[0], level.chunkShape[1], level.chunkShape[2]},
                                         VcDtype::uint16,
                                         impl_->spec.compressor,
                                         ".",
                                         impl_->spec.fillValue,
                                         impl_->spec.compressionLevel);
        for (const auto& [key, chunk] : impl_->chunks[levelIndex]) {
            dataset->writeChunkSkipEmpty(
                key.z, key.y, key.x, chunk.data(), chunk.size() * sizeof(std::uint16_t));
        }
    }

    utils::Json attrs = impl_->spec.rootAttributes;
    utils::Json schema;
    schema["version"] = 1;
    schema["dtype"] = "uint16";
    schema["background_label"] = static_cast<std::uint64_t>(impl_->spec.fillValue);
    schema["axes"] = utils::Json::array();
    for (const char* axis : {"z", "y", "x"})
        schema["axes"].push_back(axis);
    attrs["vc_annotation_volume"] = std::move(schema);

    utils::Json multiscale;
    multiscale["version"] = "0.4";
    multiscale["axes"] = utils::Json::array();
    for (const char* axis : {"z", "y", "x"})
        multiscale["axes"].push_back(utils::Json{{"name", axis}, {"type", "space"}});
    multiscale["datasets"] = utils::Json::array();
    for (std::size_t levelIndex = 0; levelIndex < impl_->levels.size(); ++levelIndex) {
        multiscale["datasets"].push_back(utils::Json{
            {"path", std::to_string(levelIndex)},
            {"coordinateTransformations", transformJson(impl_->levels[levelIndex])},
        });
    }
    attrs["multiscales"] = utils::Json::array();
    attrs["multiscales"].push_back(std::move(multiscale));
    writeZarrAttributes(destination, attrs);
    writeMetadataFile(destination, impl_->spec);
    impl_->finished = true;
}

const SparseAnnotationVolumeSpec& SparseAnnotationVolumeWriter::spec() const noexcept
{
    return impl_->spec;
}

std::size_t SparseAnnotationVolumeWriter::batchCount() const noexcept
{
    return impl_->batches;
}

std::size_t SparseAnnotationVolumeWriter::occupiedChunkCount(std::size_t level) const
{
    if (level >= impl_->chunks.size())
        throw std::out_of_range("annotation pyramid level is out of range");
    return impl_->chunks[level].size();
}

void writeSparseAnnotationVolume(const SparseAnnotationVolumeSpec& spec,
                                 const std::vector<AnnotationPointBatch>& batches,
                                 const std::filesystem::path& destination)
{
    SparseAnnotationVolumeWriter writer(spec);
    for (const auto& batch : batches)
        writer.addBatch(batch);
    writer.finish(destination);
}

} // namespace vc
