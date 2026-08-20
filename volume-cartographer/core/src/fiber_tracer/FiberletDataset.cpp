#include "vc/fiber_tracer/FiberletDataset.hpp"

#include "vc/core/util/AtomicFile.hpp"

#include <nlohmann/json.hpp>
#include <utils/hash.hpp>

#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vc::fiber_tracer
{
namespace
{

using json = nlohmann::json;

const char* profileName(FiberletStorageProfile profile)
{
    switch (profile) {
        case FiberletStorageProfile::Float32Cache:
            return "float32_cache";
        case FiberletStorageProfile::CompactQuantized:
            return "compact_quantized";
    }
    throw std::invalid_argument("unknown fiberlet storage profile");
}

FiberletStorageProfile parseProfile(const std::string& value)
{
    if (value == "float32_cache")
        return FiberletStorageProfile::Float32Cache;
    if (value == "compact_quantized")
        return FiberletStorageProfile::CompactQuantized;
    throw std::invalid_argument("unknown fiberlet storage profile in metadata");
}

const char* kindName(FiberletDatasetKind kind)
{
    return kind == FiberletDatasetKind::Anchors ? "anchors" : "fiberlets";
}

FiberletDatasetKind parseKind(const std::string& value)
{
    if (value == "anchors")
        return FiberletDatasetKind::Anchors;
    if (value == "fiberlets")
        return FiberletDatasetKind::Fiberlets;
    throw std::invalid_argument("unknown fiberlet dataset kind in metadata");
}

std::string hexFingerprint(const std::array<std::uint8_t, 32>& fingerprint)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const auto value : fingerprint) {
        result.push_back(digits[value >> 4]);
        result.push_back(digits[value & 0x0f]);
    }
    return result;
}

std::array<std::uint8_t, 32> parseFingerprint(const std::string& value)
{
    if (value.size() != 64)
        throw std::invalid_argument("fiberlet dataset fingerprint must contain 64 hexadecimal digits");
    const auto nibble = [](char c) -> std::uint8_t {
        if (c >= '0' && c <= '9')
            return static_cast<std::uint8_t>(c - '0');
        if (c >= 'a' && c <= 'f')
            return static_cast<std::uint8_t>(c - 'a' + 10);
        throw std::invalid_argument("fiberlet dataset fingerprint is not lowercase hexadecimal");
    };
    std::array<std::uint8_t, 32> result{};
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = static_cast<std::uint8_t>((nibble(value[index * 2]) << 4) | nibble(value[index * 2 + 1]));
    return result;
}

json metadataJson(const FiberletDatasetMetadata& metadata)
{
    return {
        {"vc_format", "fiberlet_dataset"},
        {"format_version", 1},
        {"dataset_kind", kindName(metadata.kind)},
        {"encoding_profile", profileName(metadata.profile)},
        {"chunk_grid_shape_zyx", metadata.chunkGridShapeZYX},
        {"coordinate_origin_zyx", metadata.coordinateOriginZYX},
        {"coordinate_units_per_chunk_zyx", metadata.coordinateUnitsPerChunkZYX},
        {"maximum_endpoint_reach_coordinate_units_zyx", metadata.maximumEndpointReachCoordinateUnitsZYX},
        {"dataset_fingerprint", hexFingerprint(metadata.datasetFingerprint)},
        {"spatial_chunk_side_base", metadata.spatialChunkSideBaseVoxels},
        {"coordinate_bits", metadata.coordinateBits},
        {"delta_bits", metadata.deltaBits},
        {"route_count_bits", metadata.routeCountBits},
        {"route_lattice_bits", metadata.routeLatticeBits},
        {"cost_bits", metadata.costBits},
        {"position_quantum_base", metadata.positionQuantumBaseVoxels == 0 ? json(nullptr) : json(metadata.positionQuantumBaseVoxels)},
        {"prediction_to_base", metadata.predictionToBaseScale},
        {"algorithm_fingerprint", metadata.algorithmFingerprint},
        {"fiber_manifest", metadata.fiberManifest},
        {"fiber_manifest_hash", metadata.fiberManifestHash},
        {"normal_manifest", metadata.normalManifest},
        {"normal_manifest_hash", metadata.normalManifestHash},
        {"build_state", "partial"},
    };
}

FiberletDatasetMetadata parseMetadata(const json& value)
{
    static const std::vector<std::string> required{
        "vc_format",
        "format_version",
        "dataset_kind",
        "encoding_profile",
        "chunk_grid_shape_zyx",
        "coordinate_origin_zyx",
        "coordinate_units_per_chunk_zyx",
        "maximum_endpoint_reach_coordinate_units_zyx",
        "dataset_fingerprint",
        "spatial_chunk_side_base",
        "coordinate_bits",
        "delta_bits",
        "route_count_bits",
        "route_lattice_bits",
        "cost_bits",
        "position_quantum_base",
        "prediction_to_base",
        "algorithm_fingerprint",
        "fiber_manifest",
        "fiber_manifest_hash",
        "normal_manifest",
        "normal_manifest_hash",
        "build_state"};
    if (!value.is_object() || value.size() != required.size())
        throw std::invalid_argument("fiberlet dataset metadata has unknown or missing fields");
    for (const auto& name : required) {
        if (!value.contains(name))
            throw std::invalid_argument("fiberlet dataset metadata is missing " + name);
    }
    if (value.at("vc_format") != "fiberlet_dataset" || value.at("format_version") != 1 || value.at("build_state") != "partial")
        throw std::invalid_argument("fiberlet dataset metadata header is invalid");
    FiberletDatasetMetadata result;
    result.kind = parseKind(value.at("dataset_kind").get<std::string>());
    result.profile = parseProfile(value.at("encoding_profile").get<std::string>());
    result.chunkGridShapeZYX = value.at("chunk_grid_shape_zyx").get<std::array<std::int32_t, 3>>();
    result.coordinateOriginZYX = value.at("coordinate_origin_zyx").get<std::array<std::int64_t, 3>>();
    result.coordinateUnitsPerChunkZYX = value.at("coordinate_units_per_chunk_zyx").get<std::array<std::int64_t, 3>>();
    result.maximumEndpointReachCoordinateUnitsZYX = value.at("maximum_endpoint_reach_coordinate_units_zyx").get<std::array<std::int64_t, 3>>();
    result.datasetFingerprint = parseFingerprint(value.at("dataset_fingerprint").get<std::string>());
    result.spatialChunkSideBaseVoxels = value.at("spatial_chunk_side_base").get<std::uint32_t>();
    result.coordinateBits = value.at("coordinate_bits").get<std::uint8_t>();
    result.deltaBits = value.at("delta_bits").get<std::uint8_t>();
    result.routeCountBits = value.at("route_count_bits").get<std::uint8_t>();
    result.routeLatticeBits = value.at("route_lattice_bits").get<std::uint8_t>();
    result.costBits = value.at("cost_bits").get<std::uint8_t>();
    result.positionQuantumBaseVoxels = value.at("position_quantum_base").is_null() ? 0 : value.at("position_quantum_base").get<std::uint32_t>();
    result.predictionToBaseScale = value.at("prediction_to_base").get<double>();
    result.algorithmFingerprint = value.at("algorithm_fingerprint").get<std::string>();
    result.fiberManifest = value.at("fiber_manifest").get<std::string>();
    result.fiberManifestHash = value.at("fiber_manifest_hash").get<std::string>();
    result.normalManifest = value.at("normal_manifest").get<std::string>();
    result.normalManifestHash = value.at("normal_manifest_hash").get<std::string>();
    return result;
}

json arrayMetadata(const FiberletDatasetMetadata& metadata, FiberletStorageChunkKind kind)
{
    const char* sampleFormat = kind == FiberletStorageChunkKind::Anchors          ? "fiberlet-anchor-v1"
                               : kind == FiberletStorageChunkKind::FiberletPrefix ? "fiberlet-edge-prefix-v1"
                                                                                  : "fiberlet-route-v1";
    return {
        {"zarr_format", 2},
        {"shape", metadata.chunkGridShapeZYX},
        {"chunks", {1, 1, 1}},
        {"dtype", "|O"},
        {"fill_value", nullptr},
        {"order", "C"},
        {"filters", {{{"id", "vc-fiberlet-chunk"}, {"codec_version", 1}, {"sample_format", sampleFormat}}}},
        {"compressor", nullptr},
    };
}

std::string readText(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), {});
}

std::optional<std::vector<std::byte>> readBytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        return std::nullopt;
    const auto end = input.tellg();
    if (end < 0)
        throw std::runtime_error("cannot size " + path.string());
    std::vector<std::byte> result(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
    if (!input)
        throw std::runtime_error("cannot read " + path.string());
    return result;
}

std::filesystem::path arrayDirectory(const std::filesystem::path& root, FiberletStorageChunkKind kind)
{
    if (kind == FiberletStorageChunkKind::Anchors)
        return root / "anchors";
    if (kind == FiberletStorageChunkKind::FiberletPrefix)
        return root / "prefix";
    return root / "routes";
}

std::filesystem::path completionPath(const std::filesystem::path& root, const vc::render::ChunkKey& key)
{
    return root / "complete" / (std::to_string(key.iz) + "." + std::to_string(key.iy) + "." + std::to_string(key.ix));
}

std::uint64_t bytesHash(std::span<const std::byte> bytes)
{
    std::uint64_t result = utils::fnv_offset_basis;
    for (const auto byte : bytes) {
        result ^= std::to_integer<unsigned char>(byte);
        result *= utils::fnv_prime;
    }
    return result;
}

json completionJson(std::span<const std::byte> prefix, std::span<const std::byte> routes)
{
    return {
        {"vc_format", "fiberlet_chunk_completion"},
        {"format_version", 1},
        {"prefix_hash", bytesHash(prefix)},
        {"routes_hash", bytesHash(routes)},
    };
}

json parseCompletion(const std::filesystem::path& path)
{
    const auto value = json::parse(readText(path));
    if (!value.is_object() || value.size() != 4 || value.value("vc_format", "") != "fiberlet_chunk_completion" ||
        value.value("format_version", 0) != 1 || !value.contains("prefix_hash") || !value.contains("routes_hash")) {
        throw std::invalid_argument("fiberlet chunk completion marker is invalid");
    }
    (void)value.at("prefix_hash").get<std::uint64_t>();
    (void)value.at("routes_hash").get<std::uint64_t>();
    return value;
}

class GeneratedFetcher final : public vc::render::IChunkFetcher
{
public:
    GeneratedFetcher(std::shared_ptr<FiberletChunkDataset> dataset, FiberletStorageChunkKind kind, FiberletChunkGenerator generator)
        : dataset_(std::move(dataset)), kind_(kind), generator_(std::move(generator))
    {
    }

    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey& key) override
    {
        vc::render::ChunkFetchResult result;
        try {
            if (auto bytes = dataset_->readChunk(kind_, key)) {
                result.status = vc::render::ChunkFetchStatus::Found;
                result.bytes = materializeFiberletPayload(*bytes);
                return result;
            }
            auto bytes = generator_(kind_, key, dataset_->codecConfig(kind_, key));
            dataset_->validateChunk(kind_, key, bytes);
            dataset_->publishChunk(kind_, key, bytes);
            result.status = vc::render::ChunkFetchStatus::Found;
            result.bytes = materializeFiberletPayload(bytes);
        } catch (const std::invalid_argument& error) {
            result.status = vc::render::ChunkFetchStatus::DecodeError;
            result.message = error.what();
        } catch (const std::exception& error) {
            result.status = vc::render::ChunkFetchStatus::IoError;
            result.message = error.what();
        }
        return result;
    }

    std::string persistentCacheExtension(const vc::render::ChunkKey&) const override { return ".fiberlet"; }

    std::optional<std::string> sourceChunkKey(const vc::render::ChunkKey& key) const override
    {
        return dataset_->chunkPath(kind_, key).string();
    }

private:
    std::shared_ptr<FiberletChunkDataset> dataset_;
    FiberletStorageChunkKind kind_;
    FiberletChunkGenerator generator_;
};

}  // namespace

FiberletChunkDataset::FiberletChunkDataset(std::filesystem::path root, FiberletDatasetMetadata metadata)
    : root_(std::move(root)), metadata_(std::move(metadata))
{
}

std::shared_ptr<FiberletChunkDataset> FiberletChunkDataset::createOrOpen(std::filesystem::path root, const FiberletDatasetMetadata& metadata)
{
    if (metadata.algorithmFingerprint.empty())
        throw std::invalid_argument("fiberlet algorithm fingerprint must not be empty");
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (metadata.chunkGridShapeZYX[axis] <= 0 || metadata.coordinateUnitsPerChunkZYX[axis] <= 0 ||
            metadata.maximumEndpointReachCoordinateUnitsZYX[axis] < 0)
            throw std::invalid_argument("fiberlet dataset grid or coordinate chunk size is invalid");
    }
    const auto expected = metadataJson(metadata);
    if (std::filesystem::exists(root / ".zattrs")) {
        const auto group = json::parse(readText(root / ".zgroup"));
        if (group != json{{"zarr_format", 2}})
            throw std::invalid_argument("fiberlet dataset .zgroup metadata is invalid");
        const auto storedJson = json::parse(readText(root / ".zattrs"));
        const auto parsed = parseMetadata(storedJson);
        if (metadataJson(parsed) != expected)
            throw std::invalid_argument("fiberlet dataset metadata does not match the requested configuration");
        const std::vector<FiberletStorageChunkKind> kinds =
            metadata.kind == FiberletDatasetKind::Anchors
                ? std::vector<FiberletStorageChunkKind>{FiberletStorageChunkKind::Anchors}
                : std::vector<FiberletStorageChunkKind>{FiberletStorageChunkKind::FiberletPrefix, FiberletStorageChunkKind::FiberletRoutes};
        for (const auto kind : kinds) {
            const auto storedArray = json::parse(readText(arrayDirectory(root, kind) / ".zarray"));
            if (storedArray != arrayMetadata(metadata, kind))
                throw std::invalid_argument("fiberlet dataset .zarray metadata is invalid");
        }
    } else {
        std::filesystem::create_directories(root);
        vc::core::util::atomicWriteString(root / ".zgroup", json{{"zarr_format", 2}}.dump(2) + "\n");
        vc::core::util::atomicWriteString(root / ".zattrs", expected.dump(2) + "\n");
        const std::vector<FiberletStorageChunkKind> kinds =
            metadata.kind == FiberletDatasetKind::Anchors
                ? std::vector<FiberletStorageChunkKind>{FiberletStorageChunkKind::Anchors}
                : std::vector<FiberletStorageChunkKind>{FiberletStorageChunkKind::FiberletPrefix, FiberletStorageChunkKind::FiberletRoutes};
        for (const auto kind : kinds) {
            const auto directory = arrayDirectory(root, kind);
            std::filesystem::create_directories(directory);
            vc::core::util::atomicWriteString(directory / ".zarray", arrayMetadata(metadata, kind).dump(2) + "\n");
        }
        if (metadata.kind == FiberletDatasetKind::Fiberlets)
            std::filesystem::create_directories(root / "complete");
    }
    return std::shared_ptr<FiberletChunkDataset>(new FiberletChunkDataset(std::move(root), metadata));
}

const std::filesystem::path& FiberletChunkDataset::root() const noexcept
{
    return root_;
}
const FiberletDatasetMetadata& FiberletChunkDataset::metadata() const noexcept
{
    return metadata_;
}

FiberletStorageCodecConfig FiberletChunkDataset::codecConfig(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const
{
    if ((metadata_.kind == FiberletDatasetKind::Anchors) != (kind == FiberletStorageChunkKind::Anchors))
        throw std::invalid_argument("fiberlet chunk kind does not belong to this dataset");
    const int expectedLevel = kind == FiberletStorageChunkKind::FiberletRoutes ? 1 : 0;
    if (key.level != expectedLevel)
        throw std::invalid_argument("fiberlet chunk level does not match its payload kind");
    if (key.level < 0 || key.iz < 0 || key.iy < 0 || key.ix < 0 || key.iz >= metadata_.chunkGridShapeZYX[0] ||
        key.iy >= metadata_.chunkGridShapeZYX[1] || key.ix >= metadata_.chunkGridShapeZYX[2])
        throw std::invalid_argument("fiberlet chunk key is outside its dataset grid");
    FiberletStorageCodecConfig result;
    result.profile = metadata_.profile;
    result.chunkZYX = {key.iz, key.iy, key.ix};
    result.datasetFingerprint = metadata_.datasetFingerprint;
    const std::array<std::int64_t, 3> chunk{key.iz, key.iy, key.ix};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (chunk[axis] > (std::numeric_limits<std::int64_t>::max() - metadata_.coordinateOriginZYX[axis]) / metadata_.coordinateUnitsPerChunkZYX[axis])
            throw std::invalid_argument("fiberlet chunk coordinate origin overflows int64");
        result.coordinateOriginZYX[axis] = metadata_.coordinateOriginZYX[axis] + chunk[axis] * metadata_.coordinateUnitsPerChunkZYX[axis];
    }
    result.coordinateBits = metadata_.coordinateBits;
    result.deltaBits = metadata_.deltaBits;
    result.routeCountBits = metadata_.routeCountBits;
    result.routeLatticeBits = metadata_.routeLatticeBits;
    result.costBits = metadata_.costBits;
    result.positionQuantumBaseVoxels = metadata_.positionQuantumBaseVoxels;
    result.predictionToBaseScale = metadata_.predictionToBaseScale;
    return result;
}

std::filesystem::path FiberletChunkDataset::chunkPath(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const
{
    (void)codecConfig(kind, key);
    return arrayDirectory(root_, kind) / (std::to_string(key.iz) + "." + std::to_string(key.iy) + "." + std::to_string(key.ix));
}

std::optional<std::vector<std::byte>> FiberletChunkDataset::readChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const
{
    std::optional<json> completion;
    if (metadata_.kind == FiberletDatasetKind::Fiberlets) {
        const auto marker = completionPath(root_, key);
        if (!std::filesystem::exists(marker))
            return std::nullopt;
        completion = parseCompletion(marker);
        const auto otherKind = kind == FiberletStorageChunkKind::FiberletPrefix ? FiberletStorageChunkKind::FiberletRoutes
                                                                                : FiberletStorageChunkKind::FiberletPrefix;
        auto otherKey = key;
        otherKey.level = otherKind == FiberletStorageChunkKind::FiberletRoutes ? 1 : 0;
        if (!std::filesystem::exists(chunkPath(otherKind, otherKey)))
            throw std::invalid_argument("completed fiberlet chunk is missing its paired payload");
    }
    auto result = readBytes(chunkPath(kind, key));
    if (result) {
        validateChunk(kind, key, *result);
        if (completion) {
            const char* hashName = kind == FiberletStorageChunkKind::FiberletPrefix ? "prefix_hash" : "routes_hash";
            if (completion->at(hashName).get<std::uint64_t>() != bytesHash(*result)) {
                throw std::invalid_argument("fiberlet chunk does not match its completion marker");
            }
        }
    } else if (completion) {
        throw std::invalid_argument("completed fiberlet chunk payload is missing");
    }
    return result;
}

void FiberletChunkDataset::publishChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const
{
    validateChunk(kind, key, bytes);
    const auto path = chunkPath(kind, key);
    if (const auto existing = readBytes(path)) {
        if (existing->size() != bytes.size() || !std::equal(existing->begin(), existing->end(), bytes.begin()))
            throw std::invalid_argument("fiberlet chunk publication conflicts with existing bytes");
    } else {
        vc::core::util::atomicWriteBytes(path, bytes);
    }
    if (metadata_.kind != FiberletDatasetKind::Fiberlets)
        return;

    const vc::render::ChunkKey prefixKey{0, key.iz, key.iy, key.ix};
    const vc::render::ChunkKey routeKey{1, key.iz, key.iy, key.ix};
    const auto prefix = readBytes(chunkPath(FiberletStorageChunkKind::FiberletPrefix, prefixKey));
    const auto routes = readBytes(chunkPath(FiberletStorageChunkKind::FiberletRoutes, routeKey));
    if (!prefix || !routes)
        return;
    validateChunk(FiberletStorageChunkKind::FiberletPrefix, prefixKey, *prefix);
    validateChunk(FiberletStorageChunkKind::FiberletRoutes, routeKey, *routes);
    const auto expected = completionJson(*prefix, *routes);
    const auto marker = completionPath(root_, key);
    if (std::filesystem::exists(marker)) {
        if (parseCompletion(marker) != expected)
            throw std::invalid_argument("fiberlet chunk completion marker conflicts with payloads");
    } else {
        vc::core::util::atomicWriteString(marker, expected.dump(2) + "\n");
    }
}

void FiberletChunkDataset::validateChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const
{
    FiberletStorageCodecConfig decoded;
    if (kind == FiberletStorageChunkKind::Anchors)
        decoded = deserializeFiberletAnchors(bytes).config;
    else if (kind == FiberletStorageChunkKind::FiberletPrefix)
        decoded = deserializeFiberletPrefixes(bytes).config;
    else
        decoded = deserializeFiberletRoutes(bytes).config;
    const auto expected = codecConfig(kind, key);
    if (decoded.profile != expected.profile || decoded.chunkZYX != expected.chunkZYX || decoded.datasetFingerprint != expected.datasetFingerprint ||
        decoded.coordinateOriginZYX != expected.coordinateOriginZYX || decoded.coordinateBits != expected.coordinateBits ||
        decoded.deltaBits != expected.deltaBits || decoded.routeCountBits != expected.routeCountBits ||
        decoded.routeLatticeBits != expected.routeLatticeBits || decoded.costBits != expected.costBits ||
        decoded.positionQuantumBaseVoxels != expected.positionQuantumBaseVoxels || decoded.predictionToBaseScale != expected.predictionToBaseScale)
        throw std::invalid_argument("fiberlet chunk header does not match its dataset metadata");
}

std::shared_ptr<vc::render::ChunkCache> createGeneratedFiberletChunkCache(
    std::shared_ptr<FiberletChunkDataset> dataset, FiberletChunkGenerator generator, vc::render::ChunkCache::Options options)
{
    if (!dataset || !generator)
        throw std::invalid_argument("generated fiberlet cache requires a dataset and generator");
    const auto shape = dataset->metadata().chunkGridShapeZYX;
    std::vector<vc::render::ChunkCache::LevelInfo> levels;
    std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers;
    if (dataset->metadata().kind == FiberletDatasetKind::Anchors) {
        levels.push_back({shape, {1, 1, 1}, {}});
        fetchers.push_back(std::make_shared<GeneratedFetcher>(dataset, FiberletStorageChunkKind::Anchors, generator));
    } else {
        levels.push_back({shape, {1, 1, 1}, {}});
        levels.push_back({shape, {1, 1, 1}, {}});
        fetchers.push_back(std::make_shared<GeneratedFetcher>(dataset, FiberletStorageChunkKind::FiberletPrefix, generator));
        fetchers.push_back(std::make_shared<GeneratedFetcher>(dataset, FiberletStorageChunkKind::FiberletRoutes, generator));
    }
    options.detectAllFillChunks = false;
    options.persistentCachePath.reset();
    return std::make_shared<vc::render::ChunkCache>(std::move(levels), std::move(fetchers), 0.0, vc::render::ChunkDtype::Opaque, std::move(options));
}

}  // namespace vc::fiber_tracer
