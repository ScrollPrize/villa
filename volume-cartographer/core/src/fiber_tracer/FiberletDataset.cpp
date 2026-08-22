#include "vc/fiber_tracer/FiberletDataset.hpp"

#include "vc/core/util/AtomicFile.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
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
        case FiberletStorageProfile::CompactDirectionsFixedCost:
            return "compact_directions_fixed_cost";
    }
    throw std::invalid_argument("unknown fiberlet storage profile");
}

FiberletStorageProfile parseProfile(const std::string& value)
{
    if (value == "float32_cache")
        return FiberletStorageProfile::Float32Cache;
    if (value == "compact_quantized")
        return FiberletStorageProfile::CompactQuantized;
    if (value == "compact_directions_fixed_cost")
        return FiberletStorageProfile::CompactDirectionsFixedCost;
    throw std::invalid_argument("unknown fiberlet storage profile in metadata");
}

const char* kindName(FiberletDatasetKind kind)
{
    switch (kind) {
        case FiberletDatasetKind::Anchors:
            return "anchors";
        case FiberletDatasetKind::Fiberlets:
            return "fiberlets";
        case FiberletDatasetKind::Combined:
            return "combined";
    }
    throw std::invalid_argument("unknown fiberlet dataset kind");
}

FiberletDatasetKind parseKind(const std::string& value)
{
    if (value == "anchors")
        return FiberletDatasetKind::Anchors;
    if (value == "fiberlets")
        return FiberletDatasetKind::Fiberlets;
    if (value == "combined")
        return FiberletDatasetKind::Combined;
    throw std::invalid_argument("unknown fiberlet dataset kind in metadata");
}

std::vector<FiberletStorageChunkKind> datasetKinds(FiberletDatasetKind kind)
{
    if (kind == FiberletDatasetKind::Anchors)
        return {FiberletStorageChunkKind::Anchors};
    if (kind == FiberletDatasetKind::Fiberlets) {
        return {FiberletStorageChunkKind::FiberletPrefix, FiberletStorageChunkKind::FiberletRoutes};
    }
    return {FiberletStorageChunkKind::Anchors, FiberletStorageChunkKind::FiberletPrefix, FiberletStorageChunkKind::FiberletRoutes};
}

bool datasetContains(FiberletDatasetKind dataset, FiberletStorageChunkKind chunk)
{
    return dataset == FiberletDatasetKind::Combined || (dataset == FiberletDatasetKind::Anchors) == (chunk == FiberletStorageChunkKind::Anchors);
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
    const char* sampleFormat = kind == FiberletStorageChunkKind::Anchors          ? "fiberlet-anchor-v2"
                               : kind == FiberletStorageChunkKind::FiberletPrefix ? "fiberlet-edge-prefix-v2"
                                                                                  : "fiberlet-route-v3";
    const int codecVersion = kind == FiberletStorageChunkKind::FiberletRoutes ? 3 : 2;
    return {
        {"zarr_format", 2},
        {"shape", metadata.chunkGridShapeZYX},
        {"chunks", {1, 1, 1}},
        {"dtype", "|O"},
        {"fill_value", nullptr},
        {"order", "C"},
        {"filters", {{{"id", "vc-fiberlet-chunk"}, {"codec_version", codecVersion}, {"sample_format", sampleFormat}}}},
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

void removeLegacyBookkeeping(const std::filesystem::path& root)
{
    std::filesystem::remove(root / "active_chunks.bin");
    std::filesystem::remove(root / "dataset.complete");
    std::filesystem::remove_all(root / "complete");
}

class GeneratedFetcher final : public vc::render::IChunkFetcher
{
public:
    GeneratedFetcher(std::shared_ptr<FiberletChunkDataset> dataset, FiberletStorageChunkKind kind, FiberletChunkGenerator generator, FiberletChunkResolvedCallback resolved)
        : dataset_(std::move(dataset)), kind_(kind), generator_(std::move(generator)), resolved_(std::move(resolved))
    {
    }

    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey& key) override
    {
        vc::render::ChunkFetchResult result;
        try {
            if (auto chunk = dataset_->readMaterializedChunk(kind_, key)) {
                result.status = vc::render::ChunkFetchStatus::Found;
                result.payload = std::move(chunk->payload);
            } else {
                auto generated = generator_(
                    kind_, key, dataset_->codecConfig(kind_, key));
                if (!generated.payload)
                    throw std::invalid_argument(
                        "fiberlet generator returned no decoded payload");
                if (!generated.alreadyPublished)
                    dataset_->publishMaterializedChunk(kind_, key, generated);
                result.status = vc::render::ChunkFetchStatus::Found;
                result.payload = std::move(generated.payload);
            }
        } catch (const std::invalid_argument& error) {
            result.status = vc::render::ChunkFetchStatus::DecodeError;
            result.message = error.what();
        } catch (const std::exception& error) {
            result.status = vc::render::ChunkFetchStatus::IoError;
            result.message = error.what();
        }
        if (resolved_) {
            try {
                resolved_(kind_, key, result.status);
            } catch (...) {
                // Progress observers must not alter an already resolved fetch.
            }
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
    FiberletChunkResolvedCallback resolved_;
};

class StoredFetcher final : public vc::render::IChunkFetcher
{
public:
    StoredFetcher(std::shared_ptr<FiberletChunkDataset> dataset, FiberletStorageChunkKind kind) : dataset_(std::move(dataset)), kind_(kind)
    {
    }

    vc::render::ChunkFetchResult fetch(const vc::render::ChunkKey& key) override
    {
        vc::render::ChunkFetchResult result;
        try {
            const auto chunk = dataset_->readMaterializedChunk(kind_, key);
            if (!chunk) {
                result.status = vc::render::ChunkFetchStatus::Missing;
            } else {
                result.status = vc::render::ChunkFetchStatus::Found;
                result.payload = chunk->payload;
            }
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
};

}  // namespace

FiberletAnchorChunkPayload::FiberletAnchorChunkPayload(FiberletDecodedAnchors decoded)
    : config(std::move(decoded.config)), anchors(std::move(decoded.anchors))
{
}

std::size_t FiberletAnchorChunkPayload::residentBytes() const noexcept
{
    return sizeof(*this) + anchors.capacity() * sizeof(FiberletStoredAnchor);
}

const FiberletStoredAnchor* FiberletAnchorChunkPayload::find(const FiberletStorageKey& key) const noexcept
{
    const auto found =
        std::lower_bound(anchors.begin(), anchors.end(), key, [](const auto& anchor, const auto& value) { return anchor.key < value; });
    return found != anchors.end() && found->key == key ? &*found : nullptr;
}

FiberletPrefixChunkPayload::FiberletPrefixChunkPayload(FiberletDecodedPrefixes decoded)
    : config(std::move(decoded.config)), prefixes(std::move(decoded.prefixes))
{
    if (prefixes.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() >> 1)) {
        throw std::overflow_error("fiberlet prefix chunk is too large for its incident index");
    }
    incidentOrder_.reserve(prefixes.size() * 2);
    for (std::size_t index = 0; index < prefixes.size(); ++index) {
        const auto encoded = static_cast<std::uint32_t>(index << 1);
        incidentOrder_.push_back(encoded);
        incidentOrder_.push_back(encoded | 1U);
    }
    std::sort(incidentOrder_.begin(), incidentOrder_.end(),
        [&](std::uint32_t left, std::uint32_t right) {
            return std::tuple{
                       endpoint(left),
                       DirectedFiberletStorageId{
                           prefixes[left >> 1].id,
                           static_cast<bool>(left & 1U)}} <
                std::tuple{
                    endpoint(right),
                    DirectedFiberletStorageId{
                        prefixes[right >> 1].id,
                        static_cast<bool>(right & 1U)}};
        });
}

std::size_t FiberletPrefixChunkPayload::residentBytes() const noexcept
{
    return sizeof(*this) +
        prefixes.capacity() * sizeof(FiberletStoredPrefix) +
        incidentOrder_.capacity() * sizeof(std::uint32_t);
}

const FiberletStoredPrefix* FiberletPrefixChunkPayload::find(
    const FiberletStorageId& id) const noexcept
{
    const auto found = std::lower_bound(
        prefixes.begin(), prefixes.end(), id,
        [](const auto& prefix, const auto& value) {
            return prefix.id < value;
        });
    return found != prefixes.end() && found->id == id ? &*found : nullptr;
}

const FiberletStorageKey& FiberletPrefixChunkPayload::endpoint(
    std::uint32_t encoded) const noexcept
{
    const auto& id = prefixes[encoded >> 1].id;
    return encoded & 1U ? id.second : id.first;
}

std::vector<FiberletIncidentPrefix> FiberletPrefixChunkPayload::incident(
    const FiberletStorageKey& key) const
{
    const auto begin = std::lower_bound(
        incidentOrder_.begin(), incidentOrder_.end(), key,
        [&](std::uint32_t encoded, const auto& value) {
            return endpoint(encoded) < value;
        });
    const auto end = std::upper_bound(
        begin, incidentOrder_.end(), key,
        [&](const auto& value, std::uint32_t encoded) {
            return value < endpoint(encoded);
        });
    std::vector<FiberletIncidentPrefix> result;
    result.reserve(static_cast<std::size_t>(end - begin));
    for (auto it = begin; it != end; ++it) {
        const auto encoded = *it;
        const auto& prefix = prefixes[encoded >> 1];
        result.push_back({
            {prefix.id, static_cast<bool>(encoded & 1U)}, prefix});
    }
    return result;
}

FiberletRouteChunkPayload::FiberletRouteChunkPayload(
    FiberletDecodedRoutes decoded)
    : config(std::move(decoded.config))
    , routes(std::move(decoded.routes))
{
}

std::size_t FiberletRouteChunkPayload::residentBytes() const noexcept
{
    std::size_t result = sizeof(*this) +
        routes.capacity() * sizeof(FiberletStoredRoute);
    for (const auto& route : routes) {
        const auto bytes = route.middleUV.capacity() *
            sizeof(std::array<std::int16_t, 2>);
        if (bytes > std::numeric_limits<std::size_t>::max() - result)
            return std::numeric_limits<std::size_t>::max();
        result += bytes;
        const auto costBytes = route.segmentCostDensities.capacity() * sizeof(float);
        if (costBytes > std::numeric_limits<std::size_t>::max() - result)
            return std::numeric_limits<std::size_t>::max();
        result += costBytes;
    }
    return result;
}

std::shared_ptr<const vc::render::DecodedChunkPayload>
decodeFiberletChunkPayload(
    FiberletStorageChunkKind kind,
    std::span<const std::byte> bytes)
{
    if (kind == FiberletStorageChunkKind::Anchors) {
        return std::make_shared<const FiberletAnchorChunkPayload>(
            deserializeFiberletAnchors(bytes));
    }
    if (kind == FiberletStorageChunkKind::FiberletPrefix) {
        return std::make_shared<const FiberletPrefixChunkPayload>(
            deserializeFiberletPrefixes(bytes));
    }
    if (kind == FiberletStorageChunkKind::FiberletRoutes) {
        return std::make_shared<const FiberletRouteChunkPayload>(
            deserializeFiberletRoutes(bytes));
    }
    throw std::invalid_argument("unknown fiberlet storage chunk kind");
}

namespace
{

const FiberletStorageCodecConfig& payloadConfig(
    FiberletStorageChunkKind kind,
    const std::shared_ptr<const vc::render::DecodedChunkPayload>& payload)
{
    if (kind == FiberletStorageChunkKind::Anchors) {
        const auto typed = std::dynamic_pointer_cast<
            const FiberletAnchorChunkPayload>(payload);
        if (!typed)
            throw std::invalid_argument("fiberlet anchor chunk payload type is invalid");
        return typed->config;
    }
    if (kind == FiberletStorageChunkKind::FiberletPrefix) {
        const auto typed = std::dynamic_pointer_cast<
            const FiberletPrefixChunkPayload>(payload);
        if (!typed)
            throw std::invalid_argument("fiberlet prefix chunk payload type is invalid");
        return typed->config;
    }
    const auto typed = std::dynamic_pointer_cast<
        const FiberletRouteChunkPayload>(payload);
    if (!typed)
        throw std::invalid_argument("fiberlet route chunk payload type is invalid");
    return typed->config;
}

void requireMatchingCodec(
    const FiberletStorageCodecConfig& decoded,
    const FiberletStorageCodecConfig& expected)
{
    if (decoded.profile != expected.profile ||
        decoded.chunkZYX != expected.chunkZYX ||
        decoded.datasetFingerprint != expected.datasetFingerprint ||
        decoded.coordinateOriginZYX != expected.coordinateOriginZYX ||
        decoded.coordinateBits != expected.coordinateBits ||
        decoded.deltaBits != expected.deltaBits ||
        decoded.routeCountBits != expected.routeCountBits ||
        decoded.routeLatticeBits != expected.routeLatticeBits ||
        decoded.costBits != expected.costBits ||
        decoded.positionQuantumBaseVoxels !=
            expected.positionQuantumBaseVoxels ||
        decoded.predictionToBaseScale != expected.predictionToBaseScale) {
        throw std::invalid_argument(
            "fiberlet chunk header does not match its dataset metadata");
    }
}

void publishBytes(
    const std::filesystem::path& path,
    std::span<const std::byte> bytes)
{
    if (const auto existing = readBytes(path)) {
        if (existing->size() != bytes.size() ||
            !std::equal(existing->begin(), existing->end(), bytes.begin())) {
            throw std::invalid_argument(
                "fiberlet chunk publication conflicts with existing bytes");
        }
    } else {
        vc::core::util::atomicWriteBytes(path, bytes);
    }
}

void requireMatchingFiberletPair(
    const std::shared_ptr<const vc::render::DecodedChunkPayload>& prefix,
    const std::shared_ptr<const vc::render::DecodedChunkPayload>& routes)
{
    const auto prefixPayload = std::dynamic_pointer_cast<
        const FiberletPrefixChunkPayload>(prefix);
    const auto routePayload = std::dynamic_pointer_cast<
        const FiberletRouteChunkPayload>(routes);
    if (!prefixPayload || !routePayload)
        throw std::invalid_argument("fiberlet pair payload types are invalid");
    if (prefixPayload->prefixes.size() != routePayload->routes.size())
        throw std::invalid_argument(
            "fiberlet prefix and route payload record counts differ");
}

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
        for (const auto kind : datasetKinds(metadata.kind)) {
            const auto storedArray = json::parse(readText(arrayDirectory(root, kind) / ".zarray"));
            if (storedArray != arrayMetadata(metadata, kind))
                throw std::invalid_argument("fiberlet dataset .zarray metadata is invalid");
        }
        removeLegacyBookkeeping(root);
    } else {
        std::filesystem::create_directories(root);
        vc::core::util::atomicWriteString(root / ".zgroup", json{{"zarr_format", 2}}.dump(2) + "\n");
        vc::core::util::atomicWriteString(root / ".zattrs", expected.dump(2) + "\n");
        for (const auto kind : datasetKinds(metadata.kind)) {
            const auto directory = arrayDirectory(root, kind);
            std::filesystem::create_directories(directory);
            vc::core::util::atomicWriteString(directory / ".zarray", arrayMetadata(metadata, kind).dump(2) + "\n");
        }
        removeLegacyBookkeeping(root);
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

bool FiberletChunkDataset::datasetComplete() const
{
    if (metadata_.kind != FiberletDatasetKind::Combined)
        throw std::invalid_argument("whole-dataset completeness requires a combined dataset");
    if (!expectedChunksConfigured_)
        throw std::invalid_argument("combined dataset completeness requires source-derived expected chunks");
    for (const auto& key : expectedChunks_) {
        if (!readMaterializedChunk(FiberletStorageChunkKind::Anchors, key))
            return false;
    }
    return true;
}

void FiberletChunkDataset::configureExpectedChunks(std::span<const vc::render::ChunkKey> chunks)
{
    if (metadata_.kind != FiberletDatasetKind::Combined)
        throw std::invalid_argument("fiberlet expected chunks require a combined dataset");
    std::vector<vc::render::ChunkKey> normalized(chunks.begin(), chunks.end());
    for (auto& key : normalized) {
        key.level = 0;
        (void)codecConfig(FiberletStorageChunkKind::Anchors, key);
    }
    const auto less = [](const auto& left, const auto& right) {
        return std::tie(left.iz, left.iy, left.ix) < std::tie(right.iz, right.iy, right.ix);
    };
    std::sort(normalized.begin(), normalized.end(), less);
    normalized.erase(
        std::unique(
            normalized.begin(),
            normalized.end(),
            [](const auto& left, const auto& right) { return left.iz == right.iz && left.iy == right.iy && left.ix == right.ix; }),
        normalized.end());
    expectedChunks_ = std::move(normalized);
    expectedChunksConfigured_ = true;
}

const std::vector<vc::render::ChunkKey>& FiberletChunkDataset::expectedChunks() const noexcept
{
    return expectedChunks_;
}

bool FiberletChunkDataset::isExpectedChunk(const vc::render::ChunkKey& key) const
{
    const auto coordinate = std::tuple{key.iz, key.iy, key.ix};
    const auto found = std::lower_bound(expectedChunks_.begin(), expectedChunks_.end(), coordinate, [](const auto& candidate, const auto& value) {
        return std::tie(candidate.iz, candidate.iy, candidate.ix) < value;
    });
    return found != expectedChunks_.end() && std::tie(found->iz, found->iy, found->ix) == coordinate;
}

FiberletChunkDataset::MaterializationStats FiberletChunkDataset::materializationStats() const noexcept
{
    return {
        materializationDecodes_[0].load(std::memory_order_relaxed),
        materializationDecodes_[1].load(std::memory_order_relaxed),
        materializationDecodes_[2].load(std::memory_order_relaxed)};
}

FiberletStorageCodecConfig FiberletChunkDataset::codecConfig(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const
{
    if (!datasetContains(metadata_.kind, kind))
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
    auto materialized = readMaterializedChunk(kind, key);
    if (!materialized)
        return std::nullopt;
    return std::move(materialized->bytes);
}

std::optional<FiberletChunkDataset::MaterializedChunk> FiberletChunkDataset::readMaterializedChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key) const
{
    if (expectedChunksConfigured_ && !isExpectedChunk(key)) {
        const auto codec = codecConfig(kind, key);
        std::vector<std::byte> bytes;
        if (kind == FiberletStorageChunkKind::Anchors)
            bytes = serializeFiberletAnchors(codec, {});
        else if (kind == FiberletStorageChunkKind::FiberletPrefix)
            bytes = serializeFiberletPrefixes(codec, {});
        else
            bytes = serializeFiberletRoutes(codec, {});
        return MaterializedChunk{bytes, decodeFiberletChunkPayload(kind, bytes), true};
    }

    if (metadata_.kind == FiberletDatasetKind::Anchors) {
        auto bytes = readBytes(chunkPath(kind, key));
        if (!bytes)
            return std::nullopt;
        auto payload = decodeFiberletChunkPayload(kind, *bytes);
        requireMatchingCodec(payloadConfig(kind, payload), codecConfig(kind, key));
        materializationDecodes_[static_cast<std::size_t>(kind) - 1].fetch_add(1, std::memory_order_relaxed);
        return MaterializedChunk{std::move(*bytes), std::move(payload), true};
    }

    const vc::render::ChunkKey ownerKey{0, key.iz, key.iy, key.ix};
    const vc::render::ChunkKey routeKey{1, key.iz, key.iy, key.ix};
    auto prefixBytes = readBytes(chunkPath(FiberletStorageChunkKind::FiberletPrefix, ownerKey));
    auto routeBytes = readBytes(chunkPath(FiberletStorageChunkKind::FiberletRoutes, routeKey));
    std::optional<std::vector<std::byte>> anchorBytes;
    if (metadata_.kind == FiberletDatasetKind::Combined)
        anchorBytes = readBytes(chunkPath(FiberletStorageChunkKind::Anchors, ownerKey));
    if (!prefixBytes || !routeBytes || (metadata_.kind == FiberletDatasetKind::Combined && !anchorBytes))
        return std::nullopt;

    auto prefixPayload = decodeFiberletChunkPayload(FiberletStorageChunkKind::FiberletPrefix, *prefixBytes);
    auto routePayload = decodeFiberletChunkPayload(FiberletStorageChunkKind::FiberletRoutes, *routeBytes);
    requireMatchingCodec(
        payloadConfig(FiberletStorageChunkKind::FiberletPrefix, prefixPayload),
        codecConfig(FiberletStorageChunkKind::FiberletPrefix, ownerKey));
    requireMatchingCodec(
        payloadConfig(FiberletStorageChunkKind::FiberletRoutes, routePayload),
        codecConfig(FiberletStorageChunkKind::FiberletRoutes, routeKey));
    requireMatchingFiberletPair(prefixPayload, routePayload);
    materializationDecodes_[static_cast<std::size_t>(FiberletStorageChunkKind::FiberletPrefix) - 1].fetch_add(1, std::memory_order_relaxed);
    materializationDecodes_[static_cast<std::size_t>(FiberletStorageChunkKind::FiberletRoutes) - 1].fetch_add(1, std::memory_order_relaxed);

    std::shared_ptr<const vc::render::DecodedChunkPayload> anchorPayload;
    if (anchorBytes) {
        anchorPayload = decodeFiberletChunkPayload(FiberletStorageChunkKind::Anchors, *anchorBytes);
        requireMatchingCodec(
            payloadConfig(FiberletStorageChunkKind::Anchors, anchorPayload),
            codecConfig(FiberletStorageChunkKind::Anchors, ownerKey));
        materializationDecodes_[static_cast<std::size_t>(FiberletStorageChunkKind::Anchors) - 1].fetch_add(1, std::memory_order_relaxed);
    }

    if (kind == FiberletStorageChunkKind::Anchors)
        return MaterializedChunk{std::move(*anchorBytes), std::move(anchorPayload), true};
    if (kind == FiberletStorageChunkKind::FiberletPrefix)
        return MaterializedChunk{std::move(*prefixBytes), std::move(prefixPayload), true};
    return MaterializedChunk{std::move(*routeBytes), std::move(routePayload), true};
}

void FiberletChunkDataset::publishChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const
{
    MaterializedChunk chunk;
    chunk.bytes.assign(bytes.begin(), bytes.end());
    chunk.payload = decodeFiberletChunkPayload(kind, bytes);
    publishMaterializedChunk(kind, key, chunk);
}

void FiberletChunkDataset::publishMaterializedChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, const MaterializedChunk& chunk) const
{
    if (!chunk.payload)
        throw std::invalid_argument("fiberlet publication has no decoded payload");
    requireMatchingCodec(payloadConfig(kind, chunk.payload), codecConfig(kind, key));
    publishBytes(chunkPath(kind, key), chunk.bytes);
}

void FiberletChunkDataset::publishFiberletChunkPair(
    const vc::render::ChunkKey& prefixKey, const MaterializedChunk& prefix, const vc::render::ChunkKey& routeKey, const MaterializedChunk& routes) const
{
    if (metadata_.kind == FiberletDatasetKind::Anchors)
        throw std::invalid_argument("fiberlet pair publication requires a fiberlet dataset");
    if (prefixKey.iz != routeKey.iz || prefixKey.iy != routeKey.iy || prefixKey.ix != routeKey.ix)
        throw std::invalid_argument("fiberlet prefix and route chunks have different coordinates");
    requireMatchingCodec(payloadConfig(FiberletStorageChunkKind::FiberletPrefix, prefix.payload), codecConfig(FiberletStorageChunkKind::FiberletPrefix, prefixKey));
    requireMatchingCodec(payloadConfig(FiberletStorageChunkKind::FiberletRoutes, routes.payload), codecConfig(FiberletStorageChunkKind::FiberletRoutes, routeKey));
    requireMatchingFiberletPair(prefix.payload, routes.payload);
    if (metadata_.kind == FiberletDatasetKind::Combined) {
        const auto anchors = readBytes(chunkPath(FiberletStorageChunkKind::Anchors, prefixKey));
        if (!anchors) {
            throw std::invalid_argument("combined fiberlet publication requires its anchor payload");
        }
        const auto anchorPayload = decodeFiberletChunkPayload(FiberletStorageChunkKind::Anchors, *anchors);
        requireMatchingCodec(payloadConfig(FiberletStorageChunkKind::Anchors, anchorPayload), codecConfig(FiberletStorageChunkKind::Anchors, prefixKey));
    }
    publishBytes(chunkPath(FiberletStorageChunkKind::FiberletPrefix, prefixKey), prefix.bytes);
    publishBytes(chunkPath(FiberletStorageChunkKind::FiberletRoutes, routeKey), routes.bytes);
}

void FiberletChunkDataset::validateChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const
{
    const auto payload = decodeFiberletChunkPayload(kind, bytes);
    requireMatchingCodec(payloadConfig(kind, payload), codecConfig(kind, key));
}

std::shared_ptr<vc::render::ChunkCache> createGeneratedFiberletChunkCache(
    std::shared_ptr<FiberletChunkDataset> dataset,
    FiberletChunkGenerator generator,
    vc::render::ChunkCache::Options options,
    FiberletChunkResolvedCallback resolved)
{
    if (!dataset || !generator)
        throw std::invalid_argument("generated fiberlet cache requires a dataset and generator");
    const auto shape = dataset->metadata().chunkGridShapeZYX;
    std::vector<vc::render::ChunkCache::LevelInfo> levels;
    std::vector<std::shared_ptr<vc::render::IChunkFetcher>> fetchers;
    if (dataset->metadata().kind == FiberletDatasetKind::Anchors) {
        levels.push_back({shape, {1, 1, 1}, {}});
        fetchers.push_back(std::make_shared<GeneratedFetcher>(dataset, FiberletStorageChunkKind::Anchors, generator, resolved));
    } else {
        levels.push_back({shape, {1, 1, 1}, {}});
        levels.push_back({shape, {1, 1, 1}, {}});
        fetchers.push_back(std::make_shared<GeneratedFetcher>(dataset, FiberletStorageChunkKind::FiberletPrefix, generator, resolved));
        fetchers.push_back(std::make_shared<GeneratedFetcher>(dataset, FiberletStorageChunkKind::FiberletRoutes, generator, resolved));
    }
    options.detectAllFillChunks = false;
    options.persistentCachePath.reset();
    return std::make_shared<vc::render::ChunkCache>(std::move(levels), std::move(fetchers), 0.0, vc::render::ChunkDtype::Opaque, std::move(options));
}

std::shared_ptr<vc::render::ChunkCache> createStoredFiberletAnchorChunkCache(std::shared_ptr<FiberletChunkDataset> dataset, vc::render::ChunkCache::Options options)
{
    if (!dataset || (dataset->metadata().kind != FiberletDatasetKind::Anchors && dataset->metadata().kind != FiberletDatasetKind::Combined)) {
        throw std::invalid_argument("stored fiberlet anchor cache requires anchor payloads");
    }
    if (dataset->metadata().kind == FiberletDatasetKind::Combined && !dataset->datasetComplete())
        throw std::invalid_argument("stored combined fiberlet cache requires a complete dataset");
    const auto shape = dataset->metadata().chunkGridShapeZYX;
    options.detectAllFillChunks = false;
    options.persistentCachePath.reset();
    return std::make_shared<
        vc::render::ChunkCache>(std::vector<vc::render::ChunkCache::LevelInfo>{{shape, {1, 1, 1}, {}}}, std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{std::make_shared<StoredFetcher>(dataset, FiberletStorageChunkKind::Anchors)}, 0.0, vc::render::ChunkDtype::Opaque, std::move(options));
}

std::shared_ptr<vc::render::ChunkCache> createStoredFiberletPathChunkCache(std::shared_ptr<FiberletChunkDataset> dataset, vc::render::ChunkCache::Options options)
{
    if (!dataset || (dataset->metadata().kind != FiberletDatasetKind::Fiberlets && dataset->metadata().kind != FiberletDatasetKind::Combined)) {
        throw std::invalid_argument("stored fiberlet path cache requires prefix and route payloads");
    }
    if (dataset->metadata().kind == FiberletDatasetKind::Combined && !dataset->datasetComplete())
        throw std::invalid_argument("stored combined fiberlet cache requires a complete dataset");
    const auto shape = dataset->metadata().chunkGridShapeZYX;
    options.detectAllFillChunks = false;
    options.persistentCachePath.reset();
    return std::make_shared<
        vc::render::ChunkCache>(std::vector<vc::render::ChunkCache::LevelInfo>{{shape, {1, 1, 1}, {}}, {shape, {1, 1, 1}, {}}}, std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{std::make_shared<StoredFetcher>(dataset, FiberletStorageChunkKind::FiberletPrefix), std::make_shared<StoredFetcher>(dataset, FiberletStorageChunkKind::FiberletRoutes)}, 0.0, vc::render::ChunkDtype::Opaque, std::move(options));
}

}  // namespace vc::fiber_tracer
