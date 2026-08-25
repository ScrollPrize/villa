#include "vc/fiber_tracer/FiberletDataset.hpp"

#include "vc/core/util/AtomicFile.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "utils/thread_pool.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace vc::fiber_tracer
{
namespace
{

using json = nlohmann::json;

bool nearlyEqual(double left, double right)
{
    return std::abs(left - right) <=
           1e-12 * std::max({1.0, std::abs(left), std::abs(right)});
}

std::array<std::size_t, 3> fiberletPredictionShape(
    const FiberletDatasetMetadata& metadata)
{
    try {
        const auto& grid = metadata.processing.at("grid");
        if (!grid.is_object() ||
            grid.at("coordinate_order") != "zyx_storage_xyz_vectors") {
            throw std::invalid_argument(
                "fiberlet grid coordinate frame is incompatible with Lasagna base coordinates");
        }
        const double structuredScale = grid.at("prediction_to_base").get<double>();
        if (!(structuredScale > 0.0) || !std::isfinite(structuredScale) ||
            !nearlyEqual(structuredScale, metadata.predictionToBaseScale)) {
            throw std::invalid_argument(
                "fiberlet prediction-to-base scale metadata is inconsistent");
        }
        const auto& shape = grid.at("shape_zyx");
        if (!shape.is_array() || shape.size() != 3)
            throw std::invalid_argument("fiberlet prediction grid shape is invalid");
        std::array<std::size_t, 3> result{};
        for (std::size_t axis = 0; axis < result.size(); ++axis) {
            if (!shape.at(axis).is_number_unsigned())
                throw std::invalid_argument("fiberlet prediction grid shape is invalid");
            result[axis] = shape.at(axis).get<std::size_t>();
            if (result[axis] == 0)
                throw std::invalid_argument("fiberlet prediction grid shape is invalid");
        }
        return result;
    } catch (const nlohmann::json::exception&) {
        throw std::invalid_argument(
            "fiberlet dataset is missing valid structured grid metadata");
    }
}

double requiredPositiveManifestNumber(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    const char* name)
{
    const auto found = manifest.raw.find(name);
    if (found == manifest.raw.end() || !found->is_number())
        throw std::invalid_argument(
            std::string("normal manifest is missing numeric field '") + name + "'");
    const double value = found->get<double>();
    if (!(value > 0.0) || !std::isfinite(value))
        throw std::invalid_argument(
            std::string("normal manifest field '") + name + "' must be positive and finite");
    return value;
}

void validateNormalBinding(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    const vc::lasagna::LasagnaChannelBinding& binding,
    std::string_view channel)
{
    if (binding.group == nullptr || binding.group->channels.size() != 1)
        throw std::invalid_argument(
            "normal channel '" + std::string(channel) +
            "' must use its own 3D Lasagna group");
    const double baseSpacing = static_cast<double>(binding.group->scaleFactor()) *
                               manifest.sourceToBase;
    if (!vc::lasagna::lasagnaChannelShapeCompatible(
            *manifest.baseShapeZYX, baseSpacing,
            binding.shapeZYX, binding.chunksZYX)) {
        throw std::invalid_argument(
            "normal channel '" + std::string(channel) +
            "' shape is incompatible with base_shape_zyx and scale");
    }
}

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

std::string stringFingerprint(std::string_view value)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    constexpr char digits[] = "0123456789abcdef";
    std::string result = "fnv1a64:";
    for (int shift = 60; shift >= 0; shift -= 4)
        result.push_back(digits[(hash >> shift) & 0x0f]);
    return result;
}

std::array<std::uint8_t, 32> datasetFingerprint(std::string_view value)
{
    std::array<std::uint8_t, 32> result{};
    for (std::size_t lane = 0; lane < 4; ++lane) {
        std::uint64_t hash = 14695981039346656037ULL ^
            (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(lane + 1));
        for (const unsigned char byte : value) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        for (std::size_t byte = 0; byte < 8; ++byte)
            result[lane * 8 + byte] =
                static_cast<std::uint8_t>(hash >> (byte * 8));
    }
    return result;
}

json algorithmIdentityJson(const FiberletDatasetMetadata& metadata)
{
    return {
        {"identity_version", 2},
        {"dataset_kind", kindName(metadata.kind)},
        {"encoding_profile", profileName(metadata.profile)},
        {"chunk_grid_shape_zyx", metadata.chunkGridShapeZYX},
        {"coordinate_origin_zyx", metadata.coordinateOriginZYX},
        {"coordinate_units_per_chunk_zyx", metadata.coordinateUnitsPerChunkZYX},
        {"maximum_endpoint_reach_coordinate_units_zyx", metadata.maximumEndpointReachCoordinateUnitsZYX},
        {"spatial_chunk_side_base", metadata.spatialChunkSideBaseVoxels},
        {"coordinate_bits", metadata.coordinateBits},
        {"delta_bits", metadata.deltaBits},
        {"route_count_bits", metadata.routeCountBits},
        {"route_lattice_bits", metadata.routeLatticeBits},
        {"cost_bits", metadata.costBits},
        {"position_quantum_base", metadata.positionQuantumBaseVoxels == 0
             ? json(nullptr)
             : json(metadata.positionQuantumBaseVoxels)},
        {"prediction_to_base", metadata.predictionToBaseScale},
        {"processing", metadata.processing},
    };
}

void validateMetadata(const FiberletDatasetMetadata& metadata)
{
    if (!metadata.sources.is_object() || !metadata.processing.is_object())
        throw std::invalid_argument(
            "fiberlet sources and processing metadata must be JSON objects");
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (metadata.chunkGridShapeZYX[axis] <= 0 ||
            metadata.coordinateUnitsPerChunkZYX[axis] <= 0 ||
            metadata.maximumEndpointReachCoordinateUnitsZYX[axis] < 0) {
            throw std::invalid_argument(
                "fiberlet dataset grid or coordinate chunk size is invalid");
        }
    }
    auto canonical = metadata;
    canonical.algorithmFingerprint = stringFingerprint(
        algorithmIdentityJson(canonical).dump());
    canonical.datasetFingerprint = datasetFingerprint(json{
        {"algorithm", algorithmIdentityJson(canonical)},
        {"sources", canonical.sources},
    }.dump());
    if (metadata.algorithmFingerprint != canonical.algorithmFingerprint ||
        metadata.datasetFingerprint != canonical.datasetFingerprint) {
        throw std::invalid_argument(
            "fiberlet metadata fingerprints do not match its structured identity");
    }
}

json metadataJson(const FiberletDatasetMetadata& metadata)
{
    return {
        {"vc_format", "fiberlet_dataset"},
        {"format_version", 2},
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
        {"sources", metadata.sources},
        {"processing", metadata.processing},
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
        "sources",
        "processing",
        "build_state"};
    if (!value.is_object() || value.size() != required.size())
        throw std::invalid_argument("fiberlet dataset metadata has unknown or missing fields");
    for (const auto& name : required) {
        if (!value.contains(name))
            throw std::invalid_argument("fiberlet dataset metadata is missing " + name);
    }
    if (value.at("vc_format") != "fiberlet_dataset" ||
        value.at("format_version") != 2 ||
        value.at("build_state") != "partial")
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
    result.sources = value.at("sources");
    result.processing = value.at("processing");
    validateMetadata(result);
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
                if (dataset_->metadata().kind != FiberletDatasetKind::Anchors) {
                    const vc::render::ChunkKey owner{
                        0, key.iz, key.iy, key.ix};
                    const vc::render::ChunkKey route{
                        1, key.iz, key.iy, key.ix};
                    std::size_t present = 0;
                    std::size_t required = 2;
                    present += std::filesystem::exists(dataset_->chunkPath(
                        FiberletStorageChunkKind::FiberletPrefix, owner));
                    present += std::filesystem::exists(dataset_->chunkPath(
                        FiberletStorageChunkKind::FiberletRoutes, route));
                    if (dataset_->metadata().kind == FiberletDatasetKind::Combined) {
                        ++required;
                        present += std::filesystem::exists(dataset_->chunkPath(
                            FiberletStorageChunkKind::Anchors, owner));
                    }
                    if (present != 0 && present != required) {
                        result.status = vc::render::ChunkFetchStatus::DecodeError;
                        result.message =
                            "fiberlet sparse chunk tuple is only partially present";
                        return result;
                    }
                }
                const auto codec = dataset_->codecConfig(kind_, key);
                std::vector<std::byte> bytes;
                if (kind_ == FiberletStorageChunkKind::Anchors)
                    bytes = serializeFiberletAnchors(codec, {});
                else if (kind_ == FiberletStorageChunkKind::FiberletPrefix)
                    bytes = serializeFiberletPrefixes(codec, {});
                else
                    bytes = serializeFiberletRoutes(codec, {});
                result.status = vc::render::ChunkFetchStatus::Found;
                result.payload = decodeFiberletChunkPayload(kind_, bytes);
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

void requireOverlayCompatible(
    const FiberletDatasetMetadata& layer,
    const FiberletDatasetMetadata& lower,
    FiberletDatasetKind expectedKind)
{
    if (layer.kind != expectedKind || lower.kind != expectedKind ||
        layer.profile != lower.profile ||
        layer.chunkGridShapeZYX != lower.chunkGridShapeZYX ||
        layer.coordinateOriginZYX != lower.coordinateOriginZYX ||
        layer.coordinateUnitsPerChunkZYX !=
            lower.coordinateUnitsPerChunkZYX ||
        layer.maximumEndpointReachCoordinateUnitsZYX !=
            lower.maximumEndpointReachCoordinateUnitsZYX ||
        layer.spatialChunkSideBaseVoxels !=
            lower.spatialChunkSideBaseVoxels ||
        layer.coordinateBits != lower.coordinateBits ||
        layer.deltaBits != lower.deltaBits ||
        layer.routeCountBits != lower.routeCountBits ||
        layer.routeLatticeBits != lower.routeLatticeBits ||
        layer.costBits != lower.costBits ||
        layer.positionQuantumBaseVoxels !=
            lower.positionQuantumBaseVoxels ||
        layer.predictionToBaseScale != lower.predictionToBaseScale ||
        layer.sources != lower.sources) {
        throw std::invalid_argument(
            "fiberlet overlay datasets have incompatible layouts");
    }
}

FiberletChunkDataset::MaterializedChunk lowerOverlayChunk(
    const std::shared_ptr<vc::render::ChunkCache>& lower,
    FiberletStorageChunkKind kind,
    const vc::render::ChunkKey& requested)
{
    const auto result = lower->getChunkBlocking(
        requested.level, requested.iz, requested.iy, requested.ix);
    if (result.status != vc::render::ChunkStatus::Data || !result.payload) {
        throw std::runtime_error(
            "lower fiberlet overlay chunk is unavailable");
    }
    if (kind == FiberletStorageChunkKind::Anchors &&
        !std::dynamic_pointer_cast<const FiberletAnchorChunkPayload>(
            result.payload)) {
        throw std::runtime_error(
            "lower fiberlet overlay anchor payload has the wrong type");
    }
    if (kind == FiberletStorageChunkKind::FiberletPrefix &&
        !std::dynamic_pointer_cast<const FiberletPrefixChunkPayload>(
            result.payload)) {
        throw std::runtime_error(
            "lower fiberlet overlay prefix payload has the wrong type");
    }
    if (kind == FiberletStorageChunkKind::FiberletRoutes &&
        !std::dynamic_pointer_cast<const FiberletRouteChunkPayload>(
            result.payload)) {
        throw std::runtime_error(
            "lower fiberlet overlay route payload has the wrong type");
    }
    return {{}, result.payload, true};
}

bool sameFloat(float left, float right)
{
    return std::bit_cast<std::uint32_t>(left) ==
        std::bit_cast<std::uint32_t>(right);
}

bool sameVec(const cv::Vec3f& left, const cv::Vec3f& right)
{
    return sameFloat(left[0], right[0]) &&
        sameFloat(left[1], right[1]) &&
        sameFloat(left[2], right[2]);
}

bool sameAnchor(
    const FiberletStoredAnchor& left, const FiberletStoredAnchor& right)
{
    return left.key == right.key &&
        sameVec(left.positionPredictionXYZ, right.positionPredictionXYZ) &&
        sameVec(left.fittedAxisXYZ, right.fittedAxisXYZ) &&
        sameVec(left.predictionAxisXYZ, right.predictionAxisXYZ) &&
        sameFloat(left.predictionPresence, right.predictionPresence) &&
        sameVec(left.normalXYZ, right.normalXYZ) &&
        left.predictionValid == right.predictionValid &&
        left.predictionPresenceValid == right.predictionPresenceValid &&
        left.normalValid == right.normalValid;
}

bool sameCost(
    const FiberletStoredPathCost& left,
    const FiberletStoredPathCost& right)
{
    return sameFloat(left.invalidPrediction, right.invalidPrediction) &&
        sameFloat(left.alignment, right.alignment) &&
        sameFloat(left.isotropicSmoothness, right.isotropicSmoothness) &&
        sameFloat(left.tangentSmoothness, right.tangentSmoothness) &&
        sameFloat(left.normalSmoothness, right.normalSmoothness);
}

bool samePrefix(
    const FiberletStoredPrefix& left, const FiberletStoredPrefix& right)
{
    return left.id == right.id &&
        left.interiorPointCount == right.interiorPointCount &&
        left.entryUV == right.entryUV && left.exitUV == right.exitUV &&
        sameFloat(left.pathLengthPredictionVoxels,
                  right.pathLengthPredictionVoxels) &&
        sameCost(left.cost, right.cost) &&
        sameVec(left.firstStepBaseXYZ, right.firstStepBaseXYZ) &&
        sameVec(left.lastStepBaseXYZ, right.lastStepBaseXYZ);
}

bool sameRoute(
    const FiberletStoredRoute& left, const FiberletStoredRoute& right)
{
    if (left.middleUV != right.middleUV ||
        left.segmentCostDensities.size() !=
            right.segmentCostDensities.size()) {
        return false;
    }
    for (std::size_t index = 0;
         index < left.segmentCostDensities.size(); ++index) {
        if (!sameFloat(
                left.segmentCostDensities[index],
                right.segmentCostDensities[index])) {
            return false;
        }
    }
    return true;
}

template <typename T, typename Id, typename GetId, typename Same>
void requireMonotoneSubset(
    std::span<const T> current,
    std::span<const T> replacement,
    GetId getId,
    Same same,
    std::string_view description)
{
    std::size_t currentIndex = 0;
    for (const auto& value : replacement) {
        const auto id = getId(value);
        while (currentIndex < current.size() &&
               getId(current[currentIndex]) < id) {
            ++currentIndex;
        }
        if (currentIndex == current.size() ||
            getId(current[currentIndex]) != id) {
            throw std::invalid_argument(
                std::string("fiberlet overlay replacement restores a ") +
                std::string(description));
        }
        if (!same(current[currentIndex], value)) {
            throw std::invalid_argument(
                std::string("fiberlet overlay replacement mutates a ") +
                std::string(description));
        }
        ++currentIndex;
    }
}

}  // namespace

struct FiberletChunkWriteBackCache::Impl {
    struct OwnerKey {
        std::uint64_t layer = 0;
        int z = 0;
        int y = 0;
        int x = 0;

        auto operator<=>(const OwnerKey&) const = default;
    };

    struct Layer {
        std::filesystem::path root;
        FiberletDatasetKind kind = FiberletDatasetKind::Anchors;
    };

    struct Buffers {
        std::shared_ptr<const std::vector<std::byte>> anchors;
        std::shared_ptr<const std::vector<std::byte>> prefixes;
        std::shared_ptr<const std::vector<std::byte>> routes;
    };

    struct Resident {
        std::uint64_t generation = 0;
        std::uint64_t touch = 0;
        std::size_t chargedBytes = 0;
        Buffers buffers;
    };

    struct Pending {
        OwnerKey key;
        std::uint64_t generation = 0;
        std::size_t chargedBytes = 0;
        Buffers buffers;
        std::filesystem::path anchorPath;
        std::filesystem::path prefixPath;
        std::filesystem::path routePath;
        mutable std::mutex mutex;
        std::condition_variable ready;
        bool done = false;
        std::exception_ptr error;
    };

    static constexpr std::size_t kEntryOverheadBytes = 256;

    explicit Impl(FiberletChunkWriteBackCache::Options input)
        : options(std::move(input))
        , maximumBytes(options.maximumBytes)
        , decodedBudget(options.decodedBudget)
        , decodedBudgetMaximum(
              decodedBudget ? decodedBudget->maximumBytes() : 0)
        , writers(std::max<std::size_t>(1, options.writerThreads))
    {
        if (maximumBytes == 0)
            throw std::invalid_argument(
                "fiberlet write-back cache capacity must be positive");
        if (!options.writeBytes) {
            options.writeBytes = [](const std::filesystem::path& path,
                                    std::span<const std::byte> bytes) {
                vc::core::util::atomicWriteBytes(path, bytes);
            };
        }
    }

    ~Impl() = default;

    static OwnerKey ownerKey(
        std::uint64_t layer, const vc::render::ChunkKey& key)
    {
        return {layer, key.iz, key.iy, key.ix};
    }

    static std::size_t bufferBytes(const Buffers& buffers)
    {
        std::size_t result = kEntryOverheadBytes;
        for (const auto* value :
             {&buffers.anchors, &buffers.prefixes, &buffers.routes}) {
            if (*value)
                result += (*value)->capacity();
        }
        return result;
    }

    const Layer& layer(std::uint64_t id) const
    {
        if (id == 0 || id > layers.size())
            throw std::invalid_argument("fiberlet write-back layer is invalid");
        return layers[id - 1];
    }

    std::filesystem::path path(
        const OwnerKey& key, FiberletStorageChunkKind kind) const
    {
        const auto& selected = layer(key.layer);
        return arrayDirectory(selected.root, kind) /
            (std::to_string(key.z) + "." + std::to_string(key.y) + "." +
             std::to_string(key.x));
    }

    void adjustDecodedBudgetLocked()
    {
        if (!decodedBudget)
            return;
        const std::size_t allowance = liveBytes < decodedBudgetMaximum
            ? decodedBudgetMaximum - liveBytes
            : 0;
        decodedBudget->setMaximumBytes(allowance);
    }

    void throwFirstFailureLocked() const
    {
        if (!failures.empty())
            std::rethrow_exception(failures.begin()->second);
    }

    std::shared_ptr<Pending> oldestPendingLocked() const
    {
        if (pending.empty())
            return {};
        return pending.begin()->second;
    }

    std::map<OwnerKey, Resident>::iterator oldestResidentLocked()
    {
        auto oldest = residents.end();
        for (auto iterator = residents.begin(); iterator != residents.end();
             ++iterator) {
            if (oldest == residents.end() ||
                std::tie(iterator->second.touch, iterator->first) <
                    std::tie(oldest->second.touch, oldest->first)) {
                oldest = iterator;
            }
        }
        return oldest;
    }

    void complete(
        const std::shared_ptr<Pending>& state, std::exception_ptr error)
    {
        {
            std::lock_guard lock(mutex);
            const auto found = pending.find(state->key);
            if (found != pending.end() && found->second == state)
                pending.erase(found);
            if (error)
                failures.try_emplace(state->key, error);
            liveBytes = liveBytes > state->chargedBytes
                ? liveBytes - state->chargedBytes
                : 0;
            adjustDecodedBudgetLocked();
        }
        {
            std::lock_guard lock(state->mutex);
            state->error = error;
            state->done = true;
        }
        state->ready.notify_all();
    }

    std::shared_ptr<Pending> spillLocked(
        std::map<OwnerKey, Resident>::iterator selected)
    {
        auto state = std::make_shared<Pending>();
        state->key = selected->first;
        state->generation = selected->second.generation;
        state->chargedBytes = selected->second.chargedBytes;
        state->buffers = std::move(selected->second.buffers);
        if (state->buffers.anchors) {
            state->anchorPath = path(
                state->key, FiberletStorageChunkKind::Anchors);
        } else {
            state->prefixPath = path(
                state->key, FiberletStorageChunkKind::FiberletPrefix);
            state->routePath = path(
                state->key, FiberletStorageChunkKind::FiberletRoutes);
        }
        pending.emplace(state->key, state);
        residents.erase(selected);
        ++statistics.spills;
        statistics.spilledBytes += state->chargedBytes;
        writers.enqueue([this, state] {
            std::exception_ptr error;
            try {
                if (state->buffers.anchors) {
                    options.writeBytes(
                        state->anchorPath, *state->buffers.anchors);
                } else {
                    try {
                        options.writeBytes(
                            state->prefixPath, *state->buffers.prefixes);
                        options.writeBytes(
                            state->routePath, *state->buffers.routes);
                    } catch (...) {
                        std::error_code ignored;
                        std::filesystem::remove(state->prefixPath, ignored);
                        std::filesystem::remove(state->routePath, ignored);
                        throw;
                    }
                }
            } catch (...) {
                error = std::current_exception();
            }
            complete(state, error);
        });
        return state;
    }

    static void wait(const std::shared_ptr<Pending>& state)
    {
        std::unique_lock lock(state->mutex);
        state->ready.wait(lock, [&] { return state->done; });
        if (state->error)
            std::rethrow_exception(state->error);
    }

    void waitUntilReplaceable(const OwnerKey& key)
    {
        while (true) {
            std::shared_ptr<Pending> state;
            {
                std::lock_guard lock(mutex);
                throwFirstFailureLocked();
                const auto found = pending.find(key);
                if (found == pending.end())
                    return;
                state = found->second;
            }
            wait(state);
        }
    }

    void enforceBudget()
    {
        std::unique_lock pressure(pressureMutex);
        while (true) {
            std::shared_ptr<Pending> waitFor;
            {
                std::lock_guard lock(mutex);
                throwFirstFailureLocked();
                if (liveBytes <= maximumBytes)
                    return;

                // Once pressure starts, queue a deterministic batch rather
                // than synchronously spilling only the minimum victim. The
                // queued buffers remain charged, so the producer still waits
                // for actual releases before exceeding the hard bound, while
                // the remaining writes can overlap subsequent computation.
                std::size_t residentBytes = 0;
                for (const auto& [key, resident] : residents) {
                    (void)key;
                    residentBytes += resident.chargedBytes;
                }
                const std::size_t lowWater = maximumBytes * 3 / 4;
                while (residentBytes > lowWater && !residents.empty()) {
                    auto selected = oldestResidentLocked();
                    residentBytes -= selected->second.chargedBytes;
                    spillLocked(selected);
                }
                waitFor = oldestPendingLocked();
            }
            if (!waitFor)
                throw std::runtime_error(
                    "fiberlet write-back cache cannot satisfy its byte budget");
            wait(waitFor);
        }
    }

    void replace(const OwnerKey& key, Buffers buffers)
    {
        waitUntilReplaceable(key);
        const std::size_t charged = bufferBytes(buffers);
        {
            std::lock_guard lock(mutex);
            if (finished)
                throw std::logic_error("fiberlet write-back cache is finished");
            throwFirstFailureLocked();
            if (const auto found = residents.find(key);
                found != residents.end()) {
                liveBytes -= found->second.chargedBytes;
                residents.erase(found);
            }
            const std::uint64_t generation = ++generations[key];
            residents.emplace(
                key, Resident{generation, ++nextTouch, charged,
                              std::move(buffers)});
            liveBytes += charged;
            statistics.peakLiveBytes =
                std::max(statistics.peakLiveBytes, liveBytes);
            adjustDecodedBudgetLocked();
        }
        enforceBudget();
    }

    mutable std::mutex mutex;
    std::mutex pressureMutex;
    FiberletChunkWriteBackCache::Options options;
    std::size_t maximumBytes = 0;
    std::shared_ptr<vc::render::DecodedChunkCacheBudget> decodedBudget;
    std::size_t decodedBudgetMaximum = 0;
    utils::ThreadPool writers;
    std::vector<Layer> layers;
    std::map<OwnerKey, Resident> residents;
    std::map<OwnerKey, std::shared_ptr<Pending>> pending;
    std::map<OwnerKey, std::uint64_t> generations;
    std::map<OwnerKey, std::exception_ptr> failures;
    std::uint64_t nextTouch = 0;
    std::size_t liveBytes = 0;
    FiberletChunkWriteBackCache::Stats statistics;
    bool finished = false;
};

std::shared_ptr<FiberletChunkWriteBackCache>
FiberletChunkWriteBackCache::create(Options options)
{
    return std::shared_ptr<FiberletChunkWriteBackCache>(
        new FiberletChunkWriteBackCache(std::move(options)));
}

FiberletChunkWriteBackCache::FiberletChunkWriteBackCache(Options options)
    : impl_(std::make_unique<Impl>(std::move(options)))
{
}

FiberletChunkWriteBackCache::~FiberletChunkWriteBackCache()
{
    try {
        finish();
    } catch (...) {
    }
}

std::uint64_t FiberletChunkWriteBackCache::registerLayer(
    const std::filesystem::path& root, FiberletDatasetKind kind)
{
    if (kind == FiberletDatasetKind::Combined)
        throw std::invalid_argument(
            "fiberlet write-back layers cannot be combined datasets");
    std::lock_guard lock(impl_->mutex);
    if (impl_->finished)
        throw std::logic_error("fiberlet write-back cache is finished");
    impl_->layers.push_back({root.lexically_normal(), kind});
    return impl_->layers.size();
}

std::optional<std::shared_ptr<const std::vector<std::byte>>>
FiberletChunkWriteBackCache::read(
    std::uint64_t layer,
    FiberletStorageChunkKind kind,
    const vc::render::ChunkKey& key) const
{
    const auto owner = Impl::ownerKey(layer, key);
    std::lock_guard lock(impl_->mutex);
    impl_->throwFirstFailureLocked();
    const auto select = [&](const Impl::Buffers& buffers) {
        if (kind == FiberletStorageChunkKind::Anchors)
            return buffers.anchors;
        if (kind == FiberletStorageChunkKind::FiberletPrefix)
            return buffers.prefixes;
        return buffers.routes;
    };
    if (auto found = impl_->residents.find(owner);
        found != impl_->residents.end()) {
        found->second.touch = ++impl_->nextTouch;
        ++impl_->statistics.memoryHits;
        return select(found->second.buffers);
    }
    if (const auto found = impl_->pending.find(owner);
        found != impl_->pending.end()) {
        ++impl_->statistics.memoryHits;
        return select(found->second->buffers);
    }
    return std::nullopt;
}

void FiberletChunkWriteBackCache::replaceAnchor(
    std::uint64_t layer,
    const vc::render::ChunkKey& key,
    std::span<const std::byte> bytes)
{
    if (impl_->layer(layer).kind != FiberletDatasetKind::Anchors)
        throw std::invalid_argument(
            "fiberlet write-back anchor layer has the wrong kind");
    Impl::Buffers buffers;
    buffers.anchors =
        std::make_shared<const std::vector<std::byte>>(bytes.begin(), bytes.end());
    impl_->replace(Impl::ownerKey(layer, key), std::move(buffers));
}

void FiberletChunkWriteBackCache::replacePair(
    std::uint64_t layer,
    const vc::render::ChunkKey& prefixKey,
    std::span<const std::byte> prefix,
    const vc::render::ChunkKey& routeKey,
    std::span<const std::byte> routes)
{
    if (impl_->layer(layer).kind != FiberletDatasetKind::Fiberlets ||
        prefixKey.level != 0 || routeKey.level != 1 ||
        prefixKey.iz != routeKey.iz || prefixKey.iy != routeKey.iy ||
        prefixKey.ix != routeKey.ix) {
        throw std::invalid_argument(
            "fiberlet write-back pair layer or keys are invalid");
    }
    Impl::Buffers buffers;
    buffers.prefixes = std::make_shared<const std::vector<std::byte>>(
        prefix.begin(), prefix.end());
    buffers.routes = std::make_shared<const std::vector<std::byte>>(
        routes.begin(), routes.end());
    impl_->replace(Impl::ownerKey(layer, prefixKey), std::move(buffers));
}

FiberletChunkDataset::PairPresence FiberletChunkWriteBackCache::pairPresence(
    std::uint64_t layer, const vc::render::ChunkKey& owner) const
{
    const auto key = Impl::ownerKey(layer, owner);
    std::lock_guard lock(impl_->mutex);
    impl_->throwFirstFailureLocked();
    if (const auto found = impl_->residents.find(key);
        found != impl_->residents.end()) {
        return found->second.buffers.prefixes && found->second.buffers.routes
            ? FiberletChunkDataset::PairPresence::Complete
            : FiberletChunkDataset::PairPresence::Partial;
    }
    if (const auto found = impl_->pending.find(key);
        found != impl_->pending.end()) {
        return found->second->buffers.prefixes && found->second->buffers.routes
            ? FiberletChunkDataset::PairPresence::Complete
            : FiberletChunkDataset::PairPresence::Partial;
    }
    return FiberletChunkDataset::PairPresence::Absent;
}

FiberletChunkWriteBackCache::Stats FiberletChunkWriteBackCache::stats() const
{
    std::lock_guard lock(impl_->mutex);
    auto result = impl_->statistics;
    result.residentEntries = impl_->residents.size();
    result.pendingEntries = impl_->pending.size();
    result.liveBytes = impl_->liveBytes;
    return result;
}

void FiberletChunkWriteBackCache::waitForSpills()
{
    impl_->writers.wait_idle();
    std::lock_guard lock(impl_->mutex);
    impl_->throwFirstFailureLocked();
}

void FiberletChunkWriteBackCache::finish()
{
    if (!impl_)
        return;
    std::exception_ptr failure;
    try {
        waitForSpills();
    } catch (...) {
        failure = std::current_exception();
    }
    {
        std::lock_guard lock(impl_->mutex);
        if (!impl_->finished) {
            impl_->finished = true;
            impl_->residents.clear();
            impl_->pending.clear();
            impl_->liveBytes = 0;
            if (impl_->decodedBudget) {
                impl_->decodedBudget->setMaximumBytes(
                    impl_->decodedBudgetMaximum);
            }
        }
    }
    if (failure)
        std::rethrow_exception(failure);
}

std::vector<FiberletChunkWriteBackCache::LogicalFile>
FiberletChunkWriteBackCache::logicalFiles(
    const std::filesystem::path& root) const
{
    const auto normalized = root.lexically_normal();
    const auto inside = [&](const std::filesystem::path& candidate) {
        const auto relative = candidate.lexically_normal().lexically_relative(
            normalized);
        return !relative.empty() && *relative.begin() != "..";
    };
    std::vector<LogicalFile> result;
    std::lock_guard lock(impl_->mutex);
    impl_->throwFirstFailureLocked();
    const auto append = [&](const Impl::OwnerKey& key,
                            const Impl::Buffers& buffers) {
        const auto add = [&](FiberletStorageChunkKind kind,
                             const auto& bytes) {
            if (!bytes)
                return;
            auto selected = impl_->path(key, kind);
            if (inside(selected))
                result.push_back({std::move(selected), bytes});
        };
        add(FiberletStorageChunkKind::Anchors, buffers.anchors);
        add(FiberletStorageChunkKind::FiberletPrefix, buffers.prefixes);
        add(FiberletStorageChunkKind::FiberletRoutes, buffers.routes);
    };
    for (const auto& [key, entry] : impl_->residents)
        append(key, entry.buffers);
    for (const auto& [key, entry] : impl_->pending)
        append(key, entry->buffers);
    std::sort(result.begin(), result.end(), [](const auto& left,
                                               const auto& right) {
        return left.path.generic_string() < right.path.generic_string();
    });
    return result;
}

void finalizeFiberletDatasetIdentity(FiberletDatasetMetadata& metadata)
{
    if (!metadata.sources.is_object() || !metadata.processing.is_object())
        throw std::invalid_argument(
            "fiberlet sources and processing metadata must be JSON objects");
    metadata.algorithmFingerprint = stringFingerprint(
        algorithmIdentityJson(metadata).dump());
    metadata.datasetFingerprint = datasetFingerprint(json{
        {"algorithm", algorithmIdentityJson(metadata)},
        {"sources", metadata.sources},
    }.dump());
}

std::string fiberletContentHash(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot hash file: " + path.string());
    std::uint64_t hash = 14695981039346656037ULL;
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        for (std::streamsize index = 0; index < count; ++index) {
            hash ^= static_cast<unsigned char>(
                buffer[static_cast<std::size_t>(index)]);
            hash *= 1099511628211ULL;
        }
    }
    if (!input.eof())
        throw std::runtime_error(
            "failed while hashing file: " + path.string());
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
           << hash;
    return output.str();
}

void validateFiberletNormalDatasetCompatibility(
    const FiberletDatasetMetadata& metadata,
    const vc::lasagna::LasagnaDataset& normals)
{
    if (!(metadata.predictionToBaseScale > 0.0) ||
        !std::isfinite(metadata.predictionToBaseScale)) {
        throw std::invalid_argument(
            "fiberlet prediction-to-base scale must be positive and finite");
    }
    if (metadata.coordinateOriginZYX != std::array<std::int64_t, 3>{0, 0, 0}) {
        throw std::invalid_argument(
            "fiberlet dataset does not use the whole-volume base coordinate frame");
    }
    const auto predictionShape = fiberletPredictionShape(metadata);
    const auto& manifest = normals.manifest();
    if (!manifest.baseShapeZYX.has_value())
        throw std::invalid_argument(
            "normal manifest must declare base_shape_zyx");
    if (!nearlyEqual(manifest.workingToBaseScale,
                     metadata.predictionToBaseScale)) {
        throw std::invalid_argument(
            "normal dataset working scale does not match the Fiberlet prediction scale");
    }
    for (std::size_t axis = 0; axis < predictionShape.size(); ++axis) {
        const auto expected = static_cast<std::size_t>(std::ceil(
            static_cast<double>((*manifest.baseShapeZYX)[axis]) /
            metadata.predictionToBaseScale));
        if (predictionShape[axis] != expected) {
            throw std::invalid_argument(
                "normal manifest base_shape_zyx is incompatible with the Fiberlet prediction grid");
        }
    }

    requiredPositiveManifestNumber(manifest, "grad_mag_encode_scale");
    requiredPositiveManifestNumber(manifest, "grad_mag_factor");
    const auto nx = vc::lasagna::bindLasagnaChannel(manifest, "nx");
    const auto ny = vc::lasagna::bindLasagnaChannel(manifest, "ny");
    const auto gradMag = vc::lasagna::bindLasagnaChannel(manifest, "grad_mag");
    validateNormalBinding(manifest, nx, "nx");
    validateNormalBinding(manifest, ny, "ny");
    validateNormalBinding(manifest, gradMag, "grad_mag");
    const double nxBaseSpacing = nx.spacing * manifest.workingToBaseScale;
    const double nyBaseSpacing = ny.spacing * manifest.workingToBaseScale;
    if (nx.shapeZYX != ny.shapeZYX ||
        !nearlyEqual(nxBaseSpacing, nyBaseSpacing)) {
        throw std::invalid_argument(
            "normal nx and ny channels must have matching shape and base scale");
    }
}

FiberletChunkDataset::FiberletChunkDataset(
    std::filesystem::path root,
    FiberletDatasetMetadata metadata,
    std::shared_ptr<FiberletChunkWriteBackCache> writeBack)
    : root_(std::move(root))
    , metadata_(std::move(metadata))
    , writeBack_(std::move(writeBack))
{
    if (writeBack_)
        writeBackLayer_ = writeBack_->registerLayer(root_, metadata_.kind);
}

std::shared_ptr<FiberletChunkDataset> FiberletChunkDataset::createOrOpen(
    std::filesystem::path root,
    const FiberletDatasetMetadata& metadata,
    std::shared_ptr<FiberletChunkWriteBackCache> writeBack)
{
    validateMetadata(metadata);
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
    return std::shared_ptr<FiberletChunkDataset>(new FiberletChunkDataset(
        std::move(root), metadata, std::move(writeBack)));
}

std::shared_ptr<FiberletChunkDataset> FiberletChunkDataset::openExisting(
    std::filesystem::path root)
{
    if (!std::filesystem::is_regular_file(root / ".zattrs") ||
        !std::filesystem::is_regular_file(root / ".zgroup")) {
        throw std::invalid_argument(
            "fiberlet dataset is missing .zgroup or .zattrs metadata");
    }
    const auto group = json::parse(readText(root / ".zgroup"));
    if (group != json{{"zarr_format", 2}})
        throw std::invalid_argument("fiberlet dataset .zgroup metadata is invalid");
    auto metadata = parseMetadata(json::parse(readText(root / ".zattrs")));
    for (const auto kind : datasetKinds(metadata.kind)) {
        const auto storedArray = json::parse(
            readText(arrayDirectory(root, kind) / ".zarray"));
        if (storedArray != arrayMetadata(metadata, kind))
            throw std::invalid_argument(
                "fiberlet dataset .zarray metadata is invalid");
    }
    return std::shared_ptr<FiberletChunkDataset>(
        new FiberletChunkDataset(std::move(root), std::move(metadata)));
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

    if (writeBack_) {
        if (const auto memory = writeBack_->read(
                writeBackLayer_, kind, key)) {
            auto bytes = **memory;
            auto payload = decodeFiberletChunkPayload(kind, bytes);
            requireMatchingCodec(
                payloadConfig(kind, payload), codecConfig(kind, key));
            materializationDecodes_[static_cast<std::size_t>(kind) - 1]
                .fetch_add(1, std::memory_order_relaxed);
            return MaterializedChunk{
                std::move(bytes), std::move(payload), true};
        }
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
    if (!prefixBytes || !routeBytes ||
        (metadata_.kind == FiberletDatasetKind::Combined && !anchorBytes))
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

void FiberletChunkDataset::replaceOverlayChunk(
    FiberletStorageChunkKind kind,
    const vc::render::ChunkKey& key,
    const MaterializedChunk& chunk) const
{
    if (metadata_.kind != FiberletDatasetKind::Anchors ||
        kind != FiberletStorageChunkKind::Anchors) {
        throw std::invalid_argument(
            "single fiberlet overlay replacement requires anchors");
    }
    if (!chunk.payload)
        throw std::invalid_argument(
            "fiberlet overlay replacement has no decoded payload");
    requireMatchingCodec(
        payloadConfig(kind, chunk.payload), codecConfig(kind, key));
    if (writeBack_) {
        writeBack_->replaceAnchor(writeBackLayer_, key, chunk.bytes);
    } else {
        vc::core::util::atomicWriteBytes(chunkPath(kind, key), chunk.bytes);
    }
}

void FiberletChunkDataset::replaceOverlayChunkPair(
    const vc::render::ChunkKey& prefixKey,
    const MaterializedChunk& prefix,
    const vc::render::ChunkKey& routeKey,
    const MaterializedChunk& routes) const
{
    if (metadata_.kind != FiberletDatasetKind::Fiberlets ||
        prefixKey.level != 0 || routeKey.level != 1 ||
        prefixKey.iz != routeKey.iz || prefixKey.iy != routeKey.iy ||
        prefixKey.ix != routeKey.ix || !prefix.payload || !routes.payload) {
        throw std::invalid_argument(
            "fiberlet overlay replacement pair is invalid");
    }
    requireMatchingCodec(
        payloadConfig(
            FiberletStorageChunkKind::FiberletPrefix, prefix.payload),
        codecConfig(FiberletStorageChunkKind::FiberletPrefix, prefixKey));
    requireMatchingCodec(
        payloadConfig(
            FiberletStorageChunkKind::FiberletRoutes, routes.payload),
        codecConfig(FiberletStorageChunkKind::FiberletRoutes, routeKey));
    requireMatchingFiberletPair(prefix.payload, routes.payload);
    if (writeBack_) {
        writeBack_->replacePair(
            writeBackLayer_, prefixKey, prefix.bytes, routeKey, routes.bytes);
    } else {
        vc::core::util::atomicWriteBytes(
            chunkPath(FiberletStorageChunkKind::FiberletPrefix, prefixKey),
            prefix.bytes);
        vc::core::util::atomicWriteBytes(
            chunkPath(FiberletStorageChunkKind::FiberletRoutes, routeKey),
            routes.bytes);
    }
}

FiberletChunkDataset::PairPresence FiberletChunkDataset::pairPresence(
    const vc::render::ChunkKey& owner) const
{
    if (metadata_.kind == FiberletDatasetKind::Anchors)
        throw std::invalid_argument(
            "fiberlet pair presence requires a path dataset");
    if (writeBack_) {
        const auto memory = writeBack_->pairPresence(writeBackLayer_, owner);
        if (memory != PairPresence::Absent)
            return memory;
    }
    const vc::render::ChunkKey prefix{0, owner.iz, owner.iy, owner.ix};
    const vc::render::ChunkKey routes{1, owner.iz, owner.iy, owner.ix};
    const bool prefixExists = std::filesystem::exists(chunkPath(
        FiberletStorageChunkKind::FiberletPrefix, prefix));
    const bool routeExists = std::filesystem::exists(chunkPath(
        FiberletStorageChunkKind::FiberletRoutes, routes));
    if (prefixExists && routeExists)
        return PairPresence::Complete;
    if (prefixExists || routeExists)
        return PairPresence::Partial;
    return PairPresence::Absent;
}

void FiberletChunkDataset::validateChunk(FiberletStorageChunkKind kind, const vc::render::ChunkKey& key, std::span<const std::byte> bytes) const
{
    const auto payload = decodeFiberletChunkPayload(kind, bytes);
    requireMatchingCodec(payloadConfig(kind, payload), codecConfig(kind, key));
}

std::shared_ptr<vc::render::ChunkCache> createGeneratedFiberletChunkCache(
    std::shared_ptr<FiberletChunkDataset> dataset,
    FiberletChunkGenerator generator,
    FiberletChunkCacheOptions options,
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
    options.cache.detectAllFillChunks = false;
    options.cache.persistentCachePath.reset();
    return std::make_shared<vc::render::ChunkCache>(
        std::move(levels), std::move(fetchers), 0.0,
        vc::render::ChunkDtype::Opaque, std::move(options.cache),
        std::move(options.service));
}

std::shared_ptr<vc::render::ChunkCache> createStoredFiberletAnchorChunkCache(std::shared_ptr<FiberletChunkDataset> dataset, FiberletChunkCacheOptions options)
{
    if (!dataset || (dataset->metadata().kind != FiberletDatasetKind::Anchors && dataset->metadata().kind != FiberletDatasetKind::Combined)) {
        throw std::invalid_argument("stored fiberlet anchor cache requires anchor payloads");
    }
    const auto shape = dataset->metadata().chunkGridShapeZYX;
    options.cache.detectAllFillChunks = false;
    options.cache.persistentCachePath.reset();
    return std::make_shared<
        vc::render::ChunkCache>(std::vector<vc::render::ChunkCache::LevelInfo>{{shape, {1, 1, 1}, {}}}, std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{std::make_shared<StoredFetcher>(dataset, FiberletStorageChunkKind::Anchors)}, 0.0, vc::render::ChunkDtype::Opaque, std::move(options.cache), std::move(options.service));
}

std::shared_ptr<vc::render::ChunkCache> createStoredFiberletPathChunkCache(std::shared_ptr<FiberletChunkDataset> dataset, FiberletChunkCacheOptions options)
{
    if (!dataset || (dataset->metadata().kind != FiberletDatasetKind::Fiberlets && dataset->metadata().kind != FiberletDatasetKind::Combined)) {
        throw std::invalid_argument("stored fiberlet path cache requires prefix and route payloads");
    }
    const auto shape = dataset->metadata().chunkGridShapeZYX;
    options.cache.detectAllFillChunks = false;
    options.cache.persistentCachePath.reset();
    return std::make_shared<
        vc::render::ChunkCache>(std::vector<vc::render::ChunkCache::LevelInfo>{{shape, {1, 1, 1}, {}}, {shape, {1, 1, 1}, {}}}, std::vector<std::shared_ptr<vc::render::IChunkFetcher>>{std::make_shared<StoredFetcher>(dataset, FiberletStorageChunkKind::FiberletPrefix), std::make_shared<StoredFetcher>(dataset, FiberletStorageChunkKind::FiberletRoutes)}, 0.0, vc::render::ChunkDtype::Opaque, std::move(options.cache), std::move(options.service));
}

std::shared_ptr<vc::render::ChunkCache>
createOverlayFiberletAnchorChunkCache(
    std::shared_ptr<FiberletChunkDataset> layer,
    std::shared_ptr<FiberletChunkDataset> lowerDataset,
    std::shared_ptr<vc::render::ChunkCache> lower,
    FiberletChunkCacheOptions options)
{
    if (!layer || !lowerDataset || !lower)
        throw std::invalid_argument(
            "fiberlet anchor overlay requires both datasets and lower cache");
    requireOverlayCompatible(
        layer->metadata(), lowerDataset->metadata(),
        FiberletDatasetKind::Anchors);
    return createGeneratedFiberletChunkCache(
        layer,
        [lower = std::move(lower)](
            FiberletStorageChunkKind kind,
            const vc::render::ChunkKey& requested,
            const FiberletStorageCodecConfig&) {
            return lowerOverlayChunk(lower, kind, requested);
        },
        std::move(options));
}

std::shared_ptr<vc::render::ChunkCache>
createOverlayFiberletPathChunkCache(
    std::shared_ptr<FiberletChunkDataset> layer,
    std::shared_ptr<FiberletChunkDataset> lowerDataset,
    std::shared_ptr<vc::render::ChunkCache> lower,
    FiberletChunkCacheOptions options)
{
    if (!layer || !lowerDataset || !lower)
        throw std::invalid_argument(
            "fiberlet path overlay requires both datasets and lower cache");
    requireOverlayCompatible(
        layer->metadata(), lowerDataset->metadata(),
        FiberletDatasetKind::Fiberlets);
    return createGeneratedFiberletChunkCache(
        layer,
        [layer, lower = std::move(lower)](
            FiberletStorageChunkKind kind,
            const vc::render::ChunkKey& requested,
            const FiberletStorageCodecConfig&) {
            const auto presence = layer->pairPresence(requested);
            if (presence == FiberletChunkDataset::PairPresence::Partial) {
                throw std::runtime_error(
                    "fiberlet overlay contains a partial prefix/route pair");
            }
            if (presence == FiberletChunkDataset::PairPresence::Complete) {
                throw std::logic_error(
                    "materialized fiberlet overlay pair was not loaded");
            }
            return lowerOverlayChunk(lower, kind, requested);
        },
        std::move(options));
}

void replaceFiberletOverlayChunk(
    const std::shared_ptr<FiberletChunkDataset>& layer,
    FiberletStorageChunkKind kind,
    const vc::render::ChunkKey& key,
    const FiberletChunkDataset::MaterializedChunk& current,
    const FiberletChunkDataset::MaterializedChunk& chunk)
{
    if (!layer || !current.payload || !chunk.payload)
        throw std::invalid_argument(
            "fiberlet overlay replacement requires a decoded payload");
    if (kind != FiberletStorageChunkKind::Anchors)
        throw std::invalid_argument(
            "single fiberlet overlay replacement requires anchors");
    requireMatchingCodec(
        payloadConfig(kind, chunk.payload), layer->codecConfig(kind, key));
    const auto oldAnchors = std::dynamic_pointer_cast<
        const FiberletAnchorChunkPayload>(current.payload);
    const auto newAnchors = std::dynamic_pointer_cast<
        const FiberletAnchorChunkPayload>(chunk.payload);
    if (!oldAnchors || !newAnchors)
        throw std::invalid_argument(
            "fiberlet overlay anchor replacement payload type is invalid");
    requireMonotoneSubset<FiberletStoredAnchor, FiberletStorageKey>(
        oldAnchors->anchors, newAnchors->anchors,
        [](const auto& value) { return value.key; }, sameAnchor, "anchor");
    layer->replaceOverlayChunk(kind, key, chunk);
}

void replaceFiberletOverlayChunkPair(
    const std::shared_ptr<FiberletChunkDataset>& layer,
    const vc::render::ChunkKey& prefixKey,
    const FiberletChunkDataset::MaterializedChunk& currentPrefix,
    const FiberletChunkDataset::MaterializedChunk& prefix,
    const vc::render::ChunkKey& routeKey,
    const FiberletChunkDataset::MaterializedChunk& currentRoutes,
    const FiberletChunkDataset::MaterializedChunk& routes)
{
    if (!layer || layer->metadata().kind != FiberletDatasetKind::Fiberlets ||
        !currentPrefix.payload || !currentRoutes.payload || !prefix.payload ||
        !routes.payload || prefixKey.level != 0 ||
        routeKey.level != 1 || prefixKey.iz != routeKey.iz ||
        prefixKey.iy != routeKey.iy || prefixKey.ix != routeKey.ix) {
        throw std::invalid_argument(
            "fiberlet overlay pair replacement is invalid");
    }
    requireMatchingCodec(
        payloadConfig(FiberletStorageChunkKind::FiberletPrefix,
                      prefix.payload),
        layer->codecConfig(
            FiberletStorageChunkKind::FiberletPrefix, prefixKey));
    requireMatchingCodec(
        payloadConfig(FiberletStorageChunkKind::FiberletRoutes,
                      routes.payload),
        layer->codecConfig(
            FiberletStorageChunkKind::FiberletRoutes, routeKey));
    requireMatchingFiberletPair(prefix.payload, routes.payload);
    const auto oldPrefixes = std::dynamic_pointer_cast<
        const FiberletPrefixChunkPayload>(currentPrefix.payload);
    const auto oldRoutes = std::dynamic_pointer_cast<
        const FiberletRouteChunkPayload>(currentRoutes.payload);
    const auto newPrefixes = std::dynamic_pointer_cast<
        const FiberletPrefixChunkPayload>(prefix.payload);
    const auto newRoutes = std::dynamic_pointer_cast<
        const FiberletRouteChunkPayload>(routes.payload);
    if (!oldPrefixes || !oldRoutes || !newPrefixes || !newRoutes ||
        oldPrefixes->prefixes.size() != oldRoutes->routes.size() ||
        newPrefixes->prefixes.size() != newRoutes->routes.size()) {
        throw std::invalid_argument(
            "fiberlet overlay pair replacement payload type is invalid");
    }
    requireMonotoneSubset<FiberletStoredPrefix, FiberletStorageId>(
        oldPrefixes->prefixes, newPrefixes->prefixes,
        [](const auto& value) { return value.id; }, samePrefix, "fiberlet");
    std::size_t oldIndex = 0;
    for (std::size_t newIndex = 0;
         newIndex < newPrefixes->prefixes.size(); ++newIndex) {
        const auto& id = newPrefixes->prefixes[newIndex].id;
        while (oldPrefixes->prefixes[oldIndex].id < id)
            ++oldIndex;
        if (!sameRoute(oldRoutes->routes[oldIndex],
                       newRoutes->routes[newIndex])) {
            throw std::invalid_argument(
                "fiberlet overlay replacement mutates a route");
        }
        ++oldIndex;
    }
    layer->replaceOverlayChunkPair(prefixKey, prefix, routeKey, routes);
}

}  // namespace vc::fiber_tracer
