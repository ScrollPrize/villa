#include "vc/fiber_tracer/FiberletDataset.hpp"
#include "vc/fiber_tracer/FiberTrace.hpp"

#include "vc/core/util/AtomicFile.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
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

const char* profileName(FiberletStorageProfile profile)
{
    switch (profile) {
        case FiberletStorageProfile::Float32Cache:
            return "float32_cache";
        case FiberletStorageProfile::CompactQuantized:
            return "compact_quantized";
        case FiberletStorageProfile::CompactDirectionsFixedCost:
            return "compact_directions_fixed_cost";
  case FiberletStorageProfile::Float64Traces:
    return "float64_traces";
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
  if (value == "float64_traces")
    return FiberletStorageProfile::Float64Traces;
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
  case FiberletDatasetKind::Traces:
    return "traces";
        case FiberletDatasetKind::Published:
            return "published";
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
  if (value == "traces")
    return FiberletDatasetKind::Traces;
    if (value == "published")
        return FiberletDatasetKind::Published;
    throw std::invalid_argument("unknown fiberlet dataset kind in metadata");
}

std::vector<FiberletStorageChunkKind> datasetKinds(FiberletDatasetKind kind)
{
    if (kind == FiberletDatasetKind::Anchors)
        return {FiberletStorageChunkKind::Anchors};
    if (kind == FiberletDatasetKind::Fiberlets) {
        return {FiberletStorageChunkKind::FiberletPrefix, FiberletStorageChunkKind::FiberletRoutes};
    }
  if (kind == FiberletDatasetKind::Traces)
    return {FiberletStorageChunkKind::FiberTraces};
    if (kind == FiberletDatasetKind::Published)
        return {FiberletStorageChunkKind::PublishedFiberlets};
    return {FiberletStorageChunkKind::Anchors, FiberletStorageChunkKind::FiberletPrefix, FiberletStorageChunkKind::FiberletRoutes};
}

bool datasetContains(FiberletDatasetKind dataset, FiberletStorageChunkKind chunk)
{
    if (dataset == FiberletDatasetKind::Anchors)
        return chunk == FiberletStorageChunkKind::Anchors;
    if (dataset == FiberletDatasetKind::Fiberlets) {
        return chunk == FiberletStorageChunkKind::FiberletPrefix ||
               chunk == FiberletStorageChunkKind::FiberletRoutes;
    }
    if (dataset == FiberletDatasetKind::Combined || dataset == FiberletDatasetKind::Published)
        return chunk == FiberletStorageChunkKind::Anchors ||
            chunk == FiberletStorageChunkKind::FiberletPrefix ||
            chunk == FiberletStorageChunkKind::FiberletRoutes ||
            (dataset == FiberletDatasetKind::Published && chunk == FiberletStorageChunkKind::PublishedFiberlets);
    return chunk == FiberletStorageChunkKind::FiberTraces;
}

const char* buildStateName(FiberletDatasetBuildState state)
{
    return state == FiberletDatasetBuildState::Complete ? "complete" : "partial";
}

FiberletDatasetBuildState parseBuildState(const std::string& value)
{
    if (value == "partial")
        return FiberletDatasetBuildState::Partial;
    if (value == "complete")
        return FiberletDatasetBuildState::Complete;
    throw std::invalid_argument("unknown fiberlet dataset build state");
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

json algorithmIdentityJson(
    const FiberletDatasetMetadata& metadata,
    bool includeProducerToolchain = false)
{
    auto processing = metadata.processing;
    if (!includeProducerToolchain)
        processing.erase("producer_toolchain");
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
        {"processing", std::move(processing)},
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
  if (metadata.kind == FiberletDatasetKind::Traces) {
    if (metadata.profile != FiberletStorageProfile::Float64Traces ||
        metadata.maximumEndpointReachCoordinateUnitsZYX !=
            std::array<std::int64_t, 3>{0, 0, 0} ||
        metadata.coordinateBits != 32 || metadata.deltaBits != 32 ||
        metadata.routeCountBits != 32 || metadata.routeLatticeBits != 32 ||
        metadata.costBits != 64 || metadata.positionQuantumBaseVoxels != 0 ||
        metadata.spatialChunkSideBaseVoxels == 0) {
      throw std::invalid_argument(
          "Fiber trace dataset metadata is not canonical");
    }
    for (const auto units : metadata.coordinateUnitsPerChunkZYX) {
      if (units != metadata.spatialChunkSideBaseVoxels) {
        throw std::invalid_argument(
            "Fiber trace chunks must use base-coordinate ownership");
      }
    }
  } else if (metadata.profile == FiberletStorageProfile::Float64Traces) {
    throw std::invalid_argument(
        "float64 trace profile requires a Fiber trace dataset");
  }
    auto canonical = metadata;
    canonical.algorithmFingerprint = stringFingerprint(
        algorithmIdentityJson(canonical).dump());
    canonical.datasetFingerprint = datasetFingerprint(json{
        {"algorithm", algorithmIdentityJson(canonical)},
        {"sources", canonical.sources},
    }.dump());
    if (metadata.algorithmFingerprint == canonical.algorithmFingerprint &&
        metadata.datasetFingerprint == canonical.datasetFingerprint) {
        return;
    }
    if (metadata.processing.contains("producer_toolchain")) {
        canonical.algorithmFingerprint = stringFingerprint(
            algorithmIdentityJson(canonical, true).dump());
        canonical.datasetFingerprint = datasetFingerprint(json{
            {"algorithm", algorithmIdentityJson(canonical, true)},
            {"sources", canonical.sources},
        }.dump());
    }
    if (metadata.algorithmFingerprint != canonical.algorithmFingerprint ||
        metadata.datasetFingerprint != canonical.datasetFingerprint) {
        throw std::invalid_argument(
            "fiberlet metadata fingerprints do not match its structured identity");
    }
}

json metadataJson(const FiberletDatasetMetadata& metadata)
{
    json result = {
        {"vc_format", "fiberlet_dataset"},
        {"format_version", metadata.kind == FiberletDatasetKind::Published ? 3 : 2},
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
    };
    if (metadata.kind == FiberletDatasetKind::Published) {
        result["build_state"] = buildStateName(metadata.buildState);
        result["expected_chunk_count"] = metadata.expectedChunkCount
            ? json(*metadata.expectedChunkCount) : json(nullptr);
        result["expected_chunk_digest"] = metadata.expectedChunkDigest
            ? json(*metadata.expectedChunkDigest) : json(nullptr);
    }
    return result;
}

FiberletDatasetMetadata parseMetadata(const json& value)
{
    static const std::vector<std::string> commonRequired{
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
        "processing"};
    if (!value.is_object())
        throw std::invalid_argument("fiberlet dataset metadata has unknown or missing fields");
    for (const auto& name : commonRequired) {
        if (!value.contains(name))
            throw std::invalid_argument("fiberlet dataset metadata is missing " + name);
    }
    if (value.at("vc_format") != "fiberlet_dataset")
        throw std::invalid_argument("fiberlet dataset metadata header is invalid");
    FiberletDatasetMetadata result;
    result.kind = parseKind(value.at("dataset_kind").get<std::string>());
    const int version = value.at("format_version").get<int>();
    const std::size_t expectedFields = commonRequired.size() +
        (result.kind == FiberletDatasetKind::Published ? 3 : value.contains("build_state") ? 1 : 0);
    if (value.size() != expectedFields ||
        version != (result.kind == FiberletDatasetKind::Published ? 3 : 2))
        throw std::invalid_argument("fiberlet dataset metadata header is invalid");
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
    if (result.kind == FiberletDatasetKind::Published) {
        result.buildState = parseBuildState(value.at("build_state").get<std::string>());
        if (!value.contains("expected_chunk_count") ||
            !value.contains("expected_chunk_digest"))
            throw std::invalid_argument(
                "published fiberlet metadata is missing its expected inventory");
        if (!value.at("expected_chunk_count").is_null())
            result.expectedChunkCount = value.at("expected_chunk_count").get<std::uint64_t>();
        if (!value.at("expected_chunk_digest").is_null())
            result.expectedChunkDigest = value.at("expected_chunk_digest").get<std::string>();
        if (result.expectedChunkCount.has_value() != result.expectedChunkDigest.has_value())
            throw std::invalid_argument(
                "published fiberlet expected inventory is incomplete");
        if (result.buildState == FiberletDatasetBuildState::Complete &&
            !result.expectedChunkCount)
            throw std::invalid_argument(
                "complete published fiberlet metadata has no expected inventory");
    } else if (value.contains("build_state") &&
               parseBuildState(value.at("build_state").get<std::string>()) != FiberletDatasetBuildState::Partial) {
        throw std::invalid_argument(
            "legacy fiberlet construction datasets must remain partial");
    }
    validateMetadata(result);
    return result;
}

json arrayMetadata(const FiberletDatasetMetadata& metadata, FiberletStorageChunkKind kind)
{
    const char* sampleFormat = kind == FiberletStorageChunkKind::Anchors          ? "fiberlet-anchor-v2"
                               : kind == FiberletStorageChunkKind::FiberletPrefix ? "fiberlet-edge-prefix-v2"
                               : kind == FiberletStorageChunkKind::FiberletRoutes ? "fiberlet-route-v3"
                               : kind == FiberletStorageChunkKind::FiberTraces ? "fiber-trace-v1"
                                                                                  : "fiberlet-published-v1";
    const int codecVersion = kind == FiberletStorageChunkKind::FiberletRoutes ? 3
                             : (kind == FiberletStorageChunkKind::FiberTraces || kind == FiberletStorageChunkKind::PublishedFiberlets) ? 1
                                                                             : 2;
    json result = {
        {"zarr_format", 2},
        {"shape", metadata.chunkGridShapeZYX},
        {"chunks", {1, 1, 1}},
        {"dtype", "|O"},
        {"fill_value", nullptr},
        {"order", "C"},
        {"filters", {{{"id", "vc-fiberlet-chunk"}, {"codec_version", codecVersion}, {"sample_format", sampleFormat}}}},
        {"compressor", nullptr},
    };
    if (metadata.kind == FiberletDatasetKind::Published)
        result["dimension_separator"] = "/";
    return result;
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
    if (kind == FiberletStorageChunkKind::PublishedFiberlets)
        return root / "fiberlets";
    if (kind == FiberletStorageChunkKind::Anchors)
        return root / "anchors";
    if (kind == FiberletStorageChunkKind::FiberletPrefix)
        return root / "prefix";
  if (kind == FiberletStorageChunkKind::FiberletRoutes)
    return root / "routes";
  return root / "traces";
}

std::string expectedChunkDigest(
    std::span<const FiberletChunkKey> chunks)
{
    std::string canonical;
    canonical.reserve(chunks.size() * 24);
    for (const auto& key : chunks) {
        canonical += std::to_string(key.iz);
        canonical.push_back('/');
        canonical += std::to_string(key.iy);
        canonical.push_back('/');
        canonical += std::to_string(key.ix);
        canonical.push_back('\n');
    }
    return "fnv256:" + hexFingerprint(datasetFingerprint(canonical));
}

bool parseNonnegativeInt(std::string_view text, int& value)
{
    if (text.empty())
        return false;
    int parsed = 0;
    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
        parsed < 0)
        return false;
    value = parsed;
    return true;
}

std::vector<FiberletChunkKey> publishedPhysicalChunks(
    const std::filesystem::path& root)
{
    const auto array = root / "fiberlets";
    const auto arrayCanonical = std::filesystem::weakly_canonical(array);
    std::vector<FiberletChunkKey> result;
    if (!std::filesystem::is_directory(array))
        return result;
    for (const auto& entry :
        std::filesystem::recursive_directory_iterator(array)) {
        if (!entry.is_regular_file())
            continue;
        const auto resolvedRelative = std::filesystem::weakly_canonical(
            entry.path()).lexically_relative(arrayCanonical);
        if (resolvedRelative.empty() || *resolvedRelative.begin() == "..")
            throw std::invalid_argument(
                "published fiberlet chunk path escapes its array root");
        const auto relative = std::filesystem::relative(entry.path(), array);
        if (relative == ".zarray")
            continue;
        const auto components = std::distance(relative.begin(), relative.end());
        if (components != 3)
            throw std::invalid_argument(
                "published fiberlet array contains an unexpected file: " +
                relative.string());
        auto it = relative.begin();
        int z = 0;
        int y = 0;
        int x = 0;
        if (!parseNonnegativeInt(it++->string(), z) ||
            !parseNonnegativeInt(it++->string(), y) ||
            !parseNonnegativeInt(it->string(), x)) {
            throw std::invalid_argument(
                "published fiberlet array contains an invalid chunk key: " +
                relative.string());
        }
        result.push_back({0, z, y, x});
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) {
        return std::tie(left.iz, left.iy, left.ix) <
            std::tie(right.iz, right.iy, right.ix);
    });
    return result;
}

void writeDatasetMetadata(
    const std::filesystem::path& root,
    const FiberletDatasetMetadata& metadata)
{
    vc::core::util::atomicWriteString(
        root / ".zattrs", metadataJson(metadata).dump(2) + "\n");
}

void removeLegacyBookkeeping(const std::filesystem::path& root)
{
    std::filesystem::remove(root / "active_chunks.bin");
    std::filesystem::remove(root / "dataset.complete");
    std::filesystem::remove_all(root / "complete");
}

}  // namespace

FiberletAnchorChunkPayload::FiberletAnchorChunkPayload(FiberletDecodedAnchors decoded)
    : config(std::move(decoded.config)), anchors(std::move(decoded.anchors))
{
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

FiberletTraceChunkPayload::FiberletTraceChunkPayload(
    FiberletDecodedTraces decoded)
    : config(std::move(decoded.config)), traces(std::move(decoded.traces)) {}

FiberletPublishedChunkPayload::FiberletPublishedChunkPayload(
    FiberletDecodedPublishedChunk decodedValue)
    : decoded(std::move(decodedValue))
{
}

std::shared_ptr<const FiberletChunkPayload>
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
  if (kind == FiberletStorageChunkKind::FiberTraces) {
    return std::make_shared<const FiberletTraceChunkPayload>(
        deserializeFiberletTraces(bytes));
  }
    if (kind == FiberletStorageChunkKind::PublishedFiberlets) {
        return std::make_shared<const FiberletPublishedChunkPayload>(
            deserializePublishedFiberletChunk(bytes));
    }
    throw std::invalid_argument("unknown fiberlet storage chunk kind");
}

namespace
{

const FiberletStorageCodecConfig& payloadConfig(
    FiberletStorageChunkKind kind,
    const std::shared_ptr<const FiberletChunkPayload>& payload)
{
    if (kind == FiberletStorageChunkKind::PublishedFiberlets) {
        const auto typed = std::dynamic_pointer_cast<const FiberletPublishedChunkPayload>(payload);
        if (!typed)
            throw std::invalid_argument("published fiberlet payload type is invalid");
        return typed->decoded.config;
    }
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
    if (kind == FiberletStorageChunkKind::FiberletRoutes) {
        const auto typed = std::dynamic_pointer_cast<
            const FiberletRouteChunkPayload>(payload);
        if (!typed)
            throw std::invalid_argument("fiberlet route chunk payload type is invalid");
        return typed->config;
    }
    const auto typed = std::dynamic_pointer_cast<
        const FiberletTraceChunkPayload>(payload);
    if (!typed)
        throw std::invalid_argument("Fiber trace chunk payload type is invalid");
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
    const std::shared_ptr<const FiberletChunkPayload>& prefix,
    const std::shared_ptr<const FiberletChunkPayload>& routes)
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

void requirePublishedOwnership(
    const FiberletChunkDataset& dataset,
    const FiberletChunkKey& key,
    const FiberletDecodedPublishedChunk& decoded)
{
    const auto sameCoordinate = [&](const FiberletChunkKey& candidate) {
        return candidate.iz == key.iz && candidate.iy == key.iy &&
            candidate.ix == key.ix;
    };
    for (const auto& anchor : decoded.ownedAnchors.anchors) {
        if (!sameCoordinate(dataset.chunkOwningAnchor(anchor.key)))
            throw std::invalid_argument(
                "published Fiberlet owned anchor belongs to another chunk");
    }
    for (const auto& anchor : decoded.referencedAnchors.anchors) {
        if (sameCoordinate(dataset.chunkOwningAnchor(anchor.key)))
            throw std::invalid_argument(
                "published Fiberlet referenced anchor belongs to its owner chunk");
    }
    for (const auto& prefix : decoded.prefixes.prefixes) {
        if (!sameCoordinate(dataset.chunkOwningAnchor(prefix.id.first)))
            throw std::invalid_argument(
                "published Fiberlet is stored outside its canonical owner chunk");
    }
}

}  // namespace

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
    validateFiberletNormalDatasetCompatibility(
        FiberPredictionGridInfo{fiberletPredictionShape(metadata), metadata.predictionToBaseScale},
        normals);
}

void validateFiberletNormalDatasetCompatibility(
    const FiberPredictionGridInfo& grid,
    const vc::lasagna::LasagnaDataset& normals)
{
    if (!(grid.predictionToBaseScale > 0.0) || !std::isfinite(grid.predictionToBaseScale))
        throw std::invalid_argument("fiberlet prediction-to-base scale must be positive and finite");
    const auto& predictionShape = grid.shapeZYX;
    const auto& manifest = normals.manifest();
    if (!manifest.baseShapeZYX.has_value())
        throw std::invalid_argument(
            "normal manifest must declare base_shape_zyx");
    if (!nearlyEqual(manifest.workingToBaseScale,
                     grid.predictionToBaseScale)) {
        throw std::invalid_argument(
            "normal dataset working scale does not match the Fiberlet prediction scale");
    }
    for (std::size_t axis = 0; axis < predictionShape.size(); ++axis) {
        const auto expected = static_cast<std::size_t>(std::ceil(
            static_cast<double>((*manifest.baseShapeZYX)[axis]) /
            grid.predictionToBaseScale));
        if (predictionShape[axis] != expected) {
            throw std::invalid_argument(
                "normal manifest base_shape_zyx is incompatible with the Fiberlet prediction grid");
        }
    }

    vc::lasagna::validateLasagnaNormalDatasetStructure(normals);
}

bool fiberletDatasetMetadataCompatible(
    const FiberletDatasetMetadata& left,
    const FiberletDatasetMetadata& right)
{
    auto canonicalLeft = left;
    auto canonicalRight = right;
    canonicalLeft.processing.erase("producer_toolchain");
    canonicalRight.processing.erase("producer_toolchain");
    finalizeFiberletDatasetIdentity(canonicalLeft);
    finalizeFiberletDatasetIdentity(canonicalRight);
    return canonicalLeft.algorithmFingerprint ==
            canonicalRight.algorithmFingerprint &&
        canonicalLeft.datasetFingerprint == canonicalRight.datasetFingerprint;
}

FiberletChunkDataset::FiberletChunkDataset(
    std::filesystem::path root,
    FiberletDatasetMetadata metadata)
    : root_(std::move(root))
    , metadata_(std::move(metadata))
{
}

std::shared_ptr<FiberletChunkDataset> FiberletChunkDataset::createOrOpen(
    std::filesystem::path root,
    const FiberletDatasetMetadata& metadata)
{
    validateMetadata(metadata);
    const auto expected = metadataJson(metadata);
    auto effectiveMetadata = metadata;
    if (std::filesystem::exists(root / ".zattrs")) {
        const auto group = json::parse(readText(root / ".zgroup"));
        if (group != json{{"zarr_format", 2}})
            throw std::invalid_argument("fiberlet dataset .zgroup metadata is invalid");
        const auto storedJson = json::parse(readText(root / ".zattrs"));
        const auto parsed = parseMetadata(storedJson);
        if (!fiberletDatasetMetadataCompatible(parsed, metadata))
            throw std::invalid_argument("fiberlet dataset metadata does not match the requested configuration");
        effectiveMetadata = parsed;
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
        std::move(root), std::move(effectiveMetadata)));
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
    if (metadata_.kind == FiberletDatasetKind::Combined) {
        if (!expectedChunksConfigured_)
            throw std::invalid_argument("combined dataset completeness requires source-derived expected chunks");
        for (const auto& key : expectedChunks_) {
            if (!readMaterializedChunk(FiberletStorageChunkKind::Anchors, key))
                return false;
        }
        return true;
    }
    if (metadata_.kind != FiberletDatasetKind::Published)
        throw std::invalid_argument(
            "whole-dataset completeness requires a combined or published dataset");
    if (metadata_.buildState != FiberletDatasetBuildState::Complete ||
        !metadata_.expectedChunkCount || !metadata_.expectedChunkDigest)
        return false;
    const auto physical = publishedPhysicalChunks(root_);
    if (physical.size() != *metadata_.expectedChunkCount ||
        expectedChunkDigest(physical) != *metadata_.expectedChunkDigest)
        return false;
    if (expectedChunksConfigured_ && physical != expectedChunks_)
        return false;
    for (const auto& key : physical) {
        if (!readPublishedChunk(key))
            return false;
    }
    return true;
}

void FiberletChunkDataset::configureExpectedChunks(std::span<const FiberletChunkKey> chunks)
{
    if (metadata_.kind != FiberletDatasetKind::Combined &&
        metadata_.kind != FiberletDatasetKind::Published)
        throw std::invalid_argument(
            "fiberlet expected chunks require a combined or published dataset");
    std::vector<FiberletChunkKey> normalized(chunks.begin(), chunks.end());
    for (auto& key : normalized) {
        key.level = 0;
        (void)codecConfig(
            metadata_.kind == FiberletDatasetKind::Published
                ? FiberletStorageChunkKind::PublishedFiberlets
                : FiberletStorageChunkKind::Anchors,
            key);
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
    if (metadata_.kind == FiberletDatasetKind::Published) {
        const auto count = static_cast<std::uint64_t>(expectedChunks_.size());
        const auto digest = expectedChunkDigest(expectedChunks_);
        const bool inventoryChanged = metadata_.expectedChunkCount != count ||
            metadata_.expectedChunkDigest != digest;
        const bool completeButInvalid =
            metadata_.buildState == FiberletDatasetBuildState::Complete &&
            !datasetComplete();
        if (inventoryChanged || completeButInvalid) {
            metadata_.buildState = FiberletDatasetBuildState::Partial;
            metadata_.expectedChunkCount = count;
            metadata_.expectedChunkDigest = digest;
            writeDatasetMetadata(root_, metadata_);
        }
    }
}

void FiberletChunkDataset::markPartial()
{
    if (metadata_.kind != FiberletDatasetKind::Published)
        throw std::invalid_argument(
            "only a published fiberlet dataset has a completion lifecycle");
    if (metadata_.buildState != FiberletDatasetBuildState::Partial) {
        metadata_.buildState = FiberletDatasetBuildState::Partial;
        writeDatasetMetadata(root_, metadata_);
    }
}

void FiberletChunkDataset::markComplete()
{
    if (metadata_.kind != FiberletDatasetKind::Published ||
        !expectedChunksConfigured_ || !metadata_.expectedChunkCount ||
        !metadata_.expectedChunkDigest) {
        throw std::invalid_argument(
            "published fiberlet completion requires configured expected chunks");
    }
    const auto physical = publishedPhysicalChunks(root_);
    if (physical != expectedChunks_)
        throw std::invalid_argument(
            "published fiberlet physical chunk inventory is incomplete or contains stale chunks");
    for (const auto& key : expectedChunks_) {
        if (!readPublishedChunk(key))
            throw std::invalid_argument(
                "published fiberlet expected chunk is missing");
    }
    metadata_.buildState = FiberletDatasetBuildState::Complete;
    writeDatasetMetadata(root_, metadata_);
}

const std::vector<FiberletChunkKey>& FiberletChunkDataset::expectedChunks() const noexcept
{
    return expectedChunks_;
}

bool FiberletChunkDataset::isExpectedChunk(const FiberletChunkKey& key) const
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
          materializationDecodes_[2].load(std::memory_order_relaxed),
          materializationDecodes_[3].load(std::memory_order_relaxed)};
}

FiberletStorageCodecConfig FiberletChunkDataset::codecConfig(FiberletStorageChunkKind kind, const FiberletChunkKey& key) const
{
    if (!datasetContains(metadata_.kind, kind))
        throw std::invalid_argument("fiberlet chunk kind does not belong to this dataset");
    const int expectedLevel =
        kind == FiberletStorageChunkKind::FiberletRoutes ? 1 : 0;
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

FiberletChunkKey FiberletChunkDataset::chunkOwningAnchor(
    const FiberletStorageKey& anchor, int level) const
{
    std::array<std::int64_t, 3> coordinate{};
    const auto floorDiv = [](std::int64_t value, std::int64_t divisor) {
        auto quotient = value / divisor;
        const auto remainder = value % divisor;
        if (remainder != 0 && ((remainder < 0) != (divisor < 0)))
            --quotient;
        return quotient;
    };
    for (std::size_t axis = 0; axis < 3; ++axis) {
        coordinate[axis] = floorDiv(
            anchor.coordinateZYX[axis] - metadata_.coordinateOriginZYX[axis],
            metadata_.coordinateUnitsPerChunkZYX[axis]);
        if (coordinate[axis] < 0 ||
            coordinate[axis] >= metadata_.chunkGridShapeZYX[axis])
            throw std::out_of_range(
                "fiberlet anchor lies outside the dataset grid");
    }
    return {level, static_cast<int>(coordinate[0]),
        static_cast<int>(coordinate[1]), static_cast<int>(coordinate[2])};
}

std::filesystem::path FiberletChunkDataset::chunkPath(FiberletStorageChunkKind kind, const FiberletChunkKey& key) const
{
    (void)codecConfig(kind, key);
    if (metadata_.kind == FiberletDatasetKind::Published) {
        return arrayDirectory(root_, FiberletStorageChunkKind::PublishedFiberlets) /
            std::to_string(key.iz) / std::to_string(key.iy) /
            std::to_string(key.ix);
    }
    return arrayDirectory(root_, kind) / (std::to_string(key.iz) + "." + std::to_string(key.iy) + "." + std::to_string(key.ix));
}

std::optional<std::vector<std::byte>> FiberletChunkDataset::readChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key) const
{
    auto materialized = readMaterializedChunk(kind, key);
    if (!materialized)
        return std::nullopt;
    return std::move(materialized->bytes);
}

std::optional<FiberletChunkDataset::MaterializedChunk> FiberletChunkDataset::readMaterializedChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key) const
{
    if (expectedChunksConfigured_ && !isExpectedChunk(key)) {
        if (kind == FiberletStorageChunkKind::PublishedFiberlets)
            return std::nullopt;
        const auto codec = codecConfig(kind, key);
        std::vector<std::byte> bytes;
        if (kind == FiberletStorageChunkKind::Anchors)
            bytes = serializeFiberletAnchors(codec, {});
        else if (kind == FiberletStorageChunkKind::FiberletPrefix)
            bytes = serializeFiberletPrefixes(codec, {});
    else if (kind == FiberletStorageChunkKind::FiberletRoutes)
            bytes = serializeFiberletRoutes(codec, {});
    else
      bytes = serializeFiberletTraces(codec, {});
        return MaterializedChunk{bytes, decodeFiberletChunkPayload(kind, bytes), true};
    }

    if (metadata_.kind == FiberletDatasetKind::Published) {
        const FiberletChunkKey ownerKey{0, key.iz, key.iy, key.ix};
        auto bytes = readBytes(chunkPath(
            FiberletStorageChunkKind::PublishedFiberlets, ownerKey));
        if (!bytes)
            return std::nullopt;
        auto decoded = deserializePublishedFiberletChunk(*bytes);
        requireMatchingCodec(
            decoded.config,
            codecConfig(FiberletStorageChunkKind::PublishedFiberlets, ownerKey));
        requirePublishedOwnership(*this, ownerKey, decoded);
        if (kind == FiberletStorageChunkKind::PublishedFiberlets) {
            materializationDecodes_[4].fetch_add(
                1, std::memory_order_relaxed);
            return MaterializedChunk{
                std::move(*bytes),
                std::make_shared<const FiberletPublishedChunkPayload>(
                    std::move(decoded)),
                true};
        }
        if (kind == FiberletStorageChunkKind::Anchors) {
            auto payload = std::make_shared<const FiberletAnchorChunkPayload>(
                std::move(decoded.ownedAnchors));
            materializationDecodes_[0].fetch_add(
                1, std::memory_order_relaxed);
            return MaterializedChunk{
                std::move(decoded.ownedAnchorPayload), std::move(payload), true};
        }
        if (kind == FiberletStorageChunkKind::FiberletPrefix) {
            auto payload = std::make_shared<FiberletPrefixChunkPayload>(
                std::move(decoded.prefixes));
            auto endpoints = std::move(decoded.ownedAnchors);
            endpoints.anchors.insert(endpoints.anchors.end(),
                decoded.referencedAnchors.anchors.begin(),
                decoded.referencedAnchors.anchors.end());
            std::sort(endpoints.anchors.begin(), endpoints.anchors.end(),
                [](const auto& a, const auto& b) { return a.key < b.key; });
            payload->endpoints = std::make_shared<const FiberletAnchorChunkPayload>(
                std::move(endpoints));
            materializationDecodes_[1].fetch_add(
                1, std::memory_order_relaxed);
            return MaterializedChunk{
                std::move(decoded.prefixPayload), std::move(payload), true};
        }
        auto payload = std::make_shared<const FiberletRouteChunkPayload>(
            std::move(decoded.routes));
        materializationDecodes_[2].fetch_add(
            1, std::memory_order_relaxed);
        return MaterializedChunk{
            std::move(decoded.routePayload), std::move(payload), true};
    }

  if (metadata_.kind == FiberletDatasetKind::Anchors ||
      metadata_.kind == FiberletDatasetKind::Traces) {
        auto bytes = readBytes(chunkPath(kind, key));
        if (!bytes)
            return std::nullopt;
        auto payload = decodeFiberletChunkPayload(kind, *bytes);
        requireMatchingCodec(payloadConfig(kind, payload), codecConfig(kind, key));
        materializationDecodes_[static_cast<std::size_t>(kind) - 1].fetch_add(1, std::memory_order_relaxed);
        return MaterializedChunk{std::move(*bytes), std::move(payload), true};
    }

    const FiberletChunkKey ownerKey{0, key.iz, key.iy, key.ix};
    const FiberletChunkKey routeKey{1, key.iz, key.iy, key.ix};
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

    std::shared_ptr<const FiberletChunkPayload> anchorPayload;
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

void FiberletChunkDataset::publishChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key, std::span<const std::byte> bytes) const
{
    MaterializedChunk chunk;
    chunk.bytes.assign(bytes.begin(), bytes.end());
    chunk.payload = decodeFiberletChunkPayload(kind, bytes);
    publishMaterializedChunk(kind, key, chunk);
}

void FiberletChunkDataset::publishMaterializedChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key, const MaterializedChunk& chunk) const
{
    if (metadata_.kind == FiberletDatasetKind::Published) {
        if (kind != FiberletStorageChunkKind::PublishedFiberlets)
            throw std::invalid_argument("published fiberlets require self-contained chunks");
        publishPublishedChunk(key, chunk.bytes);
        return;
    }
    if (!chunk.payload)
        throw std::invalid_argument("fiberlet publication has no decoded payload");
    requireMatchingCodec(payloadConfig(kind, chunk.payload), codecConfig(kind, key));
    publishBytes(chunkPath(kind, key), chunk.bytes);
}

void FiberletChunkDataset::publishPublishedChunk(
    const FiberletChunkKey& key,
    std::span<const std::byte> bytes) const
{
    if (metadata_.kind != FiberletDatasetKind::Published)
        throw std::invalid_argument(
            "self-contained publication requires a published fiberlet dataset");
    const auto decoded = deserializePublishedFiberletChunk(bytes);
    requireMatchingCodec(
        decoded.config,
        codecConfig(FiberletStorageChunkKind::PublishedFiberlets, key));
    requirePublishedOwnership(*this, key, decoded);
    publishBytes(
        chunkPath(FiberletStorageChunkKind::PublishedFiberlets, key), bytes);
}

std::optional<FiberletDecodedPublishedChunk>
FiberletChunkDataset::readPublishedChunk(const FiberletChunkKey& key) const
{
    if (metadata_.kind != FiberletDatasetKind::Published)
        throw std::invalid_argument(
            "self-contained read requires a published fiberlet dataset");
    const auto bytes = readBytes(
        chunkPath(FiberletStorageChunkKind::PublishedFiberlets, key));
    if (!bytes)
        return std::nullopt;
    auto decoded = deserializePublishedFiberletChunk(*bytes);
    requireMatchingCodec(
        decoded.config,
        codecConfig(FiberletStorageChunkKind::PublishedFiberlets, key));
    requirePublishedOwnership(*this, key, decoded);
    return decoded;
}

void FiberletChunkDataset::publishFiberletChunkPair(
    const FiberletChunkKey& prefixKey, const MaterializedChunk& prefix, const FiberletChunkKey& routeKey, const MaterializedChunk& routes) const
{
  if (metadata_.kind == FiberletDatasetKind::Anchors ||
      metadata_.kind == FiberletDatasetKind::Published ||
      metadata_.kind == FiberletDatasetKind::Traces)
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

void FiberletChunkDataset::validateChunk(FiberletStorageChunkKind kind, const FiberletChunkKey& key, std::span<const std::byte> bytes) const
{
    const auto payload = decodeFiberletChunkPayload(kind, bytes);
    requireMatchingCodec(payloadConfig(kind, payload), codecConfig(kind, key));
}

namespace
{

std::string memoryCacheKey(
    const FiberletChunkDataset& dataset,
    FiberletStorageChunkKind kind,
    const FiberletChunkKey& key)
{
    return dataset.root().string() + '#' +
        std::to_string(static_cast<int>(kind)) + '/' + std::to_string(key.iz) +
        '/' + std::to_string(key.iy) + '/' + std::to_string(key.ix);
}

// Decode a chunk the memory cache holds. Published prefix and route views come
// from their whole self-contained chunk; other kinds are their own bytes.
std::shared_ptr<const FiberletChunkPayload> cachedPayload(
    const FiberletChunkDataset& dataset,
    FiberletStorageChunkKind kind,
    const FiberletChunkKey& key,
    vcr::MemoryCache* cache)
{
    if (!cache)
        return {};
    const auto [found, chunk] = cache->get(memoryCacheKey(dataset, kind, key));
    if (!found || !chunk)
        return {};
    return decodeFiberletChunkPayload(
        kind, std::as_bytes(std::span(*chunk)));
}

void cacheBytes(
    const FiberletChunkDataset& dataset,
    FiberletStorageChunkKind kind,
    const FiberletChunkKey& key,
    std::span<const std::byte> bytes,
    vcr::MemoryCache* cache)
{
    if (!cache || bytes.empty())
        return;
    const auto* begin = reinterpret_cast<const std::uint8_t*>(bytes.data());
    cache->put(
        memoryCacheKey(dataset, kind, key),
        std::make_shared<const vcr::Chunk>(begin, begin + bytes.size()));
}

std::vector<std::byte> emptyChunkBytes(
    const FiberletChunkDataset& dataset,
    FiberletStorageChunkKind kind,
    const FiberletChunkKey& key)
{
    const auto codec = dataset.codecConfig(kind, key);
    if (kind == FiberletStorageChunkKind::Anchors)
        return serializeFiberletAnchors(codec, {});
    if (kind == FiberletStorageChunkKind::FiberletPrefix)
        return serializeFiberletPrefixes(codec, {});
    return serializeFiberletRoutes(codec, {});
}

// Absent sparse chunks are empty, unless only part of their tuple exists.
void requireCompleteOrAbsentTuple(
    const FiberletChunkDataset& dataset, const FiberletChunkKey& key)
{
    if (dataset.metadata().kind == FiberletDatasetKind::Anchors)
        return;
    const FiberletChunkKey owner{0, key.iz, key.iy, key.ix};
    const FiberletChunkKey route{1, key.iz, key.iy, key.ix};
    const bool prefixPresent = std::filesystem::exists(
        dataset.chunkPath(FiberletStorageChunkKind::FiberletPrefix, owner));
    const bool routePresent = std::filesystem::exists(
        dataset.chunkPath(FiberletStorageChunkKind::FiberletRoutes, route));
    bool anchorPresent = false;
    std::size_t present = prefixPresent + routePresent;
    std::size_t required = 2;
    if (dataset.metadata().kind == FiberletDatasetKind::Combined) {
        ++required;
        anchorPresent = std::filesystem::exists(
            dataset.chunkPath(FiberletStorageChunkKind::Anchors, owner));
        present += anchorPresent;
    }
    if (present != 0 && present != required) {
        std::ostringstream message;
        message << "fiberlet sparse chunk tuple " << key.iz << ',' << key.iy
                << ',' << key.ix << " is only partially present (anchors="
                << anchorPresent << ", prefix=" << prefixPresent
                << ", routes=" << routePresent << ')';
        throw std::invalid_argument(message.str());
    }
}

std::shared_ptr<const FiberletChunkPayload> readStoredChunk(
    const FiberletChunkDataset& dataset,
    FiberletStorageChunkKind kind,
    const FiberletChunkKey& key,
    const std::shared_ptr<vcr::MemoryCache>& cache)
{
    // Published views are decoded from the whole chunk, whose nested payloads
    // alone lack the owner's endpoint table: their decode stays on the read path.
    const bool published =
        dataset.metadata().kind == FiberletDatasetKind::Published;
    if (!published) {
        if (auto payload = cachedPayload(dataset, kind, key, cache.get()))
            return payload;
    }
    auto chunk = dataset.readMaterializedChunk(kind, key);
    if (!chunk) {
        requireCompleteOrAbsentTuple(dataset, key);
        const auto bytes = emptyChunkBytes(dataset, kind, key);
        return decodeFiberletChunkPayload(kind, bytes);
    }
    if (!published)
        cacheBytes(dataset, kind, key, chunk->bytes, cache.get());
    return std::move(chunk->payload);
}

FiberletStorageChunkKind pathKind(const FiberletChunkKey& key)
{
    if (key.level == 0)
        return FiberletStorageChunkKind::FiberletPrefix;
    if (key.level == 1)
        return FiberletStorageChunkKind::FiberletRoutes;
    throw std::invalid_argument("fiberlet path chunk level is invalid");
}

}  // namespace

FiberletChunkRead fiberletGeneratedChunkRead(
    std::shared_ptr<FiberletChunkDataset> dataset,
    FiberletChunkGenerator generator,
    std::shared_ptr<vcr::MemoryCache> cache)
{
    if (!dataset || !generator)
        throw std::invalid_argument("generated fiberlet read requires a dataset and generator");
    if (dataset->metadata().kind == FiberletDatasetKind::Published)
        throw std::invalid_argument(
            "published fiberlet datasets are not construction caches");
    return [dataset = std::move(dataset), generator = std::move(generator), cache = std::move(cache)](
               const FiberletChunkKey& key) -> std::shared_ptr<const FiberletChunkPayload> {
        const auto datasetKind = dataset->metadata().kind;
        const auto kind = datasetKind == FiberletDatasetKind::Anchors
            ? FiberletStorageChunkKind::Anchors
            : datasetKind == FiberletDatasetKind::Traces
                ? FiberletStorageChunkKind::FiberTraces
                : pathKind(key);
        if (auto payload = cachedPayload(*dataset, kind, key, cache.get()))
            return payload;
        if (auto chunk = dataset->readMaterializedChunk(kind, key)) {
            cacheBytes(*dataset, kind, key, chunk->bytes, cache.get());
            return std::move(chunk->payload);
        }
        auto generated = generator(kind, key, dataset->codecConfig(kind, key));
        if (!generated.payload)
            throw std::invalid_argument(
                "fiberlet generator returned no decoded payload");
        if (!generated.alreadyPublished)
            dataset->publishMaterializedChunk(kind, key, generated);
        cacheBytes(*dataset, kind, key, generated.bytes, cache.get());
        return std::move(generated.payload);
    };
}

FiberletChunkRead fiberletStoredAnchorRead(
    std::shared_ptr<FiberletChunkDataset> dataset,
    std::shared_ptr<vcr::MemoryCache> cache)
{
    if (!dataset ||
        (dataset->metadata().kind != FiberletDatasetKind::Anchors &&
         dataset->metadata().kind != FiberletDatasetKind::Combined &&
         dataset->metadata().kind != FiberletDatasetKind::Published)) {
        throw std::invalid_argument("stored fiberlet anchor read requires anchor payloads");
    }
    if (dataset->metadata().kind == FiberletDatasetKind::Published && !dataset->datasetComplete())
        throw std::invalid_argument("published fiberlet dataset is incomplete");
    return [dataset = std::move(dataset), cache = std::move(cache)](const FiberletChunkKey& key) {
        return readStoredChunk(*dataset, FiberletStorageChunkKind::Anchors, key, cache);
    };
}

FiberletChunkRead fiberletStoredPathRead(
    std::shared_ptr<FiberletChunkDataset> dataset,
    std::shared_ptr<vcr::MemoryCache> cache)
{
    if (!dataset ||
        (dataset->metadata().kind != FiberletDatasetKind::Fiberlets &&
         dataset->metadata().kind != FiberletDatasetKind::Combined &&
         dataset->metadata().kind != FiberletDatasetKind::Published)) {
        throw std::invalid_argument("stored fiberlet path read requires route payloads");
    }
    if (dataset->metadata().kind == FiberletDatasetKind::Published && !dataset->datasetComplete())
        throw std::invalid_argument("published fiberlet dataset is incomplete");
    return [dataset = std::move(dataset), cache = std::move(cache)](const FiberletChunkKey& key) {
        return readStoredChunk(*dataset, pathKind(key), key, cache);
    };
}

}  // namespace vc::fiber_tracer
