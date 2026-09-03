#include "vc/fiber_tracer/FiberletCropTraceArtifact.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

namespace vc::fiber_tracer
{
namespace
{

using ChunkCoordinate = std::array<std::int32_t, 3>;

bool finite(const cv::Vec3d& value)
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

std::array<double, 3> jsonVector(const cv::Vec3d& value)
{
    return {value[0], value[1], value[2]};
}

cv::Vec3d parseVector(const nlohmann::json& value, const char* name)
{
    if (!value.is_array() || value.size() != 3)
        throw std::invalid_argument(std::string(name) + " must contain three coordinates");
    const cv::Vec3d result{value[0].get<double>(), value[1].get<double>(), value[2].get<double>()};
    if (!finite(result))
        throw std::invalid_argument(std::string(name) + " is not finite");
    return result;
}

void requireObjectKeys(const nlohmann::json& value, std::initializer_list<std::string_view> keys, const char* name)
{
    if (!value.is_object() || value.size() != keys.size())
        throw std::invalid_argument(std::string(name) + " has unknown or missing fields");
    for (const auto key : keys) {
        if (!value.contains(key))
            throw std::invalid_argument(std::string(name) + " is missing " + std::string(key));
    }
}

ChunkCoordinate ownerFor(const FiberletDatasetMetadata& metadata, const cv::Vec3d& seedBaseXYZ)
{
    ChunkCoordinate owner{};
    for (std::size_t zyx = 0; zyx < 3; ++zyx) {
        const std::size_t xyz = 2 - zyx;
        const double relative = seedBaseXYZ[static_cast<int>(xyz)] - static_cast<double>(metadata.coordinateOriginZYX[zyx]);
        const double side = static_cast<double>(metadata.spatialChunkSideBaseVoxels);
        const auto coordinate = static_cast<std::int64_t>(std::floor(relative / side));
        if (coordinate < 0 || coordinate >= metadata.chunkGridShapeZYX[zyx]) {
            throw std::invalid_argument("Fiber trace seed is outside its artifact chunk grid");
        }
        owner[zyx] = static_cast<std::int32_t>(coordinate);
    }
    return owner;
}

vc::render::ChunkKey chunkKey(const ChunkCoordinate& coordinate)
{
    return {0, coordinate[0], coordinate[1], coordinate[2]};
}

ChunkCoordinate parseChunkName(const std::string& name)
{
    ChunkCoordinate result{};
    std::size_t begin = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::size_t end = axis == 2 ? name.size() : name.find('.', begin);
        if (end == std::string::npos || end == begin)
            throw std::invalid_argument("Fiber trace dataset contains a malformed chunk name");
        const auto* first = name.data() + begin;
        const auto* last = name.data() + end;
        const auto parsed = std::from_chars(first, last, result[axis]);
        if (parsed.ec != std::errc{} || parsed.ptr != last || result[axis] < 0)
            throw std::invalid_argument("Fiber trace dataset contains a malformed chunk name");
        begin = end + 1;
    }
    return result;
}

std::filesystem::path temporarySibling(const std::filesystem::path& output)
{
    std::random_device random;
    const auto nonce = (static_cast<std::uint64_t>(random()) << 32) ^ random() ^
                       static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    return output.parent_path() / ("." + output.filename().string() + ".tmp-" + std::to_string(nonce));
}

FiberletDatasetMetadata traceMetadata(
    const FiberletDatasetMetadata& source,
    const nlohmann::json& normalManifest,
    const FiberletCropTraceConfig& config,
    const std::vector<ChunkCoordinate>& populated,
    std::size_t traceCount,
    const nlohmann::json& preprocessing)
{
    if (!finite(config.minimumBaseXYZ) || !finite(config.maximumBaseXYZ))
        throw std::invalid_argument("Fiber trace crop bounds must be finite");
    FiberletDatasetMetadata result;
    result.kind = FiberletDatasetKind::Traces;
    result.profile = FiberletStorageProfile::Float64Traces;
    result.spatialChunkSideBaseVoxels = source.spatialChunkSideBaseVoxels;
    if (result.spatialChunkSideBaseVoxels == 0)
        throw std::invalid_argument("source Fiberlet spatial chunk side is zero");
    result.coordinateBits = 32;
    result.deltaBits = 32;
    result.routeCountBits = 32;
    result.routeLatticeBits = 32;
    result.costBits = 64;
    result.positionQuantumBaseVoxels = 0;
    result.predictionToBaseScale = source.predictionToBaseScale;
    result.maximumEndpointReachCoordinateUnitsZYX = {0, 0, 0};
    const auto side = static_cast<double>(result.spatialChunkSideBaseVoxels);
    for (std::size_t zyx = 0; zyx < 3; ++zyx) {
        const std::size_t xyz = 2 - zyx;
        if (!(config.maximumBaseXYZ[static_cast<int>(xyz)] > config.minimumBaseXYZ[static_cast<int>(xyz)])) {
            throw std::invalid_argument("Fiber trace crop must have positive extent");
        }
        const double aligned = std::floor(config.minimumBaseXYZ[static_cast<int>(xyz)] / side) * side;
        if (aligned < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
            aligned > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("Fiber trace crop origin exceeds int64");
        }
        result.coordinateOriginZYX[zyx] = static_cast<std::int64_t>(aligned);
        result.coordinateUnitsPerChunkZYX[zyx] = result.spatialChunkSideBaseVoxels;
        const double count = std::ceil((config.maximumBaseXYZ[static_cast<int>(xyz)] - aligned) / side);
        if (!(count >= 1.0) || count > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("Fiber trace crop chunk grid exceeds int32");
        }
        result.chunkGridShapeZYX[zyx] = static_cast<std::int32_t>(count);
    }
    nlohmann::json inventory = nlohmann::json::array();
    for (const auto& coordinate : populated)
        inventory.push_back(coordinate);
    result.sources = {
        {"source_fiberlet_dataset_fingerprint", source.datasetFingerprint},
        {"source_fiberlet_algorithm_fingerprint", source.algorithmFingerprint},
        {"normal_manifest", normalManifest},
    };
    result.processing = {
        {"trace_contract_version", kFiberletCropTraceArtifactContractVersion},
        {"coordinate_order", "zyx_storage_xyz_vectors"},
        {"crop",
         {
             {"minimum_base_xyz", jsonVector(config.minimumBaseXYZ)},
             {"maximum_base_xyz", jsonVector(config.maximumBaseXYZ)},
         }},
        {"trace",
         {
             {"beam_width", config.beamWidth},
             {"lookahead_distance_base", config.lookaheadDistanceBaseVoxels},
             {"maximum_generated_states_per_step", config.maximumGeneratedStatesPerStep},
             {"maximum_fiberlets_per_side", config.maximumFiberletsPerSide},
             {"coverage_normal_radius_base", config.coverageNormalRadiusBaseVoxels},
             {"coverage_direction_degrees", config.coverageDirectionDegrees},
             {"stop_at_covered_anchors", config.stopAtCoveredAnchors},
             {"maximum_accepted_cost_density",
              config.maximumAcceptedCostDensity
                  ? nlohmann::json(*config.maximumAcceptedCostDensity)
                  : nlohmann::json(nullptr)},
             {"maximum_attempts", config.maximumAttempts},
             {"maximum_fibers", config.maximumFibers},
         }},
        {"artifact",
         {
             {"trace_count", traceCount},
             {"populated_chunks_zyx", std::move(inventory)},
         }},
    };
    if (!preprocessing.empty())
        result.processing["preprocessing"] = preprocessing;
    finalizeFiberletDatasetIdentity(result);
    return result;
}

void validateEqual(const std::vector<FiberletCropTraceLine>& expected, const std::vector<FiberletCropTraceLine>& actual)
{
    if (expected.size() != actual.size())
        throw std::runtime_error("published Fiber trace count changed during validation");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const auto& left = expected[index];
        const auto& right = actual[index];
        if (left.seedBaseXYZ != right.seedBaseXYZ || left.seedPresence != right.seedPresence || left.totalMetricCost != right.totalMetricCost ||
            left.pathLengthPredictionVoxels != right.pathLengthPredictionVoxels || left.pointsBaseXYZ != right.pointsBaseXYZ) {
            throw std::runtime_error("published Fiber trace changed during validation");
        }
    }
}

}  // namespace

void validateFiberletCropTraceNormalDatasetCompatibility(
    const FiberletCropTraceArtifact& artifact,
    const vc::lasagna::LasagnaDataset& normals)
{
    if (!(artifact.metadata.predictionToBaseScale > 0.0) ||
        !std::isfinite(artifact.metadata.predictionToBaseScale)) {
        throw std::invalid_argument(
            "Fiber trace prediction-to-base scale must be positive and finite");
    }
    const auto& manifest = normals.manifest();
    if (std::abs(manifest.workingToBaseScale - 1.0) > 1.0e-12) {
        throw std::invalid_argument(
            "Fiber trace constraints require normals opened in base coordinates");
    }
    vc::lasagna::validateLasagnaNormalDatasetStructure(normals);
    if (!manifest.baseShapeZYX.has_value())
        throw std::invalid_argument("normal manifest must declare base_shape_zyx");
    for (std::size_t xyz = 0; xyz < 3; ++xyz) {
        const std::size_t zyx = 2 - xyz;
        const double begin = artifact.minimumBaseXYZ[static_cast<int>(xyz)];
        const double end = artifact.maximumBaseXYZ[static_cast<int>(xyz)];
        if (!(begin >= 0.0) ||
            !(end <= static_cast<double>((*manifest.baseShapeZYX)[zyx]))) {
            throw std::invalid_argument(
                "Fiber trace crop is outside the normal manifest base_shape_zyx");
        }
    }
}

void writeFiberletCropTraceArtifact(
    const std::filesystem::path& output,
    const FiberletDatasetMetadata& sourceMetadata,
    const nlohmann::json& normalManifest,
    const FiberletCropTraceConfig& config,
    const std::vector<FiberletCropTraceLine>& lines,
    const nlohmann::json& preprocessing)
{
    if (std::filesystem::exists(output))
        throw std::invalid_argument("Fiber trace output already exists: " + output.string());
    std::map<ChunkCoordinate, std::vector<FiberletStoredTrace>> grouped;
    FiberletDatasetMetadata layout = traceMetadata(
        sourceMetadata, normalManifest, config, {}, lines.size(),
        preprocessing);
    for (std::size_t index = 0; index < lines.size(); ++index) {
        const auto& line = lines[index];
        grouped[ownerFor(layout, line.seedBaseXYZ)].push_back({
            index,
            line.seedBaseXYZ,
            line.seedPresence,
            line.totalMetricCost,
            line.pathLengthPredictionVoxels,
            line.pointsBaseXYZ,
        });
    }
    std::vector<ChunkCoordinate> populated;
    populated.reserve(grouped.size());
    for (const auto& [coordinate, traces] : grouped) {
        (void)traces;
        populated.push_back(coordinate);
    }
    auto metadata = traceMetadata(
        sourceMetadata, normalManifest, config, populated, lines.size(),
        preprocessing);

    const auto parent = output.parent_path().empty() ? std::filesystem::path{"."} : output.parent_path();
    std::filesystem::create_directories(parent);
    const auto temporary = temporarySibling(output);
    try {
        auto dataset = FiberletChunkDataset::createOrOpen(temporary, metadata);
        for (const auto& [coordinate, traces] : grouped) {
            const auto key = chunkKey(coordinate);
            dataset->publishChunk(FiberletStorageChunkKind::FiberTraces, key, serializeFiberletTraces(dataset->codecConfig(FiberletStorageChunkKind::FiberTraces, key), traces));
        }
        dataset.reset();
        const auto reopened = readFiberletCropTraceArtifact(temporary);
        validateEqual(lines, reopened.lines);
        std::filesystem::rename(temporary, output);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(temporary, ignored);
        throw;
    }
}

FiberletCropTraceArtifact readFiberletCropTraceArtifact(const std::filesystem::path& input)
{
    auto dataset = FiberletChunkDataset::openExisting(input);
    const auto& metadata = dataset->metadata();
    if (metadata.kind != FiberletDatasetKind::Traces || metadata.profile != FiberletStorageProfile::Float64Traces) {
        throw std::invalid_argument("input is not a Fiber trace dataset");
    }
    requireObjectKeys(
        metadata.sources,
        {"source_fiberlet_dataset_fingerprint", "source_fiberlet_algorithm_fingerprint", "normal_manifest"},
        "Fiber trace source metadata");
    if (!metadata.sources.at("source_fiberlet_dataset_fingerprint").is_array() ||
        !metadata.sources.at("source_fiberlet_algorithm_fingerprint").is_string() || !metadata.sources.at("normal_manifest").is_object()) {
        throw std::invalid_argument("Fiber trace source metadata is invalid");
    }
    const auto& processing = metadata.processing;
    if (processing.contains("preprocessing")) {
        requireObjectKeys(
            processing,
            {"trace_contract_version", "coordinate_order", "crop", "trace",
             "artifact", "preprocessing"},
            "Fiber trace processing metadata");
        if (!processing.at("preprocessing").is_object())
            throw std::invalid_argument(
                "Fiber trace preprocessing metadata is not an object");
    } else {
        requireObjectKeys(
            processing,
            {"trace_contract_version", "coordinate_order", "crop", "trace",
             "artifact"},
            "Fiber trace processing metadata");
    }
    if (processing.at("trace_contract_version") != kFiberletCropTraceArtifactContractVersion ||
        processing.at("coordinate_order") != "zyx_storage_xyz_vectors") {
        throw std::invalid_argument("Fiber trace processing contract is unsupported");
    }
    requireObjectKeys(processing.at("crop"), {"minimum_base_xyz", "maximum_base_xyz"}, "Fiber trace crop metadata");
    const auto& trace = processing.at("trace");
    const bool hasCoveredStop = trace.contains("stop_at_covered_anchors");
    const bool hasQualityThreshold =
        trace.contains("maximum_accepted_cost_density");
    if (hasCoveredStop && hasQualityThreshold) {
        requireObjectKeys(
            trace,
            {"beam_width", "lookahead_distance_base",
             "maximum_generated_states_per_step",
             "maximum_fiberlets_per_side", "coverage_normal_radius_base",
             "coverage_direction_degrees", "stop_at_covered_anchors",
             "maximum_accepted_cost_density", "maximum_attempts",
             "maximum_fibers"},
            "Fiber trace parameters");
    } else if (hasCoveredStop) {
        requireObjectKeys(
            trace,
            {"beam_width", "lookahead_distance_base",
             "maximum_generated_states_per_step",
             "maximum_fiberlets_per_side", "coverage_normal_radius_base",
             "coverage_direction_degrees", "stop_at_covered_anchors",
             "maximum_attempts", "maximum_fibers"},
            "Fiber trace parameters");
    } else if (hasQualityThreshold) {
        requireObjectKeys(
            trace,
            {"beam_width", "lookahead_distance_base",
             "maximum_generated_states_per_step",
             "maximum_fiberlets_per_side", "coverage_normal_radius_base",
             "coverage_direction_degrees", "maximum_accepted_cost_density",
             "maximum_attempts", "maximum_fibers"},
            "Fiber trace parameters");
    } else {
        requireObjectKeys(
            trace,
            {"beam_width", "lookahead_distance_base",
             "maximum_generated_states_per_step",
             "maximum_fiberlets_per_side", "coverage_normal_radius_base",
             "coverage_direction_degrees", "maximum_attempts",
             "maximum_fibers"},
            "Fiber trace parameters");
    }
    if (hasCoveredStop && !trace.at("stop_at_covered_anchors").is_boolean())
        throw std::invalid_argument(
            "Fiber trace covered-anchor parameter is not boolean");
    if (hasQualityThreshold &&
        !trace.at("maximum_accepted_cost_density").is_null() &&
        (!trace.at("maximum_accepted_cost_density").is_number() ||
         !std::isfinite(trace.at("maximum_accepted_cost_density").get<double>()) ||
         trace.at("maximum_accepted_cost_density").get<double>() < 0.0)) {
        throw std::invalid_argument(
            "Fiber trace quality threshold parameter is invalid");
    }
    requireObjectKeys(processing.at("artifact"), {"trace_count", "populated_chunks_zyx"}, "Fiber trace artifact metadata");
    FiberletCropTraceArtifact result;
    result.metadata = metadata;
    result.minimumBaseXYZ = parseVector(processing.at("crop").at("minimum_base_xyz"), "Fiber trace crop minimum");
    result.maximumBaseXYZ = parseVector(processing.at("crop").at("maximum_base_xyz"), "Fiber trace crop maximum");

    std::vector<ChunkCoordinate> expected;
    const auto& inventory = processing.at("artifact").at("populated_chunks_zyx");
    if (!inventory.is_array())
        throw std::invalid_argument("Fiber trace chunk inventory is not an array");
    for (const auto& entry : inventory) {
        if (!entry.is_array() || entry.size() != 3)
            throw std::invalid_argument("Fiber trace chunk inventory entry is invalid");
        const ChunkCoordinate coordinate{entry[0].get<std::int32_t>(), entry[1].get<std::int32_t>(), entry[2].get<std::int32_t>()};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (coordinate[axis] < 0 || coordinate[axis] >= metadata.chunkGridShapeZYX[axis]) {
                throw std::invalid_argument("Fiber trace chunk inventory is outside the grid");
            }
        }
        if (!expected.empty() && !(expected.back() < coordinate))
            throw std::invalid_argument("Fiber trace chunk inventory is not strictly sorted");
        expected.push_back(coordinate);
    }

    std::vector<ChunkCoordinate> actual;
    const auto traceDirectory = input / "traces";
    for (const auto& entry : std::filesystem::directory_iterator(traceDirectory)) {
        if (entry.path().filename() == ".zarray")
            continue;
        if (!entry.is_regular_file())
            throw std::invalid_argument("Fiber trace array contains an unexpected entry");
        actual.push_back(parseChunkName(entry.path().filename().string()));
    }
    std::sort(actual.begin(), actual.end());
    if (actual != expected)
        throw std::invalid_argument("Fiber trace chunk inventory does not match stored chunks");

    struct OrdinalLine {
        std::uint64_t ordinal = 0;
        FiberletCropTraceLine line;
    };
    std::vector<OrdinalLine> ordered;
    for (const auto& coordinate : expected) {
        const auto key = chunkKey(coordinate);
        const auto chunk = dataset->readMaterializedChunk(FiberletStorageChunkKind::FiberTraces, key);
        if (!chunk)
            throw std::invalid_argument("Fiber trace inventory chunk is missing");
        const auto payload = std::dynamic_pointer_cast<const FiberletTraceChunkPayload>(chunk->payload);
        if (!payload || payload->traces.empty())
            throw std::invalid_argument("Fiber trace inventory chunk is empty or invalid");
        for (const auto& stored : payload->traces) {
            if (ownerFor(metadata, stored.seedBaseXYZ) != coordinate)
                throw std::invalid_argument("Fiber trace is stored under the wrong owner chunk");
            FiberletCropTraceLine line;
            line.seedBaseXYZ = stored.seedBaseXYZ;
            line.seedPresence = stored.seedPresence;
            line.totalMetricCost = stored.totalMetricCost;
            line.pathLengthPredictionVoxels = stored.pathLengthPredictionVoxels;
            line.pointsBaseXYZ = stored.pointsBaseXYZ;
            ordered.push_back({stored.ordinal, std::move(line)});
        }
    }
    const auto expectedCount = processing.at("artifact").at("trace_count").get<std::uint64_t>();
    if (ordered.size() != expectedCount)
        throw std::invalid_argument("Fiber trace record count does not match metadata");
    std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) { return left.ordinal < right.ordinal; });
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        if (ordered[index].ordinal != index)
            throw std::invalid_argument("Fiber trace ordinals are duplicate or incomplete");
        result.lines.push_back(std::move(ordered[index].line));
    }
    return result;
}

}  // namespace vc::fiber_tracer
