#pragma once

#include "vc/lasagna/Manifest.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace vc::lasagna {

struct MaxflowInt3 {
    int64_t x = 0;
    int64_t y = 0;
    int64_t z = 0;
};

struct MaxflowDouble3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct MaxflowBox3 {
    MaxflowInt3 begin;
    MaxflowInt3 end;
};

struct MaxflowGraphBuildOptions {
    bool storeNodeMetadata = false;
};

struct MaxflowManifestBuildOptions {
    MaxflowDouble3 sourceBase;
    MaxflowDouble3 sinkBase;
    std::vector<MaxflowDouble3> sourcesBase;
    std::vector<MaxflowDouble3> sinksBase;
    int64_t marginBaseVoxels = 1000;
    uint8_t threshold = 110;
    double workingToBaseScale = 1.0;
};

enum class MaxflowNodeKind : uint8_t {
    Voxel,
    ChunkCenter,
    ChunkPortal
};

struct MaxflowNodeMetadata {
    MaxflowNodeKind kind = MaxflowNodeKind::Voxel;
    MaxflowBox3 predBoundsXYZ;
    int direction = -1;
};

struct MaxflowGraphStats {
    uint64_t passableVoxels = 0;
    uint64_t fullBlocks = 0;
    uint64_t mixedBlocks = 0;
    uint64_t leafVoxels = 0;
    uint64_t obstacleBlocks = 0;
    uint64_t graphNodes = 0;
    uint64_t directedEdges = 0;
    uint64_t undirectedEdges = 0;
    double averageEdgesPerNode = 0.0;
    uint64_t totalExternalCapacity = 0;
    uint64_t uncontractedPassableVoxelGraphNodes = 0;
    uint64_t uncontractedPassableVoxelGraphUndirectedEdges = 0;
    double contractionRatio = 0.0;
    std::chrono::duration<double> buildTime{};
};

struct MaxflowGraph {
    std::vector<uint64_t> nindex;
    std::vector<int32_t> nlist;
    std::vector<int32_t> capacity;
    std::vector<MaxflowNodeMetadata> nodeMetadata;
    MaxflowGraphStats stats;
};

struct MaxflowTerminal {
    MaxflowInt3 predVoxelXYZ;
    int32_t node = -1;
    bool exact = false;
};

struct MaxflowPredDtVolume {
    std::filesystem::path zarrPath;
    std::array<size_t, 3> shapeZYX{0, 0, 0};
    std::array<size_t, 3> chunksZYX{0, 0, 0};
    double spacingBase = 1.0;
    MaxflowBox3 cropBaseXYZ;
    MaxflowBox3 cropPredXYZ;
    std::vector<uint8_t> passable;
    uint64_t passableVoxels = 0;
};

struct MaxflowGraphBuildReport {
    LasagnaDatasetManifest manifest;
    MaxflowPredDtVolume predDt;
    MaxflowGraph graph;
    MaxflowTerminal source;
    MaxflowTerminal sink;
    std::vector<MaxflowTerminal> sources;
    std::vector<MaxflowTerminal> sinks;
};

[[nodiscard]] MaxflowGraph buildMaxflowGraphFromPassability(
    std::span<const uint8_t> passableZYX,
    std::array<size_t, 3> shapeZYX,
    const MaxflowGraphBuildOptions& options = {});

[[nodiscard]] MaxflowPredDtVolume loadPredDtPassability(
    const LasagnaDatasetManifest& manifest,
    const MaxflowManifestBuildOptions& options);

[[nodiscard]] MaxflowGraphBuildReport buildMaxflowGraphFromManifest(
    const std::filesystem::path& manifestPath,
    const MaxflowManifestBuildOptions& options);

[[nodiscard]] MaxflowTerminal findNearestPassableNode(
    std::span<const uint8_t> passableZYX,
    std::array<size_t, 3> shapeZYX,
    const MaxflowBox3& cropPredXYZ,
    const MaxflowDouble3& pointBase,
    double predDtSpacingBase);

[[nodiscard]] const char* toString(MaxflowNodeKind kind) noexcept;

} // namespace vc::lasagna
