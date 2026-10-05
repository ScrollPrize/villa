#pragma once
#include "vc/fiber_tracer/FiberletCropTraceArtifact.hpp"
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"

namespace vc::fiber_tracer {
struct FiberTraceTile {
    std::array<int,3> indexXYZ{};
    cv::Vec3d coreMin{}, coreMax{}, haloMin{}, haloMax{};
    std::vector<FiberletCropTraceLine> lines;
};
struct FiberSeamConfig {
    FiberPieceOverlapConfig overlap;
    double minimumSupportBase = 32.;
    double minimumFraction = .8;
};
struct FiberTiledTraceResult {
    std::vector<FiberletCropTraceLine> lines;
    nlohmann::json topology;
};
// Input order does not affect output IDs. No global deduplication or invented
// connecting geometry. Segment costs are linearly partitioned within a step.
FiberTiledTraceResult assembleFiberTraceTiles(std::vector<FiberTraceTile> tiles,
    const vc::lasagna::NormalSampler& normals, const FiberSeamConfig& config);
void validateFiberTiledTopology(const nlohmann::json& topology, std::size_t lines);
// Current pairwise BP supports unique seams only. Never silently drop alternatives.
void requireUnambiguousFiberTiledTopology(const nlohmann::json& topology);
void addFiberTileContinuities(FiberTraceConstraintReport& report,
    const nlohmann::json& topology, std::span<const std::size_t> originalTraceIds);
}
