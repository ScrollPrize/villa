#pragma once
#include "FiberConstraintDatabase.hpp"

inline constexpr double defaultFiberConstraintChunkSizeBaseVoxels=2048.;

// Large spatial jobs own discovery, refinement, prediction sampling, local
// normal frames and streaming output. The only inter-job scientific state is
// the sign connecting consecutive pieces of an original source fiber.
nlohmann::json extractFiberConstraintChunks(
    const nlohmann::json& config, nlohmann::json& metadata,
    const std::vector<std::string>& ids,
    const std::vector<vc::fiber_tracer::FiberletCropTraceLine>& lines,
    const vc::fiber_tracer::FiberTraceConstraintConfig& extraction,
    const vc::lasagna::LasagnaNormalSampler& normals,
    const vc::lasagna::LasagnaNormalSampler& windings,
    const vc::core::util::Umbilicus& umbilicus,
    FiberConstraintDatabaseOptions databaseOptions,
    std::function<nlohmann::json()> memorySnapshot = {},
    // normal_source "fused": the prediction's presence channel, weighting the
    // predicted sheet axis per strip chain node.
    const vc::lasagna::LasagnaChannelBinding* presence = nullptr);
