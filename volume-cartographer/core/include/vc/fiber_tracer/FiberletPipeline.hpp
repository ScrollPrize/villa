#pragma once
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
namespace vc::fiber_tracer {
FiberletDatasetMetadata makeFiberletDatasetMetadata(
    FiberletDatasetKind kind, const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& anchors, const FiberletPathConfig& paths,
    int storageChunkSideBaseVoxels, const nlohmann::json& sources,
    const std::vector<cv::Vec3d>& corridorReferenceBase = {},
    double corridorRadiusBaseVoxels = 0.0,
    const FiberletGeometryCacheProfile& cacheProfile = {},
    FiberletStorageProfile storageProfile = FiberletStorageProfile::Float32Cache,
    std::string_view selectionIdentity = "whole_volume_presence_chunks_v1");
// Canonical, full-chunk generation settings shared with preprocess-volume.
void configureCanonicalFiberletGeneration(FiberletOnDemandConfig& config);
}
