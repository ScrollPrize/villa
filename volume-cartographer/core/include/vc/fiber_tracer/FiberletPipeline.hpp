#pragma once
#include "vc/fiber_tracer/FiberletOnDemand.hpp"
namespace vc::fiber_tracer {
// Content identities are independent of the requested crop and output path.
nlohmann::json makeFiberletSourceIdentity(
    const std::filesystem::path& fiberManifest,
    const std::filesystem::path& normalManifest,
    nlohmann::json sourceContext = nlohmann::json::object());
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
FiberletOnDemandConfig makeCanonicalFiberletOnDemandConfig(
    const std::filesystem::path& anchorRoot,
    const std::filesystem::path& fiberletRoot,
    const FiberletDatasetMetadata& anchorMetadata,
    const FiberletDatasetMetadata& fiberletMetadata,
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& anchors,
    const FiberletPathConfig& paths,
    FiberStoredPredictionBatchSampler predictionSampler,
    std::shared_ptr<const vc::lasagna::NormalSampler> normalSampler,
    const FiberletChunkCacheOptions& cacheOptions);
}
