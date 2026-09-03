#pragma once

#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include "vc/fiber_tracer/FiberletDataset.hpp"

#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>

namespace vc::fiber_tracer
{

inline constexpr std::uint32_t kFiberletCropTraceArtifactContractVersion = 1;

struct FiberletCropTraceArtifact {
    FiberletDatasetMetadata metadata;
    cv::Vec3d minimumBaseXYZ{0.0, 0.0, 0.0};
    cv::Vec3d maximumBaseXYZ{0.0, 0.0, 0.0};
    std::vector<FiberletCropTraceLine> lines;
};

void validateFiberletCropTraceNormalDatasetCompatibility(
    const FiberletCropTraceArtifact& artifact,
    const vc::lasagna::LasagnaDataset& normals);

void writeFiberletCropTraceArtifact(
    const std::filesystem::path& output,
    const FiberletDatasetMetadata& sourceMetadata,
    const nlohmann::json& normalManifest,
    const FiberletCropTraceConfig& config,
    const std::vector<FiberletCropTraceLine>& lines,
    const nlohmann::json& preprocessing = nlohmann::json::object());

[[nodiscard]] FiberletCropTraceArtifact readFiberletCropTraceArtifact(const std::filesystem::path& input);

}  // namespace vc::fiber_tracer
