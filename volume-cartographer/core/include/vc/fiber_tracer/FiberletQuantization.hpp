#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"

#include <array>
#include <compare>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vc::fiber_tracer
{

enum class FiberletCostQuantizationDomain : std::uint8_t {
    RawTotal,
    SqrtPerPredictionVoxel,
};

[[nodiscard]] std::string_view fiberletCostQuantizationDomainName(
    FiberletCostQuantizationDomain domain);

struct FiberletQuantizationScenario {
    std::string name;
    double positionQuantumBaseVoxels = 0.0;
    bool compactAxes = false;
    int costBits = 0;
    FiberletCostQuantizationDomain costDomain =
        FiberletCostQuantizationDomain::RawTotal;
    float costDensityMaximum = 0.0F;
};

struct FiberletEvaluationQuantization {
    double positionQuantumBaseVoxels = 0.0;
    bool compactDirections = false;
    int costBits = 0;
    int storageChunkSideBaseVoxels = 512;
    FiberletCostQuantizationDomain costDomain =
        FiberletCostQuantizationDomain::RawTotal;
    float costDensityMaximum = 0.0F;

    auto operator<=>(const FiberletEvaluationQuantization&) const = default;

    [[nodiscard]] bool enabled() const noexcept
    {
        return positionQuantumBaseVoxels > 0 || compactDirections ||
            costBits > 0;
    }
};

[[nodiscard]] FiberletEvaluationQuantization
defaultFiberletReplayQuantization(int storageChunkSideBaseVoxels = 512);

[[nodiscard]] FiberletEvaluationQuantization
exactFiberletReplayQuantization(int storageChunkSideBaseVoxels = 512);

struct FiberletGeometryQuantization {
    double positionQuantumBaseVoxels = 0.0;
    bool compactDirections = false;

    auto operator<=>(const FiberletGeometryQuantization&) const = default;

    [[nodiscard]] bool enabled() const noexcept
    {
        return positionQuantumBaseVoxels > 0 || compactDirections;
    }
};

struct FiberletGeometryCacheProfile {
    FiberletGeometryQuantization geometry;
    int compatibilityCostTagBits = 0;
    int storageChunkSideBaseVoxels = 512;

    auto operator<=>(const FiberletGeometryCacheProfile&) const = default;

    [[nodiscard]] bool enabled() const noexcept
    {
        return geometry.enabled();
    }
};

[[nodiscard]] FiberletGeometryCacheProfile
fiberletGeometryCacheProfile(
    FiberletEvaluationQuantization replayQuantization);

[[nodiscard]] cv::Vec3f quantizeFiberletPositionForEvaluation(
    const cv::Vec3f& positionPredictionXYZ,
    const FiberPredictionGridInfo& grid,
    double quantumBaseVoxels);

void validateFiberletPositionQuantumForEvaluation(double quantumBaseVoxels);

[[nodiscard]] std::uint64_t fiberletPositionBinCountForEvaluation(
    int chunkSideBaseVoxels,
    double quantumBaseVoxels);

[[nodiscard]] cv::Vec3f quantizeFiberletDirectionForEvaluation(
    const cv::Vec3f& directionXYZ);

[[nodiscard]] std::vector<float> quantizeFiberletCostsForEvaluation(
    std::span<const float> costs,
    int bits);

[[nodiscard]] float fiberletCostQuantizationValueForEvaluation(
    float totalCost,
    float pathLengthPredictionVoxels,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum = 0.0F);

[[nodiscard]] float quantizeFiberletCostForEvaluation(
    float totalCost,
    float pathLengthPredictionVoxels,
    float minimumValue,
    float maximumValue,
    int bits,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum = 0.0F);

[[nodiscard]] std::vector<float> quantizeFiberletCostsForEvaluation(
    std::span<const float> costs,
    std::span<const float> pathLengthsPredictionVoxels,
    int bits,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum = 0.0F);

} // namespace vc::fiber_tracer
