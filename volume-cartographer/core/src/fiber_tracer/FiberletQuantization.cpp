#include "vc/fiber_tracer/FiberletQuantization.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace vc::fiber_tracer {
namespace {
constexpr double kEpsilon = 1.0e-12;
double length(const cv::Vec3f& value)
{
    return std::sqrt(static_cast<double>(value.dot(value)));
}

cv::Vec3f normalized(const cv::Vec3f& value)
{
    const double norm = length(value);
    if (!(norm > kEpsilon) || !std::isfinite(norm))
        throw std::invalid_argument("quantized fiberlet has a degenerate axis");
    return value * static_cast<float>(1.0 / norm);
}

}
cv::Vec3f quantizeFiberletPositionForEvaluation(
    const cv::Vec3f& positionPredictionXYZ,
    const FiberPredictionGridInfo& grid,
    double quantumBaseVoxels)
{
    validateFiberletPositionQuantumForEvaluation(quantumBaseVoxels);
    if (quantumBaseVoxels == 0.0)
        return positionPredictionXYZ;
    if (!(grid.predictionToBaseScale > 0.0) ||
        !std::isfinite(grid.predictionToBaseScale)) {
        throw std::invalid_argument(
            "fiberlet evaluation position scale is invalid");
    }
    cv::Vec3f result;
    for (size_t xyz = 0; xyz < 3; ++xyz) {
        const double base = static_cast<double>(positionPredictionXYZ[xyz]) *
            grid.predictionToBaseScale;
        if (!(base >= 0.0) || !std::isfinite(base))
            throw std::invalid_argument(
                "fiberlet evaluation position is outside nonnegative coordinates");
        const double decodedBase = quantumBaseVoxels *
            std::floor(base / quantumBaseVoxels + 0.5);
        const size_t shapeAxis = grid.shapeZYX[2 - xyz];
        if (!(decodedBase < static_cast<double>(shapeAxis) *
                grid.predictionToBaseScale)) {
            throw std::invalid_argument(
                "fiberlet evaluation position leaves the prediction volume");
        }
        result[static_cast<int>(xyz)] = static_cast<float>(
            decodedBase / grid.predictionToBaseScale);
    }
    return result;
}

void validateFiberletPositionQuantumForEvaluation(double quantumBaseVoxels)
{
    if (quantumBaseVoxels == 0.0)
        return;
    if (!(quantumBaseVoxels > 0.0) || !std::isfinite(quantumBaseVoxels)) {
        throw std::invalid_argument(
            "fiberlet evaluation position quantum must be zero or positive and finite");
    }
}

std::uint64_t fiberletPositionBinCountForEvaluation(
    int chunkSideBaseVoxels,
    double quantumBaseVoxels)
{
    validateFiberletPositionQuantumForEvaluation(quantumBaseVoxels);
    if (chunkSideBaseVoxels <= 0)
        throw std::invalid_argument(
            "fiberlet evaluation position chunk side must be positive");
    if (quantumBaseVoxels == 0.0)
        return 0;
    const double bins =
        static_cast<double>(chunkSideBaseVoxels) / quantumBaseVoxels;
    const double roundedBins = std::round(bins);
    const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(bins));
    if (!(roundedBins >= 1.0) || !std::isfinite(roundedBins) ||
        std::abs(bins - roundedBins) > tolerance ||
        roundedBins > static_cast<double>(
            std::numeric_limits<std::uint64_t>::max())) {
        throw std::invalid_argument(
            "fiberlet evaluation position quantum does not form an integral chunk grid");
    }
    return static_cast<std::uint64_t>(roundedBins);
}

cv::Vec3f quantizeFiberletDirectionForEvaluation(
    const cv::Vec3f& directionXYZ)
{
    const auto encoded = vc::lasagna::encodeCompactNormalToRaw(
        {directionXYZ[0], directionXYZ[1], directionXYZ[2]});
    if (!encoded.has_value())
        throw std::invalid_argument(
            "fiberlet evaluation direction is not compact-encodable");
    const cv::Vec3d decoded = vc::lasagna::decodeCompactNormalFromRaw(
        (*encoded)[0], (*encoded)[1]);
    return normalized({static_cast<float>(decoded[0]),
        static_cast<float>(decoded[1]), static_cast<float>(decoded[2])});
}

std::vector<float> quantizeFiberletCostsForEvaluation(
    std::span<const float> costs,
    int bits)
{
    if (bits != 8 && bits != 16)
        return {costs.begin(), costs.end()};
    if (costs.empty())
        return {};
    for (const float cost : costs) {
        if (!(cost >= 0.0F) || !std::isfinite(cost))
            throw std::invalid_argument(
                "fiberlet evaluation cost is invalid");
    }
    const auto [minimumIt, maximumIt] =
        std::minmax_element(costs.begin(), costs.end());
    const float minimum = *minimumIt;
    const float maximum = *maximumIt;
    std::vector<float> result;
    result.reserve(costs.size());
    for (const float cost : costs)
        result.push_back(quantizeFiberletCostForEvaluation(
            cost, 1.0F, minimum, maximum, bits,
            FiberletCostQuantizationDomain::RawTotal));
    return result;
}

std::string_view fiberletCostQuantizationDomainName(
    FiberletCostQuantizationDomain domain)
{
    switch (domain) {
    case FiberletCostQuantizationDomain::RawTotal:
        return "raw_total";
    case FiberletCostQuantizationDomain::SqrtPerPredictionVoxel:
        return "sqrt_per_prediction_voxel";
    }
    throw std::invalid_argument("unknown fiberlet cost quantization domain");
}

float fiberletCostQuantizationValueForEvaluation(
    float totalCost,
    float pathLengthPredictionVoxels,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum)
{
    if (!(totalCost >= 0.0F) || !std::isfinite(totalCost))
        throw std::invalid_argument("fiberlet evaluation cost is invalid");
    if (domain == FiberletCostQuantizationDomain::RawTotal)
        return totalCost;
    if (domain != FiberletCostQuantizationDomain::SqrtPerPredictionVoxel ||
        !(pathLengthPredictionVoxels > 0.0F) ||
        !std::isfinite(pathLengthPredictionVoxels) ||
        !(costDensityMaximum > 0.0F) || !std::isfinite(costDensityMaximum)) {
        throw std::invalid_argument("fiberlet evaluation cost density length is invalid");
    }
    const float density = totalCost / pathLengthPredictionVoxels;
    if (!(density >= 0.0F) || !std::isfinite(density))
        throw std::invalid_argument("fiberlet evaluation cost density is invalid");
    return std::sqrt(std::min(density / costDensityMaximum, 1.0F));
}

float quantizeFiberletCostForEvaluation(
    float totalCost,
    float pathLengthPredictionVoxels,
    float minimumValue,
    float maximumValue,
    int bits,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum)
{
    if (bits != 8 && bits != 16)
        return totalCost;
    const float value = fiberletCostQuantizationValueForEvaluation(
        totalCost, pathLengthPredictionVoxels, domain,
        costDensityMaximum);
    if (domain == FiberletCostQuantizationDomain::SqrtPerPredictionVoxel) {
        minimumValue = 0.0F;
        maximumValue = 1.0F;
    }
    if (!(minimumValue >= 0.0F) || !std::isfinite(minimumValue) ||
        maximumValue < minimumValue || !std::isfinite(maximumValue) ||
        value < minimumValue || value > maximumValue) {
        throw std::invalid_argument("fiberlet evaluation cost lies outside its affine range");
    }
    const std::uint32_t levels = bits == 8 ? 255U : 65535U;
    const float scale = maximumValue == minimumValue
        ? 0.0F
        : (maximumValue - minimumValue) / static_cast<float>(levels);
    std::uint32_t code = 0;
    if (value == maximumValue) {
        code = levels;
    } else if (scale > 0.0F) {
        const float raw = (value - minimumValue) / scale;
        if (!std::isfinite(raw) || raw < 0.0F ||
            raw > static_cast<float>(levels)) {
            throw std::invalid_argument("fiberlet evaluation cost lies outside its affine range");
        }
        code = static_cast<std::uint32_t>(std::floor(raw + 0.5F));
    }
    const float decodedValue = minimumValue + scale * static_cast<float>(code);
    const float decodedTotal = domain == FiberletCostQuantizationDomain::SqrtPerPredictionVoxel
        ? decodedValue * decodedValue * costDensityMaximum *
            pathLengthPredictionVoxels
        : decodedValue;
    if (!(decodedTotal >= 0.0F) || !std::isfinite(decodedTotal))
        throw std::invalid_argument("decoded fiberlet evaluation cost is invalid");
    return decodedTotal;
}

std::vector<float> quantizeFiberletCostsForEvaluation(
    std::span<const float> costs,
    std::span<const float> pathLengthsPredictionVoxels,
    int bits,
    FiberletCostQuantizationDomain domain,
    float costDensityMaximum)
{
    if (costs.size() != pathLengthsPredictionVoxels.size())
        throw std::invalid_argument("fiberlet evaluation cost and length counts differ");
    if (costs.empty())
        return {};
    std::vector<float> values;
    values.reserve(costs.size());
    for (size_t index = 0; index < costs.size(); ++index) {
        values.push_back(fiberletCostQuantizationValueForEvaluation(
            costs[index], pathLengthsPredictionVoxels[index], domain,
            costDensityMaximum));
    }
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    std::vector<float> result;
    result.reserve(costs.size());
    for (size_t index = 0; index < costs.size(); ++index) {
        result.push_back(quantizeFiberletCostForEvaluation(
            costs[index], pathLengthsPredictionVoxels[index], *minimum, *maximum,
            bits, domain, costDensityMaximum));
    }
    return result;
}

FiberletEvaluationQuantization defaultFiberletReplayQuantization(
    int storageChunkSideBaseVoxels)
{
    if (storageChunkSideBaseVoxels <= 0)
        throw std::invalid_argument(
            "fiberlet replay storage chunk side must be positive");
    return {
        0.0,
        true,
        16,
        storageChunkSideBaseVoxels,
        FiberletCostQuantizationDomain::SqrtPerPredictionVoxel,
        256.0F,
    };
}

FiberletEvaluationQuantization exactFiberletReplayQuantization(
    int storageChunkSideBaseVoxels)
{
    if (storageChunkSideBaseVoxels <= 0)
        throw std::invalid_argument(
            "fiberlet replay storage chunk side must be positive");
    return {
        0.0,
        false,
        0,
        storageChunkSideBaseVoxels,
        FiberletCostQuantizationDomain::RawTotal,
        0.0F,
    };
}

FiberletGeometryCacheProfile fiberletGeometryCacheProfile(
    FiberletEvaluationQuantization replayQuantization)
{
    (void)fiberletPositionBinCountForEvaluation(
        replayQuantization.storageChunkSideBaseVoxels,
        replayQuantization.positionQuantumBaseVoxels);
    const bool quantizedGeometry =
        replayQuantization.positionQuantumBaseVoxels > 0 ||
        replayQuantization.compactDirections;
    return {
        .geometry = {
            replayQuantization.positionQuantumBaseVoxels,
            replayQuantization.compactDirections,
        },
        // Existing generated geometry caches were fingerprinted by the
        // combined u8 scenarios. This is an opaque namespace compatibility
        // tag; it is deliberately absent from FiberletGeometryQuantization.
        .compatibilityCostTagBits = quantizedGeometry ? 8 : 0,
        .storageChunkSideBaseVoxels =
            replayQuantization.storageChunkSideBaseVoxels,
    };
}

} // namespace vc::fiber_tracer
