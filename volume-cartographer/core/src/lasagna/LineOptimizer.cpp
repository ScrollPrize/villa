#include "vc/lasagna/LineOptimizer.hpp"

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <future>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;
constexpr double kControlSpanInitPriorWeight = 0.05;
constexpr int kLocalControlOptimizationSegments = 3;
constexpr double kMovedControlDistanceThreshold = 1.0e-6;

[[nodiscard]] double length(const cv::Vec3d& v)
{
    return std::sqrt(v.dot(v));
}

[[nodiscard]] cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    const double len = length(v);
    if (len <= kEpsilon || !std::isfinite(len)) {
        return {0.0, 0.0, 0.0};
    }
    return v / len;
}

[[nodiscard]] cv::Vec3d deterministicTangentFromNormal(const NormalSample& sample)
{
    if (!sample.valid) {
        return {1.0, 0.0, 0.0};
    }

    const cv::Vec3d normal = normalizedOrZero(sample.normal);
    if (length(normal) <= kEpsilon) {
        return {1.0, 0.0, 0.0};
    }

    const cv::Vec3d reference = std::abs(normal[0]) < 0.9 ? cv::Vec3d{1.0, 0.0, 0.0}
                                                          : cv::Vec3d{0.0, 1.0, 0.0};
    const cv::Vec3d tangent = normal.cross(reference);
    const cv::Vec3d normalized = normalizedOrZero(tangent);
    if (length(normalized) <= kEpsilon) {
        return {1.0, 0.0, 0.0};
    }
    return normalized;
}

[[nodiscard]] LineOptimizationConfig sanitizedConfig(LineOptimizationConfig config)
{
    config.segmentsPerSide = std::max(1, config.segmentsPerSide);
    config.segmentLength = std::max(kEpsilon, config.segmentLength);
    config.straightnessWeight = std::max(0.0, config.straightnessWeight);
    if (config.tangentStraightnessWeight < 0.0) {
        config.tangentStraightnessWeight = config.straightnessWeight;
    }
    if (config.normalStraightnessWeight < 0.0) {
        config.normalStraightnessWeight = config.straightnessWeight;
    }
    config.tangentStraightnessWeight = std::max(0.0, config.tangentStraightnessWeight);
    config.normalStraightnessWeight = std::max(0.0, config.normalStraightnessWeight);
    config.normalAlignmentWeight = std::max(0.0, config.normalAlignmentWeight);
    config.distanceWeight = std::max(0.0, config.distanceWeight);
    config.initialTangentWeight = std::max(0.0, config.initialTangentWeight);
    config.tangentGuideWeight = std::max(0.0, config.tangentGuideWeight);
    config.samplesPerSegment = std::max(1, config.samplesPerSegment);
    config.maxIterations = std::max(0, config.maxIterations);
    config.numThreads = std::max(1, config.numThreads);
    return config;
}

[[nodiscard]] cv::Vec3d initialTangentFromConfig(
    const NormalSample& seedNormal,
    const LineOptimizationConfig& config)
{
    if (!config.useInitialTangent) {
        return deterministicTangentFromNormal(seedNormal);
    }

    cv::Vec3d tangent = normalizedOrZero(config.initialTangent);
    if (length(tangent) <= kEpsilon) {
        return deterministicTangentFromNormal(seedNormal);
    }

    if (seedNormal.valid) {
        const cv::Vec3d normal = normalizedOrZero(seedNormal.normal);
        if (length(normal) > kEpsilon) {
            tangent = normalizedOrZero(tangent - normal * tangent.dot(normal));
        }
    }

    if (length(tangent) <= kEpsilon) {
        return deterministicTangentFromNormal(seedNormal);
    }
    return tangent;
}

[[nodiscard]] cv::Vec3d lerp(const cv::Vec3d& a, const cv::Vec3d& b, double t)
{
    return a * (1.0 - t) + b * t;
}

[[nodiscard]] std::vector<LineSegmentSamples> sampleSegments(
    const std::vector<std::array<double, 3>>& points,
    const NormalSampler& sampler,
    int sampleIntervals,
    int* validSamples,
    int* invalidSamples)
{
    if (validSamples) {
        *validSamples = 0;
    }
    if (invalidSamples) {
        *invalidSamples = 0;
    }

    std::vector<LineSegmentSamples> segmentSamples;
    if (points.size() < 2) {
        return segmentSamples;
    }

    segmentSamples.reserve(points.size() - 1);
    for (size_t segment = 0; segment + 1 < points.size(); ++segment) {
        const cv::Vec3d a{points[segment][0], points[segment][1], points[segment][2]};
        const cv::Vec3d b{points[segment + 1][0], points[segment + 1][1], points[segment + 1][2]};

        LineSegmentSamples samples;
        samples.samples.reserve(static_cast<size_t>(sampleIntervals) + 1);
        for (int i = 0; i <= sampleIntervals; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(sampleIntervals);
            SegmentNormalSample sample;
            sample.t = t;
            sample.position = lerp(a, b, t);
            sample.sampledNormal = sampler.sampleNormal(sample.position);
            sample.sampledNormal.normal = normalizedOrZero(sample.sampledNormal.normal);
            sample.sampledNormal.valid = sample.sampledNormal.valid && length(sample.sampledNormal.normal) > kEpsilon;

            if (sample.sampledNormal.valid) {
                if (validSamples) {
                    ++(*validSamples);
                }
            }
            else if (invalidSamples) {
                ++(*invalidSamples);
            }

            samples.samples.push_back(std::move(sample));
        }
        segmentSamples.push_back(std::move(samples));
    }
    return segmentSamples;
}

struct DistanceResidual {
    DistanceResidual(double targetLength, double weight)
        : targetLength(targetLength)
        , weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const
    {
        const T dx = b[0] - a[0];
        const T dy = b[1] - a[1];
        const T dz = b[2] - a[2];
        residual[0] = T(weight) *
                      (ceres::sqrt(dx * dx + dy * dy + dz * dz + T(kEpsilon)) - T(targetLength));
        return true;
    }

    double targetLength = 0.0;
    double weight = 1.0;
};

struct EvenStepResidual {
    explicit EvenStepResidual(double weight)
        : weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const a, const T* const b, const T* const c, T* residual) const
    {
        const T abx = b[0] - a[0];
        const T aby = b[1] - a[1];
        const T abz = b[2] - a[2];
        const T bcx = c[0] - b[0];
        const T bcy = c[1] - b[1];
        const T bcz = c[2] - b[2];
        const T ab = ceres::sqrt(abx * abx + aby * aby + abz * abz + T(kEpsilon));
        const T bc = ceres::sqrt(bcx * bcx + bcy * bcy + bcz * bcz + T(kEpsilon));
        residual[0] = T(weight) * (ab - bc);
        return true;
    }

    double weight = 1.0;
};

struct PositionPriorResidual {
    PositionPriorResidual(const cv::Vec3d& target, double weight)
        : target(target)
        , weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const point, T* residuals) const
    {
        residuals[0] = T(weight) * (point[0] - T(target[0]));
        residuals[1] = T(weight) * (point[1] - T(target[1]));
        residuals[2] = T(weight) * (point[2] - T(target[2]));
        return true;
    }

    cv::Vec3d target{0.0, 0.0, 0.0};
    double weight = 1.0;
};

void evaluateSurfaceStraightnessResidual(const cv::Vec3d& normal,
                                         double tangentWeight,
                                         double normalWeight,
                                         double const* const* parameters,
                                         double* residuals,
                                         double** jacobians)
{
    cv::Vec3d curvature;
    for (int axis = 0; axis < 3; ++axis) {
        curvature[axis] = parameters[0][axis] -
                          2.0 * parameters[1][axis] +
                          parameters[2][axis];
    }

    const double normalComponent = curvature.dot(normal);
    const cv::Vec3d tangentComponent = curvature - normal * normalComponent;
    for (int axis = 0; axis < 3; ++axis) {
        residuals[axis] = tangentWeight * tangentComponent[axis];
    }
    residuals[3] = normalWeight * normalComponent;

    if (jacobians) {
        double tangentProjection[3][3];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                tangentProjection[r][c] = (r == c ? 1.0 : 0.0) - normal[r] * normal[c];
            }
        }
        for (int block = 0; block < 3; ++block) {
            if (!jacobians[block]) {
                continue;
            }
            const double coefficient = block == 1 ? -2.0 : 1.0;
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    jacobians[block][r * 3 + c] =
                        coefficient * tangentWeight * tangentProjection[r][c];
                }
            }
            for (int c = 0; c < 3; ++c) {
                jacobians[block][3 * 3 + c] = coefficient * normalWeight * normal[c];
            }
        }
    }
}

struct LiveSurfaceStraightnessResidual final : public ceres::SizedCostFunction<4, 3, 3, 3> {
    LiveSurfaceStraightnessResidual(const NormalSampler& sampler,
                                    double tangentWeight,
                                    double normalWeight)
        : sampler(sampler)
        , tangentWeight(tangentWeight)
        , normalWeight(normalWeight)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const cv::Vec3d point{parameters[1][0], parameters[1][1], parameters[1][2]};
        NormalSample sample = sampler.sampleNormal(point);
        cv::Vec3d normal = normalizedOrZero(sample.normal);
        if (!sample.valid || length(normal) <= kEpsilon) {
            normal = {0.0, 0.0, 1.0};
        }

        evaluateSurfaceStraightnessResidual(normal,
                                            tangentWeight,
                                            normalWeight,
                                            parameters,
                                            residuals,
                                            jacobians);
        return true;
    }

    const NormalSampler& sampler;
    double tangentWeight = 1.0;
    double normalWeight = 1.0;
};

enum class SegmentSpacingMode {
    None,
    FixedStep,
    EvenStep,
};

struct SegmentSpacingConstraint {
    SegmentSpacingMode mode = SegmentSpacingMode::FixedStep;
    int spanId = -1;
};

struct LiveNormalAlignmentResidual final : public ceres::SizedCostFunction<1, 3, 3> {
    LiveNormalAlignmentResidual(const NormalSampler& sampler, double t, double weight)
        : sampler(sampler)
        , t(t)
        , weight(weight)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const cv::Vec3d a{parameters[0][0], parameters[0][1], parameters[0][2]};
        const cv::Vec3d b{parameters[1][0], parameters[1][1], parameters[1][2]};
        const cv::Vec3d d = b - a;
        const double len = length(d);
        if (len <= kEpsilon) {
            residuals[0] = 0.0;
            if (jacobians) {
                for (int block = 0; block < 2; ++block) {
                    if (jacobians[block]) {
                        std::fill(jacobians[block], jacobians[block] + 3, 0.0);
                    }
                }
            }
            return true;
        }

        NormalSample sample = sampler.sampleNormal(lerp(a, b, t));
        const cv::Vec3d normal = normalizedOrZero(sample.normal);
        if (!sample.valid || length(normal) <= kEpsilon) {
            residuals[0] = 0.0;
            if (jacobians) {
                for (int block = 0; block < 2; ++block) {
                    if (jacobians[block]) {
                        std::fill(jacobians[block], jacobians[block] + 3, 0.0);
                    }
                }
            }
            return true;
        }

        const cv::Vec3d tangent = d / len;
        const double dot = tangent.dot(normal);
        residuals[0] = weight * dot;

        if (jacobians) {
            const cv::Vec3d gradD = (normal - tangent * dot) * (weight / len);
            if (jacobians[0]) {
                jacobians[0][0] = -gradD[0];
                jacobians[0][1] = -gradD[1];
                jacobians[0][2] = -gradD[2];
            }
            if (jacobians[1]) {
                jacobians[1][0] = gradD[0];
                jacobians[1][1] = gradD[1];
                jacobians[1][2] = gradD[2];
            }
        }
        return true;
    }

    const NormalSampler& sampler;
    double t = 0.0;
    double weight = 1.0;
};

struct DifferentiableNormalAlignmentResidual final : public ceres::SizedCostFunction<1, 3, 3> {
    DifferentiableNormalAlignmentResidual(const NormalSampler& sampler, double t, double weight)
        : sampler(sampler)
        , t(t)
        , weight(weight)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const cv::Vec3d a{parameters[0][0], parameters[0][1], parameters[0][2]};
        const cv::Vec3d b{parameters[1][0], parameters[1][1], parameters[1][2]};
        const cv::Vec3d d = b - a;
        const double len = length(d);
        if (len <= kEpsilon) {
            residuals[0] = 0.0;
            if (jacobians) {
                for (int block = 0; block < 2; ++block) {
                    if (jacobians[block]) {
                        std::fill(jacobians[block], jacobians[block] + 3, 0.0);
                    }
                }
            }
            return true;
        }

        const cv::Vec3d samplePoint = lerp(a, b, t);
        const NormalSampleWithDerivative sampled = sampler.sampleNormalWithDerivative(samplePoint);
        const cv::Vec3d normal = normalizedOrZero(sampled.sample.normal);
        if (!sampled.sample.valid || length(normal) <= kEpsilon) {
            residuals[0] = 0.0;
            if (jacobians) {
                for (int block = 0; block < 2; ++block) {
                    if (jacobians[block]) {
                        std::fill(jacobians[block], jacobians[block] + 3, 0.0);
                    }
                }
            }
            return true;
        }

        const cv::Vec3d tangent = d / len;
        const double dot = tangent.dot(normal);
        residuals[0] = weight * dot;

        if (jacobians) {
            const cv::Vec3d gradD = (normal - tangent * dot) * (weight / len);
            cv::Vec3d gradSamplePoint{0.0, 0.0, 0.0};
            if (sampled.hasDerivative) {
                for (int axis = 0; axis < 3; ++axis) {
                    gradSamplePoint[axis] = weight *
                        (tangent[0] * sampled.dNormalDVolume(0, axis) +
                         tangent[1] * sampled.dNormalDVolume(1, axis) +
                         tangent[2] * sampled.dNormalDVolume(2, axis));
                }
            }

            if (jacobians[0]) {
                jacobians[0][0] = -gradD[0] + (1.0 - t) * gradSamplePoint[0];
                jacobians[0][1] = -gradD[1] + (1.0 - t) * gradSamplePoint[1];
                jacobians[0][2] = -gradD[2] + (1.0 - t) * gradSamplePoint[2];
            }
            if (jacobians[1]) {
                jacobians[1][0] = gradD[0] + t * gradSamplePoint[0];
                jacobians[1][1] = gradD[1] + t * gradSamplePoint[1];
                jacobians[1][2] = gradD[2] + t * gradSamplePoint[2];
            }
        }
        return true;
    }

    const NormalSampler& sampler;
    double t = 0.0;
    double weight = 1.0;
};

struct PrefetchedNormalSamples {
    std::vector<NormalSampleWithDerivative> samples;
};

struct PrefetchTiming {
    int calls = 0;
    double chunkPrefetchMs = 0.0;
    double materializeMs = 0.0;
    uint64_t requestedChunks = 0;
    uint64_t chunksRead = 0;
};

struct PrefetchedSurfaceStraightnessResidual final : public ceres::SizedCostFunction<4, 3, 3, 3> {
    PrefetchedSurfaceStraightnessResidual(const PrefetchedNormalSamples& prefetched,
                                          size_t sampleIndex,
                                          double tangentWeight,
                                          double normalWeight)
        : prefetched(prefetched)
        , sampleIndex(sampleIndex)
        , tangentWeight(tangentWeight)
        , normalWeight(normalWeight)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        cv::Vec3d normal{0.0, 0.0, 1.0};
        if (sampleIndex < prefetched.samples.size()) {
            const NormalSampleWithDerivative& sampled = prefetched.samples[sampleIndex];
            const cv::Vec3d sampledNormal = normalizedOrZero(sampled.sample.normal);
            if (sampled.sample.valid && length(sampledNormal) > kEpsilon) {
                normal = sampledNormal;
            }
        }

        evaluateSurfaceStraightnessResidual(normal,
                                            tangentWeight,
                                            normalWeight,
                                            parameters,
                                            residuals,
                                            jacobians);
        return true;
    }

    const PrefetchedNormalSamples& prefetched;
    size_t sampleIndex = 0;
    double tangentWeight = 1.0;
    double normalWeight = 1.0;
};

struct PrefetchedNormalAlignmentResidual final : public ceres::SizedCostFunction<1, 3, 3> {
    PrefetchedNormalAlignmentResidual(const PrefetchedNormalSamples& prefetched, size_t sampleIndex, double t, double weight)
        : prefetched(prefetched)
        , sampleIndex(sampleIndex)
        , t(t)
        , weight(weight)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const cv::Vec3d a{parameters[0][0], parameters[0][1], parameters[0][2]};
        const cv::Vec3d b{parameters[1][0], parameters[1][1], parameters[1][2]};
        const cv::Vec3d d = b - a;
        const double len = length(d);
        const auto zero = [&]() {
            residuals[0] = 0.0;
            if (jacobians) {
                for (int block = 0; block < 2; ++block) {
                    if (jacobians[block]) {
                        std::fill(jacobians[block], jacobians[block] + 3, 0.0);
                    }
                }
            }
            return true;
        };
        if (len <= kEpsilon || sampleIndex >= prefetched.samples.size()) {
            return zero();
        }

        const NormalSampleWithDerivative& sampled = prefetched.samples[sampleIndex];
        const cv::Vec3d normal = normalizedOrZero(sampled.sample.normal);
        if (!sampled.sample.valid || length(normal) <= kEpsilon) {
            return zero();
        }

        const cv::Vec3d tangent = d / len;
        const double dot = tangent.dot(normal);
        residuals[0] = weight * dot;

        if (jacobians) {
            const cv::Vec3d gradD = (normal - tangent * dot) * (weight / len);
            cv::Vec3d gradSamplePoint{0.0, 0.0, 0.0};
            if (sampled.hasDerivative) {
                for (int axis = 0; axis < 3; ++axis) {
                    gradSamplePoint[axis] = weight *
                        (tangent[0] * sampled.dNormalDVolume(0, axis) +
                         tangent[1] * sampled.dNormalDVolume(1, axis) +
                         tangent[2] * sampled.dNormalDVolume(2, axis));
                }
            }

            if (jacobians[0]) {
                jacobians[0][0] = -gradD[0] + (1.0 - t) * gradSamplePoint[0];
                jacobians[0][1] = -gradD[1] + (1.0 - t) * gradSamplePoint[1];
                jacobians[0][2] = -gradD[2] + (1.0 - t) * gradSamplePoint[2];
            }
            if (jacobians[1]) {
                jacobians[1][0] = gradD[0] + t * gradSamplePoint[0];
                jacobians[1][1] = gradD[1] + t * gradSamplePoint[1];
                jacobians[1][2] = gradD[2] + t * gradSamplePoint[2];
            }
        }
        return true;
    }

    const PrefetchedNormalSamples& prefetched;
    size_t sampleIndex = 0;
    double weight = 1.0;
    double t = 0.0;
};

struct DirectionResidual {
    DirectionResidual(cv::Vec3d direction, double weight)
        : direction(normalizedOrZero(direction))
        , weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residuals) const
    {
        const T dx = b[0] - a[0];
        const T dy = b[1] - a[1];
        const T dz = b[2] - a[2];
        const T invLength = T(1.0) / ceres::sqrt(dx * dx + dy * dy + dz * dz + T(kEpsilon));
        const T ux = dx * invLength;
        const T uy = dy * invLength;
        const T uz = dz * invLength;
        residuals[0] = T(weight) * (uy * T(direction[2]) - uz * T(direction[1]));
        residuals[1] = T(weight) * (uz * T(direction[0]) - ux * T(direction[2]));
        residuals[2] = T(weight) * (ux * T(direction[1]) - uy * T(direction[0]));
        return true;
    }

    cv::Vec3d direction{0.0, 0.0, 0.0};
    double weight = 1.0;
};

[[nodiscard]] cv::Vec3d tangentGuideDirection(
    const cv::Vec3d& normal,
    const LineOptimizationConfig& config)
{
    const cv::Vec3d guide = normalizedOrZero(config.tangentGuideVector);
    if (length(guide) <= kEpsilon || length(normal) <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }

    switch (config.tangentGuideMode) {
    case LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane:
        return normalizedOrZero(guide - normal * guide.dot(normal));
    case LineOptimizationConfig::TangentGuideMode::CrossVectorWithNormal:
        return normalizedOrZero(guide.cross(normal));
    case LineOptimizationConfig::TangentGuideMode::None:
        break;
    }
    return {0.0, 0.0, 0.0};
}

struct LiveTangentGuideResidual final : public ceres::SizedCostFunction<3, 3, 3> {
    LiveTangentGuideResidual(const NormalSampler& sampler, double t, LineOptimizationConfig config)
        : sampler(sampler)
        , t(t)
        , config(config)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const cv::Vec3d a{parameters[0][0], parameters[0][1], parameters[0][2]};
        const cv::Vec3d b{parameters[1][0], parameters[1][1], parameters[1][2]};
        const cv::Vec3d d = b - a;
        const double len = length(d);
        const auto zero = [&]() {
            std::fill(residuals, residuals + 3, 0.0);
            if (jacobians) {
                for (int block = 0; block < 2; ++block) {
                    if (jacobians[block]) {
                        std::fill(jacobians[block], jacobians[block] + 9, 0.0);
                    }
                }
            }
            return true;
        };

        if (len <= kEpsilon) {
            return zero();
        }

        NormalSample sample = sampler.sampleNormal(lerp(a, b, t));
        const cv::Vec3d normal = normalizedOrZero(sample.normal);
        if (!sample.valid || length(normal) <= kEpsilon) {
            return zero();
        }

        const cv::Vec3d guide = tangentGuideDirection(normal, config);
        if (length(guide) <= kEpsilon) {
            return zero();
        }

        const cv::Vec3d tangent = d / len;
        const cv::Vec3d residual = (tangent - guide) * config.tangentGuideWeight;
        residuals[0] = residual[0];
        residuals[1] = residual[1];
        residuals[2] = residual[2];

        if (jacobians) {
            const double w = config.tangentGuideWeight / len;
            double projection[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    projection[r][c] = (r == c ? 1.0 : 0.0) - tangent[r] * tangent[c];
                }
            }

            double jd[3][3] = {};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    jd[r][c] = w * projection[r][c];
                }
            }

            if (jacobians[0]) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        jacobians[0][r * 3 + c] = -jd[r][c];
                    }
                }
            }
            if (jacobians[1]) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        jacobians[1][r * 3 + c] = jd[r][c];
                    }
                }
            }
        }
        return true;
    }

    const NormalSampler& sampler;
    double t = 0.0;
    LineOptimizationConfig config;
};

struct PrefetchedTangentGuideResidual final : public ceres::SizedCostFunction<3, 3, 3> {
    PrefetchedTangentGuideResidual(
        const PrefetchedNormalSamples& prefetched,
        size_t sampleIndex,
        LineOptimizationConfig config)
        : prefetched(prefetched)
        , sampleIndex(sampleIndex)
        , config(config)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        const cv::Vec3d a{parameters[0][0], parameters[0][1], parameters[0][2]};
        const cv::Vec3d b{parameters[1][0], parameters[1][1], parameters[1][2]};
        const cv::Vec3d d = b - a;
        const double len = length(d);
        const auto zero = [&]() {
            std::fill(residuals, residuals + 3, 0.0);
            if (jacobians) {
                for (int block = 0; block < 2; ++block) {
                    if (jacobians[block]) {
                        std::fill(jacobians[block], jacobians[block] + 9, 0.0);
                    }
                }
            }
            return true;
        };

        if (len <= kEpsilon || sampleIndex >= prefetched.samples.size()) {
            return zero();
        }

        const NormalSampleWithDerivative& sampled = prefetched.samples[sampleIndex];
        const cv::Vec3d normal = normalizedOrZero(sampled.sample.normal);
        if (!sampled.sample.valid || length(normal) <= kEpsilon) {
            return zero();
        }

        const cv::Vec3d guide = tangentGuideDirection(normal, config);
        if (length(guide) <= kEpsilon) {
            return zero();
        }

        const cv::Vec3d tangent = d / len;
        const cv::Vec3d residual = (tangent - guide) * config.tangentGuideWeight;
        residuals[0] = residual[0];
        residuals[1] = residual[1];
        residuals[2] = residual[2];

        if (jacobians) {
            const double w = config.tangentGuideWeight / len;
            double projection[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    projection[r][c] = (r == c ? 1.0 : 0.0) - tangent[r] * tangent[c];
                }
            }

            double jd[3][3] = {};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    jd[r][c] = w * projection[r][c];
                }
            }

            if (jacobians[0]) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        jacobians[0][r * 3 + c] = -jd[r][c];
                    }
                }
            }
            if (jacobians[1]) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        jacobians[1][r * 3 + c] = jd[r][c];
                    }
                }
            }
        }
        return true;
    }

    const PrefetchedNormalSamples& prefetched;
    size_t sampleIndex = 0;
    LineOptimizationConfig config;
};

struct LossAccumulator {
    std::string name;
    double weight = 0.0;
    int residuals = 0;
    double rawCost = 0.0;
    double weightedCost = 0.0;

    void add(double rawResidual)
    {
        if (!std::isfinite(rawResidual)) {
            rawResidual = 0.0;
        }
        const double weightedResidual = rawResidual * weight;
        rawCost += 0.5 * rawResidual * rawResidual;
        weightedCost += 0.5 * weightedResidual * weightedResidual;
        ++residuals;
    }
};

[[nodiscard]] LineOptimizationLossReport toReport(const LossAccumulator& loss)
{
    return {loss.name, loss.weight, loss.residuals, loss.rawCost, loss.weightedCost};
}

[[nodiscard]] double totalWeightedCost(const std::vector<LineOptimizationLossReport>& losses)
{
    double total = 0.0;
    for (const auto& loss : losses) {
        total += loss.weightedCost;
    }
    return total;
}

[[nodiscard]] int totalResidualCount(const std::vector<LineOptimizationLossReport>& losses)
{
    int total = 0;
    for (const auto& loss : losses) {
        total += loss.residuals;
    }
    return total;
}

[[nodiscard]] double ceresResidualRms(double cost, int residuals)
{
    if (residuals <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(std::max(0.0, 2.0 * cost) / static_cast<double>(residuals));
}

[[nodiscard]] cv::Vec3d straightnessSplitNormalAt(const std::array<double, 3>& point,
                                                  const NormalSampler& sampler)
{
    const NormalSample sample = sampler.sampleNormal({point[0], point[1], point[2]});
    const cv::Vec3d normal = normalizedOrZero(sample.normal);
    if (sample.valid && length(normal) > kEpsilon) {
        return normal;
    }
    return {0.0, 0.0, 1.0};
}

[[nodiscard]] std::vector<LineOptimizationLossReport> evaluateLosses(
    const std::vector<std::array<double, 3>>& points,
    const std::vector<SegmentSpacingConstraint>& spacingConstraints,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    int seedIndex,
    const cv::Vec3d& seedTangent,
    bool useSeedGuides)
{
    LossAccumulator stepDistance{"step_distance", config.distanceWeight};
    LossAccumulator evenStep{"even_step", config.distanceWeight};
    LossAccumulator tangentStraightness{"tangent_straightness", config.tangentStraightnessWeight};
    LossAccumulator normalStraightness{"normal_straightness", config.normalStraightnessWeight};
    LossAccumulator normalAlignment{"normal_alignment", config.normalAlignmentWeight};
    LossAccumulator initialDirection{"initial_direction", config.initialTangentWeight};
    LossAccumulator tangentGuide{"tangent_guide", config.tangentGuideWeight};
    const bool useTangentGuide = useSeedGuides &&
                                 config.tangentGuideMode != LineOptimizationConfig::TangentGuideMode::None;
    const auto addZeroTangentGuide = [&]() {
        if (!useTangentGuide) {
            return;
        }
        tangentGuide.add(0.0);
        tangentGuide.add(0.0);
        tangentGuide.add(0.0);
    };

    for (size_t i = 0; i + 1 < points.size(); ++i) {
        const cv::Vec3d a{points[i][0], points[i][1], points[i][2]};
        const cv::Vec3d b{points[i + 1][0], points[i + 1][1], points[i + 1][2]};
        const SegmentSpacingConstraint spacing = i < spacingConstraints.size()
            ? spacingConstraints[i]
            : SegmentSpacingConstraint{};
        if (spacing.mode == SegmentSpacingMode::FixedStep) {
            stepDistance.add(std::sqrt((b - a).dot(b - a) + kEpsilon) - config.segmentLength);
        }
    }

    for (size_t i = 0; i + 2 < points.size(); ++i) {
        const SegmentSpacingConstraint left = i < spacingConstraints.size()
            ? spacingConstraints[i]
            : SegmentSpacingConstraint{};
        const SegmentSpacingConstraint right = i + 1 < spacingConstraints.size()
            ? spacingConstraints[i + 1]
            : SegmentSpacingConstraint{};
        if (left.mode != SegmentSpacingMode::EvenStep ||
            right.mode != SegmentSpacingMode::EvenStep ||
            left.spanId != right.spanId) {
            continue;
        }
        const cv::Vec3d a{points[i][0], points[i][1], points[i][2]};
        const cv::Vec3d b{points[i + 1][0], points[i + 1][1], points[i + 1][2]};
        const cv::Vec3d c{points[i + 2][0], points[i + 2][1], points[i + 2][2]};
        evenStep.add(std::sqrt((b - a).dot(b - a) + kEpsilon) -
                     std::sqrt((c - b).dot(c - b) + kEpsilon));
    }

    for (size_t i = 1; i + 1 < points.size(); ++i) {
        const cv::Vec3d normal = straightnessSplitNormalAt(points[i], sampler);
        const cv::Vec3d curvature{
            points[i - 1][0] - 2.0 * points[i][0] + points[i + 1][0],
            points[i - 1][1] - 2.0 * points[i][1] + points[i + 1][1],
            points[i - 1][2] - 2.0 * points[i][2] + points[i + 1][2],
        };
        const double normalComponent = curvature.dot(normal);
        const cv::Vec3d tangentComponent = curvature - normal * normalComponent;
        for (int axis = 0; axis < 3; ++axis) {
            tangentStraightness.add(tangentComponent[axis]);
        }
        normalStraightness.add(normalComponent);
    }

    for (size_t segment = 0; segment + 1 < points.size(); ++segment) {
        const bool seedAdjacent = static_cast<int>(segment) == seedIndex ||
                                  static_cast<int>(segment + 1) == seedIndex;
        const cv::Vec3d a{points[segment][0], points[segment][1], points[segment][2]};
        const cv::Vec3d b{points[segment + 1][0], points[segment + 1][1], points[segment + 1][2]};
        const cv::Vec3d d = b - a;
        const double len = length(d);
        for (int sample = 0; sample <= config.samplesPerSegment; ++sample) {
            if (len <= kEpsilon) {
                normalAlignment.add(0.0);
                if (seedAdjacent) {
                    addZeroTangentGuide();
                }
                continue;
            }
            const double t = static_cast<double>(sample) /
                             static_cast<double>(config.samplesPerSegment);
            const NormalSample normalSample = sampler.sampleNormal(lerp(a, b, t));
            const cv::Vec3d normal = normalizedOrZero(normalSample.normal);
            if (!normalSample.valid || length(normal) <= kEpsilon) {
                normalAlignment.add(0.0);
                if (seedAdjacent) {
                    addZeroTangentGuide();
                }
                continue;
            }
            const cv::Vec3d tangent = d / len;
            normalAlignment.add(tangent.dot(normal));
            if (seedAdjacent && useTangentGuide) {
                const cv::Vec3d guide = tangentGuideDirection(normal, config);
                if (length(guide) <= kEpsilon) {
                    addZeroTangentGuide();
                } else {
                    const cv::Vec3d residual = tangent - guide;
                    tangentGuide.add(residual[0]);
                    tangentGuide.add(residual[1]);
                    tangentGuide.add(residual[2]);
                }
            }
        }
    }

    if (useSeedGuides &&
        config.useInitialTangent &&
        config.initialTangentWeight > 0.0 &&
        length(seedTangent) > kEpsilon &&
        seedIndex > 0 &&
        seedIndex + 1 < static_cast<int>(points.size())) {
        const cv::Vec3d direction = normalizedOrZero(seedTangent);
        const auto addDirection = [&](const std::array<double, 3>& p0,
                                      const std::array<double, 3>& p1) {
            const cv::Vec3d d{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
            const double len = std::sqrt(d.dot(d) + kEpsilon);
            const cv::Vec3d unit = d * (1.0 / len);
            const cv::Vec3d cross = unit.cross(direction);
            initialDirection.add(cross[0]);
            initialDirection.add(cross[1]);
            initialDirection.add(cross[2]);
        };
        addDirection(points[seedIndex - 1], points[seedIndex]);
        addDirection(points[seedIndex], points[seedIndex + 1]);
    }

    std::vector<LineOptimizationLossReport> losses;
    losses.reserve(7);
    losses.push_back(toReport(stepDistance));
    losses.push_back(toReport(evenStep));
    losses.push_back(toReport(tangentStraightness));
    losses.push_back(toReport(normalStraightness));
    losses.push_back(toReport(normalAlignment));
    losses.push_back(toReport(initialDirection));
    losses.push_back(toReport(tangentGuide));
    return losses;
}

struct SpanDeviationMetrics {
    double maxEvenStepDeviation = 0.0;
    double maxTangentSmoothDeviation = 0.0;
    double maxNormalSmoothDeviation = 0.0;
    double avgNormalAlignmentAbs = 0.0;
    double p95NormalAlignmentAbs = 0.0;
    double maxNormalAlignmentAbs = 0.0;
    int normalAlignmentSamples = 0;
};

[[nodiscard]] double percentile(std::vector<double> values, double fraction)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    fraction = std::clamp(fraction, 0.0, 1.0);
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper) {
        return values[lower];
    }
    const double t = position - static_cast<double>(lower);
    return values[lower] * (1.0 - t) + values[upper] * t;
}

[[nodiscard]] double alignmentChoiceScore(const SpanDeviationMetrics& metrics)
{
    if (metrics.normalAlignmentSamples <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    return (metrics.avgNormalAlignmentAbs +
            metrics.p95NormalAlignmentAbs +
            metrics.maxNormalAlignmentAbs) / 3.0;
}

enum class SpanCandidateAnchorSide {
    Left,
    Right,
};

[[nodiscard]] double endpointDirectionDot(
    const std::vector<std::array<double, 3>>& points,
    SpanCandidateAnchorSide anchorSide,
    const cv::Vec3d& referenceDirection)
{
    if (points.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const cv::Vec3d reference = normalizedOrZero(referenceDirection);
    if (length(reference) <= kEpsilon) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto point = [](const std::array<double, 3>& value) {
        return cv::Vec3d{value[0], value[1], value[2]};
    };
    const cv::Vec3d startDirection =
        normalizedOrZero(point(points[1]) - point(points[0]));
    const cv::Vec3d endDirection =
        normalizedOrZero(point(points[points.size() - 2]) - point(points.back()));
    const cv::Vec3d connectedDirection =
        anchorSide == SpanCandidateAnchorSide::Left ? startDirection : endDirection;
    if (length(connectedDirection) <= kEpsilon) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return connectedDirection.dot(reference);
}

[[nodiscard]] SpanDeviationMetrics spanDeviationMetrics(
    const std::vector<std::array<double, 3>>& points,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    SpanDeviationMetrics metrics;
    for (size_t i = 0; i + 2 < points.size(); ++i) {
        const cv::Vec3d a{points[i][0], points[i][1], points[i][2]};
        const cv::Vec3d b{points[i + 1][0], points[i + 1][1], points[i + 1][2]};
        const cv::Vec3d c{points[i + 2][0], points[i + 2][1], points[i + 2][2]};
        const double leftLength = length(b - a);
        const double rightLength = length(c - b);
        metrics.maxEvenStepDeviation =
            std::max(metrics.maxEvenStepDeviation, std::abs(leftLength - rightLength));

        const cv::Vec3d normal = straightnessSplitNormalAt(points[i + 1], sampler);
        const cv::Vec3d curvature = a - b * 2.0 + c;
        const double normalComponent = curvature.dot(normal);
        const cv::Vec3d tangentComponent = curvature - normal * normalComponent;
        metrics.maxTangentSmoothDeviation =
            std::max(metrics.maxTangentSmoothDeviation, length(tangentComponent));
        metrics.maxNormalSmoothDeviation =
            std::max(metrics.maxNormalSmoothDeviation, std::abs(normalComponent));
    }

    double normalAlignmentAbsSum = 0.0;
    std::vector<double> normalAlignmentAbsValues;
    normalAlignmentAbsValues.reserve(points.size() * static_cast<size_t>(config.samplesPerSegment + 1));
    for (size_t segment = 0; segment + 1 < points.size(); ++segment) {
        const cv::Vec3d a{points[segment][0], points[segment][1], points[segment][2]};
        const cv::Vec3d b{points[segment + 1][0], points[segment + 1][1], points[segment + 1][2]};
        const cv::Vec3d d = b - a;
        const double len = length(d);
        if (len <= kEpsilon) {
            continue;
        }
        const cv::Vec3d tangent = d / len;
        for (int sample = 0; sample <= config.samplesPerSegment; ++sample) {
            const double t = static_cast<double>(sample) /
                             static_cast<double>(config.samplesPerSegment);
            const NormalSample normalSample = sampler.sampleNormal(lerp(a, b, t));
            const cv::Vec3d normal = normalizedOrZero(normalSample.normal);
            if (!normalSample.valid || length(normal) <= kEpsilon) {
                continue;
            }
            const double alignmentAbs = std::abs(tangent.dot(normal));
            normalAlignmentAbsSum += alignmentAbs;
            normalAlignmentAbsValues.push_back(alignmentAbs);
            metrics.maxNormalAlignmentAbs =
                std::max(metrics.maxNormalAlignmentAbs, alignmentAbs);
        }
    }
    metrics.normalAlignmentSamples = static_cast<int>(normalAlignmentAbsValues.size());
    if (metrics.normalAlignmentSamples > 0) {
        metrics.avgNormalAlignmentAbs =
            normalAlignmentAbsSum / static_cast<double>(metrics.normalAlignmentSamples);
        metrics.p95NormalAlignmentAbs = percentile(std::move(normalAlignmentAbsValues), 0.95);
    }
    return metrics;
}

[[nodiscard]] std::vector<LineOptimizationIterationReport> iterationReports(
    const ceres::Solver::Summary& summary)
{
    std::vector<LineOptimizationIterationReport> reports;
    reports.reserve(summary.iterations.size());
    for (const auto& iteration : summary.iterations) {
        reports.push_back({
            iteration.iteration,
            iteration.cost,
            iteration.cost_change,
            iteration.gradient_max_norm,
            iteration.step_norm,
            iteration.trust_region_radius,
            iteration.linear_solver_iterations,
            iteration.step_is_successful,
        });
    }
    return reports;
}

void addResiduals(
    ceres::Problem& problem,
    std::vector<std::array<double, 3>>& points,
    const std::vector<SegmentSpacingConstraint>& spacingConstraints,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    int seedIndex,
    const cv::Vec3d& seedTangent,
    bool useSeedGuides,
    const PrefetchedNormalSamples* prefetchedSamples = nullptr,
    int activeStart = 0,
    int activeEnd = -1)
{
    if (points.empty()) {
        return;
    }
    activeStart = std::clamp(activeStart, 0, static_cast<int>(points.size()) - 1);
    activeEnd = activeEnd < 0
        ? static_cast<int>(points.size()) - 1
        : std::clamp(activeEnd, activeStart, static_cast<int>(points.size()) - 1);

    for (int i = activeStart; i + 1 <= activeEnd; ++i) {
        const SegmentSpacingConstraint spacing = i < spacingConstraints.size()
            ? spacingConstraints[static_cast<size_t>(i)]
            : SegmentSpacingConstraint{};
        if (spacing.mode != SegmentSpacingMode::FixedStep) {
            continue;
        }
        auto* cost = new ceres::AutoDiffCostFunction<DistanceResidual, 1, 3, 3>(
            new DistanceResidual(config.segmentLength, config.distanceWeight));
        problem.AddResidualBlock(cost, nullptr,
                                 points[static_cast<size_t>(i)].data(),
                                 points[static_cast<size_t>(i + 1)].data());
    }

    for (int i = activeStart; i + 2 <= activeEnd; ++i) {
        const SegmentSpacingConstraint left = i < spacingConstraints.size()
            ? spacingConstraints[static_cast<size_t>(i)]
            : SegmentSpacingConstraint{};
        const SegmentSpacingConstraint right = i + 1 < spacingConstraints.size()
            ? spacingConstraints[static_cast<size_t>(i + 1)]
            : SegmentSpacingConstraint{};
        if (left.mode != SegmentSpacingMode::EvenStep ||
            right.mode != SegmentSpacingMode::EvenStep ||
            left.spanId != right.spanId) {
            continue;
        }
        auto* cost = new ceres::AutoDiffCostFunction<EvenStepResidual, 1, 3, 3, 3>(
            new EvenStepResidual(config.distanceWeight));
        problem.AddResidualBlock(cost, nullptr,
                                 points[static_cast<size_t>(i)].data(),
                                 points[static_cast<size_t>(i + 1)].data(),
                                 points[static_cast<size_t>(i + 2)].data());
    }

    for (int i = std::max(activeStart + 1, 1); i + 1 <= activeEnd; ++i) {
        if (config.tangentStraightnessWeight <= 0.0 && config.normalStraightnessWeight <= 0.0) {
            continue;
        }
        const size_t sampleIndex = static_cast<size_t>(i - 1) *
                                       static_cast<size_t>(config.samplesPerSegment + 1) +
                                   static_cast<size_t>(config.samplesPerSegment);
        ceres::CostFunction* cost = prefetchedSamples
            ? static_cast<ceres::CostFunction*>(
                  new PrefetchedSurfaceStraightnessResidual(*prefetchedSamples,
                                                            sampleIndex,
                                                            config.tangentStraightnessWeight,
                                                            config.normalStraightnessWeight))
            : static_cast<ceres::CostFunction*>(
                  new LiveSurfaceStraightnessResidual(sampler,
                                                      config.tangentStraightnessWeight,
                                                      config.normalStraightnessWeight));
        problem.AddResidualBlock(cost, nullptr,
                                 points[static_cast<size_t>(i - 1)].data(),
                                 points[static_cast<size_t>(i)].data(),
                                 points[static_cast<size_t>(i + 1)].data());
    }

    for (int segment = activeStart; segment + 1 <= activeEnd; ++segment) {
        for (int sample = 0; sample <= config.samplesPerSegment; ++sample) {
            const double t = static_cast<double>(sample) /
                             static_cast<double>(config.samplesPerSegment);
            const size_t sampleIndex = static_cast<size_t>(segment) *
                                           static_cast<size_t>(config.samplesPerSegment + 1) +
                                       static_cast<size_t>(sample);
            ceres::CostFunction* cost = prefetchedSamples
                ? static_cast<ceres::CostFunction*>(
                      new PrefetchedNormalAlignmentResidual(*prefetchedSamples, sampleIndex, t, config.normalAlignmentWeight))
                : (config.differentiableNormalSampling
                       ? static_cast<ceres::CostFunction*>(
                             new DifferentiableNormalAlignmentResidual(sampler, t, config.normalAlignmentWeight))
                       : static_cast<ceres::CostFunction*>(
                             new LiveNormalAlignmentResidual(sampler, t, config.normalAlignmentWeight)));
            problem.AddResidualBlock(cost, nullptr,
                                     points[static_cast<size_t>(segment)].data(),
                                     points[static_cast<size_t>(segment + 1)].data());

            const bool seedAdjacent = static_cast<int>(segment) == seedIndex ||
                                      static_cast<int>(segment + 1) == seedIndex;
            if (useSeedGuides &&
                seedAdjacent &&
                config.tangentGuideMode != LineOptimizationConfig::TangentGuideMode::None &&
                config.tangentGuideWeight > 0.0) {
                ceres::CostFunction* guideCost = prefetchedSamples
                    ? static_cast<ceres::CostFunction*>(
                          new PrefetchedTangentGuideResidual(*prefetchedSamples, sampleIndex, config))
                    : static_cast<ceres::CostFunction*>(
                          new LiveTangentGuideResidual(sampler, t, config));
                problem.AddResidualBlock(guideCost, nullptr, points[segment].data(), points[segment + 1].data());
            }
        }
    }

    if (useSeedGuides &&
        config.useInitialTangent &&
        config.initialTangentWeight > 0.0 &&
        length(seedTangent) > kEpsilon &&
        seedIndex > activeStart &&
        seedIndex + 1 <= activeEnd) {
        auto* prevCost = new ceres::AutoDiffCostFunction<DirectionResidual, 3, 3, 3>(
            new DirectionResidual(seedTangent, config.initialTangentWeight));
        problem.AddResidualBlock(prevCost, nullptr,
                                 points[static_cast<size_t>(seedIndex - 1)].data(),
                                 points[static_cast<size_t>(seedIndex)].data());

        auto* nextCost = new ceres::AutoDiffCostFunction<DirectionResidual, 3, 3, 3>(
            new DirectionResidual(seedTangent, config.initialTangentWeight));
        problem.AddResidualBlock(nextCost, nullptr,
                                 points[static_cast<size_t>(seedIndex)].data(),
                                 points[static_cast<size_t>(seedIndex + 1)].data());
    }
}

void addSingleSegmentResiduals(ceres::Problem& problem,
                               std::array<double, 3>* previous,
                               std::array<double, 3>& fixed,
                               std::array<double, 3>& moving,
                               const NormalSampler& sampler,
                               const LineOptimizationConfig& config,
                               bool useTangentGuide)
{
    auto* distanceCost = new ceres::AutoDiffCostFunction<DistanceResidual, 1, 3, 3>(
        new DistanceResidual(config.segmentLength, config.distanceWeight));
    problem.AddResidualBlock(distanceCost, nullptr, fixed.data(), moving.data());

    if (previous != nullptr &&
        (config.tangentStraightnessWeight > 0.0 || config.normalStraightnessWeight > 0.0)) {
        auto* straightnessCost = new LiveSurfaceStraightnessResidual(sampler,
                                                                     config.tangentStraightnessWeight,
                                                                     config.normalStraightnessWeight);
        problem.AddResidualBlock(straightnessCost, nullptr, previous->data(), fixed.data(), moving.data());
    }

    for (int sample = 0; sample <= config.samplesPerSegment; ++sample) {
        const double t = static_cast<double>(sample) /
                         static_cast<double>(config.samplesPerSegment);
        ceres::CostFunction* normalCost = config.differentiableNormalSampling
            ? static_cast<ceres::CostFunction*>(
                  new DifferentiableNormalAlignmentResidual(sampler, t, config.normalAlignmentWeight))
            : static_cast<ceres::CostFunction*>(
                  new LiveNormalAlignmentResidual(sampler, t, config.normalAlignmentWeight));
        problem.AddResidualBlock(normalCost, nullptr, fixed.data(), moving.data());

        if (useTangentGuide &&
            config.tangentGuideMode != LineOptimizationConfig::TangentGuideMode::None &&
            config.tangentGuideWeight > 0.0) {
            auto* guideCost = new LiveTangentGuideResidual(sampler, t, config);
            problem.AddResidualBlock(guideCost, nullptr, fixed.data(), moving.data());
        }
    }
}

ceres::Solver::Options solverOptions(const LineOptimizationConfig& config, bool progress)
{
    ceres::Solver::Options options;
    options.max_num_iterations = config.maxIterations;
    switch (config.linearSolver) {
    case LineOptimizationConfig::LinearSolver::DenseQR:
        options.linear_solver_type = ceres::DENSE_QR;
        break;
    case LineOptimizationConfig::LinearSolver::DenseNormalCholesky:
        options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        break;
    case LineOptimizationConfig::LinearSolver::SparseNormalCholesky:
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        break;
    case LineOptimizationConfig::LinearSolver::DenseSchur:
        options.linear_solver_type = ceres::DENSE_SCHUR;
        break;
    case LineOptimizationConfig::LinearSolver::SparseSchur:
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        break;
    case LineOptimizationConfig::LinearSolver::IterativeSchur:
        options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        break;
    case LineOptimizationConfig::LinearSolver::CGNR:
        options.linear_solver_type = ceres::CGNR;
        break;
    }
    options.minimizer_progress_to_stdout = progress;
    options.logging_type = progress ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
    options.num_threads = config.numThreads;
    return options;
}

[[nodiscard]] std::array<double, 3> toArray(const cv::Vec3d& point)
{
    return {point[0], point[1], point[2]};
}

[[nodiscard]] cv::Vec3d toVec3d(const std::array<double, 3>& point)
{
    return {point[0], point[1], point[2]};
}

[[nodiscard]] cv::Vec3d guidedTangentAtPoint(
    const cv::Vec3d& point,
    const cv::Vec3d& fallbackTangent,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    bool allowFlip)
{
    const NormalSample sample = sampler.sampleNormal(point);
    const cv::Vec3d normal = normalizedOrZero(sample.normal);
    cv::Vec3d tangent{0.0, 0.0, 0.0};
    if (sample.valid && length(normal) > kEpsilon &&
        config.tangentGuideMode != LineOptimizationConfig::TangentGuideMode::None) {
        tangent = tangentGuideDirection(normal, config);
    }
    if (length(tangent) <= kEpsilon) {
        tangent = normalizedOrZero(fallbackTangent);
    }
    const cv::Vec3d fallback = normalizedOrZero(fallbackTangent);
    if (allowFlip && length(tangent) > kEpsilon && length(fallback) > kEpsilon &&
        tangent.dot(fallback) < 0.0) {
        tangent *= -1.0;
    }
    return tangent;
}

struct SequentialSolveResult {
    std::vector<std::array<double, 3>> points;
    double initialCost = 0.0;
    double finalCost = 0.0;
    int iterations = 0;
    bool usable = true;
    std::string report;
};

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

[[nodiscard]] SequentialSolveResult growSequentialLine(
    const cv::Vec3d& seedPoint,
    const cv::Vec3d& seedTangent,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    SequentialSolveResult result;
    std::vector<std::array<double, 3>> backward;
    std::vector<std::array<double, 3>> forward;
    backward.reserve(static_cast<size_t>(config.segmentsPerSide));
    forward.reserve(static_cast<size_t>(config.segmentsPerSide));

    auto solveStep = [&](const cv::Vec3d& fixedPoint,
                         const std::optional<cv::Vec3d>& previousPoint,
                         const cv::Vec3d& initialDirection,
                         int sign,
                         bool useTangentGuide,
                         const char* label,
                         int step) {
        std::array<double, 3> previous;
        std::array<double, 3>* previousPtr = nullptr;
        if (previousPoint.has_value()) {
            previous = toArray(*previousPoint);
            previousPtr = &previous;
        }
        std::array<double, 3> fixed = toArray(fixedPoint);
        const cv::Vec3d guide = guidedTangentAtPoint(fixedPoint, initialDirection, sampler, config, true);
        const cv::Vec3d signedGuide = guide * static_cast<double>(sign);
        std::array<double, 3> moving = toArray(fixedPoint + signedGuide * config.segmentLength);

        LineOptimizationConfig stepConfig = config;
        stepConfig.tangentGuideVector = normalizedOrZero(signedGuide);
        stepConfig.tangentGuideMode = LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane;

        ceres::Problem problem;
        if (previousPtr) {
            problem.AddParameterBlock(previousPtr->data(), 3);
            problem.SetParameterBlockConstant(previousPtr->data());
        }
        problem.AddParameterBlock(fixed.data(), 3);
        problem.AddParameterBlock(moving.data(), 3);
        problem.SetParameterBlockConstant(fixed.data());
        addSingleSegmentResiduals(problem, previousPtr, fixed, moving, sampler, stepConfig, useTangentGuide);

        double initialCost = 0.0;
        problem.Evaluate(ceres::Problem::EvaluateOptions{}, &initialCost, nullptr, nullptr, nullptr);

        ceres::Solver::Summary summary;
        ceres::Solve(solverOptions(config, false), &problem, &summary);

        result.initialCost += initialCost;
        result.finalCost += summary.final_cost;
        result.iterations += static_cast<int>(summary.iterations.size());
        result.usable = result.usable && summary.IsSolutionUsable();
        result.report += std::string(label) + " step " + std::to_string(step) + ":\n" +
                         summary.FullReport() + "\n";
        return toVec3d(moving);
    };

    const cv::Vec3d seedGuide = guidedTangentAtPoint(seedPoint, seedTangent, sampler, config, true);

    cv::Vec3d prevPoint = seedPoint;
    std::optional<cv::Vec3d> prevPrevPoint;
    cv::Vec3d prevDirection = seedGuide;
    for (int step = 0; step < config.segmentsPerSide; ++step) {
        const cv::Vec3d next = solveStep(prevPoint,
                                         prevPrevPoint,
                                         prevDirection,
                                         1,
                                         step == 0,
                                         "forward",
                                         step + 1);
        forward.push_back(toArray(next));
        prevDirection = normalizedOrZero(next - prevPoint);
        prevPrevPoint = prevPoint;
        prevPoint = next;
    }

    prevPoint = seedPoint;
    prevPrevPoint = std::nullopt;
    prevDirection = seedGuide;
    for (int step = 0; step < config.segmentsPerSide; ++step) {
        const cv::Vec3d next = solveStep(prevPoint,
                                         prevPrevPoint,
                                         prevDirection,
                                         -1,
                                         step == 0,
                                         "backward",
                                         step + 1);
        backward.push_back(toArray(next));
        prevDirection = normalizedOrZero(prevPoint - next);
        prevPrevPoint = prevPoint;
        prevPoint = next;
    }

    result.points.reserve(backward.size() + 1 + forward.size());
    for (auto it = backward.rbegin(); it != backward.rend(); ++it) {
        result.points.push_back(*it);
    }
    result.points.push_back(toArray(seedPoint));
    result.points.insert(result.points.end(), forward.begin(), forward.end());
    return result;
}

[[nodiscard]] std::vector<std::array<double, 3>> straightLinePoints(
    const cv::Vec3d& seedPoint,
    const cv::Vec3d& tangent,
    const LineOptimizationConfig& config)
{
    const int pointCount = config.segmentsPerSide * 2 + 1;
    const int seedIndex = config.segmentsPerSide;
    std::vector<std::array<double, 3>> points;
    points.reserve(static_cast<size_t>(pointCount));
    for (int i = 0; i < pointCount; ++i) {
        const cv::Vec3d point = seedPoint +
            tangent * (static_cast<double>(i - seedIndex) * config.segmentLength);
        points.push_back(toArray(point));
    }
    return points;
}

[[nodiscard]] cv::Vec3d projectDirectionToNormalPlane(
    const cv::Vec3d& direction,
    const cv::Vec3d& normal)
{
    cv::Vec3d projected = direction - normal * direction.dot(normal);
    projected = normalizedOrZero(projected);
    const cv::Vec3d normalizedDirection = normalizedOrZero(direction);
    if (length(projected) > kEpsilon &&
        length(normalizedDirection) > kEpsilon &&
        projected.dot(normalizedDirection) < 0.0) {
        projected *= -1.0;
    }
    if (length(projected) <= kEpsilon) {
        return normalizedDirection;
    }
    return projected;
}

[[nodiscard]] cv::Vec3d rotateAroundAxis(const cv::Vec3d& vector,
                                         const cv::Vec3d& axis,
                                         double angle)
{
    const cv::Vec3d unitAxis = normalizedOrZero(axis);
    if (length(unitAxis) <= kEpsilon) {
        return vector;
    }
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return vector * c + unitAxis.cross(vector) * s +
           unitAxis * (unitAxis.dot(vector) * (1.0 - c));
}

[[nodiscard]] cv::Vec3d transportedDirectionToNormalPlane(
    const cv::Vec3d& direction,
    const cv::Vec3d& previousNormal,
    cv::Vec3d targetNormal)
{
    const cv::Vec3d normalizedDirection = normalizedOrZero(direction);
    if (length(normalizedDirection) <= kEpsilon) {
        return {1.0, 0.0, 0.0};
    }

    cv::Vec3d previous = normalizedOrZero(previousNormal);
    targetNormal = normalizedOrZero(targetNormal);
    if (length(targetNormal) <= kEpsilon) {
        return normalizedDirection;
    }
    if (length(previous) <= kEpsilon) {
        return projectDirectionToNormalPlane(normalizedDirection, targetNormal);
    }

    if (previous.dot(targetNormal) < 0.0) {
        targetNormal *= -1.0;
    }

    const cv::Vec3d axis = previous.cross(targetNormal);
    const double sinAngle = length(axis);
    const double cosAngle = std::clamp(previous.dot(targetNormal), -1.0, 1.0);
    cv::Vec3d transported = normalizedDirection;
    if (sinAngle > kEpsilon) {
        transported = rotateAroundAxis(normalizedDirection,
                                       axis,
                                       std::atan2(sinAngle, cosAngle));
    }

    transported = projectDirectionToNormalPlane(transported, targetNormal);
    if (length(transported) <= kEpsilon) {
        return normalizedDirection;
    }
    return transported;
}

[[nodiscard]] std::vector<std::array<double, 3>> directNormalConstructedPoints(
    const cv::Vec3d& seedPoint,
    const cv::Vec3d& seedTangent,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    std::vector<std::array<double, 3>> backward;
    std::vector<std::array<double, 3>> forward;
    backward.reserve(static_cast<size_t>(config.segmentsPerSide));
    forward.reserve(static_cast<size_t>(config.segmentsPerSide));

    auto grow = [&](int sign, std::vector<std::array<double, 3>>& out) {
        cv::Vec3d point = seedPoint;
        cv::Vec3d direction = normalizedOrZero(seedTangent) * static_cast<double>(sign);
        NormalSample previousSample = sampler.sampleNormal(point);
        cv::Vec3d previousNormal = normalizedOrZero(previousSample.normal);
        if (!previousSample.valid || length(previousNormal) <= kEpsilon) {
            previousNormal = {0.0, 0.0, 0.0};
        }
        for (int i = 0; i < config.segmentsPerSide; ++i) {
            const cv::Vec3d predicted = point + direction * config.segmentLength;
            const NormalSample sample = sampler.sampleNormal(predicted);
            cv::Vec3d normal = normalizedOrZero(sample.normal);
            if (sample.valid && length(normal) > kEpsilon) {
                direction = transportedDirectionToNormalPlane(direction, previousNormal, normal);
                if (length(previousNormal) > kEpsilon && previousNormal.dot(normal) < 0.0) {
                    normal *= -1.0;
                }
                previousNormal = normal;
            }
            point += direction * config.segmentLength;
            out.push_back(toArray(point));
        }
    };

    grow(1, forward);
    grow(-1, backward);

    std::vector<std::array<double, 3>> points;
    points.reserve(backward.size() + 1 + forward.size());
    for (auto it = backward.rbegin(); it != backward.rend(); ++it) {
        points.push_back(*it);
    }
    points.push_back(toArray(seedPoint));
    points.insert(points.end(), forward.begin(), forward.end());
    return points;
}

[[nodiscard]] std::vector<SegmentSpacingConstraint> fixedStepConstraints(size_t segmentCount)
{
    return std::vector<SegmentSpacingConstraint>(segmentCount, SegmentSpacingConstraint{});
}

void growNormalConstructedExtension(
    const cv::Vec3d& startPoint,
    cv::Vec3d direction,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    std::vector<std::array<double, 3>>& out);

[[nodiscard]] std::vector<SegmentSpacingConstraint> evenStepSpanConstraints(size_t segmentCount, int spanId)
{
    return std::vector<SegmentSpacingConstraint>(
        segmentCount,
        SegmentSpacingConstraint{SegmentSpacingMode::EvenStep, spanId});
}

[[nodiscard]] double polylineLength(const std::vector<cv::Vec3d>& points)
{
    double total = 0.0;
    for (size_t i = 1; i < points.size(); ++i) {
        total += length(points[i] - points[i - 1]);
    }
    return total;
}

[[nodiscard]] std::vector<cv::Vec3d> resamplePolylineByArcLength(
    const std::vector<cv::Vec3d>& points,
    double segmentLength)
{
    if (points.size() < 2) {
        return points;
    }

    const double totalLength = polylineLength(points);
    const int segments = std::max(1, static_cast<int>(std::llround(totalLength / segmentLength)));
    std::vector<cv::Vec3d> resampled;
    resampled.reserve(static_cast<size_t>(segments + 1));
    if (totalLength <= kEpsilon) {
        for (int segment = 0; segment <= segments; ++segment) {
            const double t = static_cast<double>(segment) / static_cast<double>(segments);
            resampled.push_back(lerp(points.front(), points.back(), t));
        }
        return resampled;
    }

    std::vector<double> cumulative(points.size(), 0.0);
    for (size_t i = 1; i < points.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + length(points[i] - points[i - 1]);
    }
    for (int segment = 0; segment <= segments; ++segment) {
        const double target = totalLength * static_cast<double>(segment) /
                              static_cast<double>(segments);
        auto upper = std::lower_bound(cumulative.begin(), cumulative.end(), target);
        if (upper == cumulative.begin()) {
            resampled.push_back(points.front());
        } else if (upper == cumulative.end()) {
            resampled.push_back(points.back());
        } else {
            const size_t index = static_cast<size_t>(std::distance(cumulative.begin(), upper));
            const double a = cumulative[index - 1];
            const double b = cumulative[index];
            const double t = std::abs(b - a) <= kEpsilon ? 0.0 : (target - a) / (b - a);
            resampled.push_back(lerp(points[index - 1], points[index], t));
        }
    }
    resampled.front() = points.front();
    resampled.back() = points.back();
    return resampled;
}

[[nodiscard]] std::vector<cv::Vec3d> endpointCorrectedResampledSpan(
    const std::vector<cv::Vec3d>& points,
    const cv::Vec3d& left,
    const cv::Vec3d& right,
    double segmentLength)
{
    if (points.size() < 2) {
        return {left, right};
    }

    const double totalLength = polylineLength(points);
    const cv::Vec3d leftOffset = left - points.front();
    const cv::Vec3d rightOffset = right - points.back();
    std::vector<cv::Vec3d> corrected;
    corrected.reserve(points.size());
    double cumulative = 0.0;
    corrected.push_back(left);
    for (size_t i = 1; i + 1 < points.size(); ++i) {
        cumulative += length(points[i] - points[i - 1]);
        const double t = totalLength <= kEpsilon
            ? static_cast<double>(i) / static_cast<double>(points.size() - 1)
            : std::clamp(cumulative / totalLength, 0.0, 1.0);
        corrected.push_back(points[i] + lerp(leftOffset, rightOffset, t));
    }
    corrected.push_back(right);

    std::vector<cv::Vec3d> resampled = resamplePolylineByArcLength(corrected, segmentLength);
    if (resampled.size() < 2) {
        return {left, right};
    }
    resampled.front() = left;
    resampled.back() = right;
    return resampled;
}

struct SpanRolloutResult {
    std::vector<cv::Vec3d> points;
    int rolloutSteps = 0;
    int truncatedPoints = 0;
    int selectedSign = 0;
    double closestTargetDistance = std::numeric_limits<double>::infinity();
};

struct OptimizedControlSpan {
    std::vector<std::array<double, 3>> points;
    LineReinitializationSpanReport report;
    bool failed = false;
    std::string failureReason;
    double candidateFinalCostDiff = 0.0;
    double candidateFinalRmsDiff = 0.0;
    double candidateAlignmentScoreDiff = 0.0;
    std::vector<LineDebugPolyline> continuationCandidateLines;
};

[[nodiscard]] cv::Vec3d projectedChordTangentAtStart(
    const cv::Vec3d& start,
    const cv::Vec3d& target,
    const NormalSampler& sampler)
{
    const cv::Vec3d chord = target - start;
    const NormalSample sample = sampler.sampleNormal(start);
    const cv::Vec3d normal = normalizedOrZero(sample.normal);
    if (sample.valid && length(normal) > kEpsilon) {
        const cv::Vec3d projected = normalizedOrZero(chord - normal * chord.dot(normal));
        if (length(projected) > kEpsilon) {
            return projected;
        }
        return deterministicTangentFromNormal(sample);
    }

    const cv::Vec3d fallback = normalizedOrZero(chord);
    if (length(fallback) > kEpsilon) {
        return fallback;
    }
    return deterministicTangentFromNormal(sample);
}

[[nodiscard]] cv::Vec3d projectedDirectionAtStart(
    const cv::Vec3d& start,
    const cv::Vec3d& direction,
    const NormalSampler& sampler)
{
    const NormalSample sample = sampler.sampleNormal(start);
    const cv::Vec3d normal = normalizedOrZero(sample.normal);
    if (sample.valid && length(normal) > kEpsilon) {
        cv::Vec3d projected = normalizedOrZero(direction - normal * direction.dot(normal));
        const cv::Vec3d normalizedDirection = normalizedOrZero(direction);
        if (length(projected) > kEpsilon) {
            if (length(normalizedDirection) > kEpsilon &&
                projected.dot(normalizedDirection) < 0.0) {
                projected *= -1.0;
            }
            return projected;
        }
        return deterministicTangentFromNormal(sample);
    }

    const cv::Vec3d fallback = normalizedOrZero(direction);
    if (length(fallback) > kEpsilon) {
        return fallback;
    }
    return deterministicTangentFromNormal(sample);
}

[[nodiscard]] SpanRolloutResult signedSpanRolloutCandidate(
    const cv::Vec3d& start,
    const cv::Vec3d& target,
    const cv::Vec3d& unsignedStartTangent,
    int sign,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    int rolloutSteps)
{
    LineOptimizationConfig rolloutConfig = config;
    rolloutConfig.segmentsPerSide = rolloutSteps;
    std::vector<std::array<double, 3>> grown;
    growNormalConstructedExtension(start,
                                   unsignedStartTangent * static_cast<double>(sign),
                                   sampler,
                                   rolloutConfig,
                                   grown);

    std::vector<cv::Vec3d> rollout;
    rollout.reserve(grown.size() + 1);
    rollout.push_back(start);
    for (const auto& point : grown) {
        rollout.push_back(toVec3d(point));
    }

    int bestIndex = 0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < rollout.size(); ++i) {
        const double candidateDistance = length(rollout[i] - target);
        if (candidateDistance < bestDistance) {
            bestDistance = candidateDistance;
            bestIndex = static_cast<int>(i);
        }
    }
    if (rollout.size() > 1) {
        bestIndex = std::max(1, bestIndex);
    }
    rollout.resize(static_cast<size_t>(bestIndex + 1));

    SpanRolloutResult result;
    result.rolloutSteps = rolloutSteps;
    result.truncatedPoints = static_cast<int>(rollout.size());
    result.selectedSign = sign;
    result.closestTargetDistance = bestDistance;
    result.points = endpointCorrectedResampledSpan(rollout, start, target, config.segmentLength);
    return result;
}

[[nodiscard]] SpanRolloutResult spanRolloutCandidate(
    const cv::Vec3d& start,
    const cv::Vec3d& target,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    const double budget = std::max(2.0 * length(target - start), 1000.0);
    const int steps = std::max(1, static_cast<int>(std::ceil(budget / config.segmentLength)));
    const cv::Vec3d unsignedTangent = projectedChordTangentAtStart(start, target, sampler);

    SpanRolloutResult positive = signedSpanRolloutCandidate(start,
                                                            target,
                                                            unsignedTangent,
                                                            1,
                                                            sampler,
                                                            config,
                                                            steps);
    SpanRolloutResult negative = signedSpanRolloutCandidate(start,
                                                            target,
                                                            unsignedTangent,
                                                            -1,
                                                            sampler,
                                                            config,
                                                            steps);
    if (negative.closestTargetDistance < positive.closestTargetDistance) {
        return negative;
    }
    return positive;
}

[[nodiscard]] SpanRolloutResult directedSpanRolloutCandidate(
    const cv::Vec3d& start,
    const cv::Vec3d& target,
    const cv::Vec3d& startDirection,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    const double budget = std::max(2.0 * length(target - start), 1000.0);
    const int steps = std::max(1, static_cast<int>(std::ceil(budget / config.segmentLength)));
    const cv::Vec3d tangent = projectedDirectionAtStart(start, startDirection, sampler);
    return signedSpanRolloutCandidate(start,
                                      target,
                                      tangent,
                                      1,
                                      sampler,
                                      config,
                                      steps);
}

[[nodiscard]] std::vector<std::array<double, 3>> toArrayPoints(const std::vector<cv::Vec3d>& points)
{
    std::vector<std::array<double, 3>> out;
    out.reserve(points.size());
    for (const auto& point : points) {
        out.push_back(toArray(point));
    }
    return out;
}

struct ControlPointInitialization {
    std::vector<std::array<double, 3>> points;
    std::vector<SegmentSpacingConstraint> spacingConstraints;
    std::vector<int> fixedPointIndices;
    std::vector<LineControlPoint> controlPoints;
    cv::Vec3d seedTangent{1.0, 0.0, 0.0};
};

void growNormalConstructedExtension(
    const cv::Vec3d& startPoint,
    cv::Vec3d direction,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    std::vector<std::array<double, 3>>& out)
{
    cv::Vec3d point = startPoint;
    direction = normalizedOrZero(direction);
    if (length(direction) <= kEpsilon) {
        direction = {1.0, 0.0, 0.0};
    }
    NormalSample previousSample = sampler.sampleNormal(point);
    cv::Vec3d previousNormal = normalizedOrZero(previousSample.normal);
    if (!previousSample.valid || length(previousNormal) <= kEpsilon) {
        previousNormal = {0.0, 0.0, 0.0};
    }
    out.reserve(static_cast<size_t>(config.segmentsPerSide));
    for (int i = 0; i < config.segmentsPerSide; ++i) {
        const cv::Vec3d predicted = point + direction * config.segmentLength;
        const NormalSample sample = sampler.sampleNormal(predicted);
        cv::Vec3d normal = normalizedOrZero(sample.normal);
        if (sample.valid && length(normal) > kEpsilon) {
            direction = transportedDirectionToNormalPlane(direction, previousNormal, normal);
            if (length(previousNormal) > kEpsilon && previousNormal.dot(normal) < 0.0) {
                normal *= -1.0;
            }
            previousNormal = normal;
        }
        point += direction * config.segmentLength;
        out.push_back(toArray(point));
    }
}

[[nodiscard]] cv::Vec3d interpolateInitialLinePoint(
    const std::vector<cv::Vec3d>& points,
    double linePosition)
{
    if (points.empty()) {
        return {0.0, 0.0, 0.0};
    }
    if (points.size() == 1) {
        return points.front();
    }
    linePosition = std::clamp(linePosition, 0.0, static_cast<double>(points.size() - 1));
    const int lower = static_cast<int>(std::floor(linePosition));
    const int upper = std::min<int>(lower + 1, static_cast<int>(points.size() - 1));
    const double t = linePosition - static_cast<double>(lower);
    return lerp(points[static_cast<size_t>(lower)], points[static_cast<size_t>(upper)], t);
}

[[nodiscard]] bool canInitializeFromExistingLine(
    const std::vector<LineControlPoint>& controlPoints,
    const LineOptimizationConfig& config)
{
    if (config.initialLinePoints.size() < 2 || controlPoints.size() < 2) {
        return false;
    }
    const double maxPosition = static_cast<double>(config.initialLinePoints.size() - 1);
    for (const auto& control : controlPoints) {
        if (!std::isfinite(control.linePosition) ||
            control.linePosition < 0.0 ||
            control.linePosition > maxPosition) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] cv::Vec3d warpedDeltaAtPosition(
    const std::vector<LineControlPoint>& controls,
    const std::vector<cv::Vec3d>& basePoints,
    double linePosition)
{
    if (controls.empty() || basePoints.empty()) {
        return {0.0, 0.0, 0.0};
    }
    const auto controlDelta = [&](size_t index) {
        return controls[index].volumePoint -
               interpolateInitialLinePoint(basePoints, controls[index].linePosition);
    };
    if (linePosition <= controls.front().linePosition) {
        return controlDelta(0);
    }
    for (size_t i = 0; i + 1 < controls.size(); ++i) {
        const double a = controls[i].linePosition;
        const double b = controls[i + 1].linePosition;
        if (linePosition > b) {
            continue;
        }
        const double denom = b - a;
        const double t = std::abs(denom) <= kEpsilon
            ? 1.0
            : std::clamp((linePosition - a) / denom, 0.0, 1.0);
        return lerp(controlDelta(i), controlDelta(i + 1), t);
    }
    return controlDelta(controls.size() - 1);
}

[[nodiscard]] int spanIdForSegment(
    const std::vector<LineControlPoint>& controls,
    double leftPosition,
    double rightPosition)
{
    for (size_t i = 0; i + 1 < controls.size(); ++i) {
        if (leftPosition + kEpsilon >= controls[i].linePosition &&
            rightPosition <= controls[i + 1].linePosition + kEpsilon) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

[[nodiscard]] std::optional<size_t> linePositionIndex(
    const std::vector<double>& linePositions,
    double linePosition)
{
    auto it = std::lower_bound(linePositions.begin(), linePositions.end(), linePosition);
    if (it == linePositions.end() || std::abs(*it - linePosition) > 1.0e-6) {
        return std::nullopt;
    }
    return static_cast<size_t>(std::distance(linePositions.begin(), it));
}

[[nodiscard]] bool controlMovedFromInitialLine(const LineControlPoint& control,
                                               const LineOptimizationConfig& config)
{
    const cv::Vec3d base = interpolateInitialLinePoint(config.initialLinePoints, control.linePosition);
    return length(control.volumePoint - base) > kMovedControlDistanceThreshold;
}

[[nodiscard]] bool shouldOptimizeControlSpanInitialization(
    const std::vector<LineControlPoint>& controls,
    const LineOptimizationConfig& config,
    size_t controlIndex)
{
    if (controlIndex + 1 >= controls.size()) {
        return false;
    }
    return controlMovedFromInitialLine(controls[controlIndex], config) ||
           controlMovedFromInitialLine(controls[controlIndex + 1], config);
}

[[nodiscard]] bool solveControlSpanInitializationStage(
    std::vector<std::array<double, 3>>& localPoints,
    const std::vector<std::array<double, 3>>* priorPoints,
    size_t controlIndex,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    if (localPoints.size() < 3) {
        return false;
    }

    ceres::Problem problem;
    for (auto& point : localPoints) {
        problem.AddParameterBlock(point.data(), 3);
    }
    problem.SetParameterBlockConstant(localPoints.front().data());
    problem.SetParameterBlockConstant(localPoints.back().data());

    std::vector<SegmentSpacingConstraint> noSpacing(
        localPoints.size() - 1,
        SegmentSpacingConstraint{SegmentSpacingMode::None, static_cast<int>(controlIndex)});
    addResiduals(problem,
                 localPoints,
                 noSpacing,
                 sampler,
                 config,
                 0,
                 {0.0, 0.0, 0.0},
                 false);

    if (priorPoints != nullptr && priorPoints->size() == localPoints.size()) {
        for (size_t i = 1; i + 1 < localPoints.size(); ++i) {
            auto* prior = new ceres::AutoDiffCostFunction<PositionPriorResidual, 3, 3>(
                new PositionPriorResidual(toVec3d((*priorPoints)[i]), kControlSpanInitPriorWeight));
            problem.AddResidualBlock(prior, nullptr, localPoints[i].data());
        }
    }

    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions(config, false), &problem, &summary);
    return summary.IsSolutionUsable();
}

void optimizeControlSpanInitialization(
    std::vector<std::array<double, 3>>& spanPoints,
    const std::vector<double>& linePositions,
    const std::vector<LineControlPoint>& controls,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    if (spanPoints.size() != linePositions.size()) {
        return;
    }

    for (size_t controlIndex = 0; controlIndex + 1 < controls.size(); ++controlIndex) {
        if (!shouldOptimizeControlSpanInitialization(controls, config, controlIndex)) {
            continue;
        }

        const std::optional<size_t> startIndex = linePositionIndex(linePositions, controls[controlIndex].linePosition);
        const std::optional<size_t> endIndex = linePositionIndex(linePositions, controls[controlIndex + 1].linePosition);
        if (!startIndex.has_value() || !endIndex.has_value() || *endIndex <= *startIndex) {
            continue;
        }

        std::vector<std::array<double, 3>> localPoints(
            spanPoints.begin() + static_cast<std::ptrdiff_t>(*startIndex),
            spanPoints.begin() + static_cast<std::ptrdiff_t>(*endIndex + 1));
        const std::vector<std::array<double, 3>> priorPoints = localPoints;
        if (localPoints.size() < 3) {
            continue;
        }

        LineOptimizationConfig localConfig = config;
        localConfig.maxIterations = std::min(config.maxIterations, 100);
        localConfig.printSolverProgress = false;

        if (!solveControlSpanInitializationStage(localPoints,
                                                 &priorPoints,
                                                 controlIndex,
                                                 sampler,
                                                 localConfig)) {
            continue;
        }
        (void)solveControlSpanInitializationStage(localPoints,
                                                 nullptr,
                                                 controlIndex,
                                                 sampler,
                                                 localConfig);
        std::copy(localPoints.begin(),
                  localPoints.end(),
                  spanPoints.begin() + static_cast<std::ptrdiff_t>(*startIndex));
    }
}

[[nodiscard]] std::optional<ControlPointInitialization> initializeFromExistingLine(
    std::vector<LineControlPoint> controlPoints,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    if (!canInitializeFromExistingLine(controlPoints, config)) {
        return std::nullopt;
    }

    const cv::Vec3d firstPoint = controlPoints.front().volumePoint;
    const cv::Vec3d secondPoint = controlPoints[1].volumePoint;
    const cv::Vec3d lastPoint = controlPoints.back().volumePoint;
    const cv::Vec3d beforeLastPoint = controlPoints[controlPoints.size() - 2].volumePoint;

    std::vector<std::array<double, 3>> backward;
    std::vector<std::array<double, 3>> forward;
    growNormalConstructedExtension(firstPoint, firstPoint - secondPoint, sampler, config, backward);
    growNormalConstructedExtension(lastPoint, lastPoint - beforeLastPoint, sampler, config, forward);

    const double firstLinePosition = controlPoints.front().linePosition;
    std::vector<double> linePositions;
    linePositions.reserve(config.initialLinePoints.size() + controlPoints.size());
    linePositions.push_back(firstLinePosition);
    for (size_t controlIndex = 0; controlIndex + 1 < controlPoints.size(); ++controlIndex) {
        const double a = controlPoints[controlIndex].linePosition;
        const double b = controlPoints[controlIndex + 1].linePosition;
        const int segments = std::max(1, static_cast<int>(std::llround(std::abs(b - a))));
        for (int segment = 1; segment <= segments; ++segment) {
            const double t = static_cast<double>(segment) / static_cast<double>(segments);
            linePositions.push_back(a * (1.0 - t) + b * t);
        }
    }

    ControlPointInitialization init;
    init.controlPoints = std::move(controlPoints);
    init.points.reserve(backward.size() + linePositions.size() + forward.size());
    init.spacingConstraints.reserve(init.points.capacity() > 0 ? init.points.capacity() - 1 : 0);

    for (auto it = backward.rbegin(); it != backward.rend(); ++it) {
        init.points.push_back(*it);
        init.spacingConstraints.push_back({SegmentSpacingMode::FixedStep, -1});
    }

    if (linePositions.empty()) {
        linePositions.push_back(firstLinePosition);
    }
    std::vector<std::array<double, 3>> warpedSpanPoints;
    warpedSpanPoints.reserve(linePositions.size());
    for (size_t positionIndex = 0; positionIndex < linePositions.size(); ++positionIndex) {
        const double linePosition = linePositions[positionIndex];
        if (positionIndex > 0) {
            const int spanId = spanIdForSegment(init.controlPoints,
                                                linePositions[positionIndex - 1],
                                                linePosition);
            init.spacingConstraints.push_back(spanId >= 0
                ? SegmentSpacingConstraint{SegmentSpacingMode::None, spanId}
                : SegmentSpacingConstraint{SegmentSpacingMode::FixedStep, -1});
        }
        const cv::Vec3d base = interpolateInitialLinePoint(config.initialLinePoints, linePosition);
        const cv::Vec3d delta = warpedDeltaAtPosition(init.controlPoints,
                                                      config.initialLinePoints,
                                                      linePosition);
        warpedSpanPoints.push_back(toArray(base + delta));
    }
    optimizeControlSpanInitialization(warpedSpanPoints,
                                      linePositions,
                                      init.controlPoints,
                                      sampler,
                                      config);
    init.points.insert(init.points.end(), warpedSpanPoints.begin(), warpedSpanPoints.end());

    for (const auto& point : forward) {
        init.spacingConstraints.push_back({SegmentSpacingMode::FixedStep, -1});
        init.points.push_back(point);
    }

    for (auto& control : init.controlPoints) {
        auto it = std::lower_bound(linePositions.begin(), linePositions.end(), control.linePosition);
        if (it == linePositions.end() || std::abs(*it - control.linePosition) > 1.0e-6) {
            continue;
        }
        control.optimizedIndex = static_cast<int>(backward.size() +
                                                  static_cast<size_t>(std::distance(linePositions.begin(), it)));
        init.fixedPointIndices.push_back(control.optimizedIndex);
    }

    const auto seedIt = std::find_if(init.controlPoints.begin(),
                                     init.controlPoints.end(),
                                     [](const LineControlPoint& control) {
                                         return control.isSeed && control.optimizedIndex >= 0;
                                     });
    const int tangentIndex = seedIt == init.controlPoints.end()
        ? init.fixedPointIndices.front()
        : seedIt->optimizedIndex;
    if (tangentIndex > 0 && tangentIndex + 1 < static_cast<int>(init.points.size())) {
        init.seedTangent = normalizedOrZero(toVec3d(init.points[static_cast<size_t>(tangentIndex + 1)]) -
                                            toVec3d(init.points[static_cast<size_t>(tangentIndex - 1)]));
    } else if (tangentIndex + 1 < static_cast<int>(init.points.size())) {
        init.seedTangent = normalizedOrZero(toVec3d(init.points[static_cast<size_t>(tangentIndex + 1)]) -
                                            toVec3d(init.points[static_cast<size_t>(tangentIndex)]));
    } else if (tangentIndex > 0) {
        init.seedTangent = normalizedOrZero(toVec3d(init.points[static_cast<size_t>(tangentIndex)]) -
                                            toVec3d(init.points[static_cast<size_t>(tangentIndex - 1)]));
    }
    if (length(init.seedTangent) <= kEpsilon) {
        init.seedTangent = {1.0, 0.0, 0.0};
    }
    return init;
}

[[nodiscard]] ControlPointInitialization initializeFromControlPoints(
    std::vector<LineControlPoint> controlPoints,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config)
{
    if (controlPoints.empty()) {
        throw std::invalid_argument("LineOptimizer requires at least one control point");
    }

    std::stable_sort(controlPoints.begin(), controlPoints.end(),
                     [](const LineControlPoint& a, const LineControlPoint& b) {
                         return a.linePosition < b.linePosition;
                     });

    ControlPointInitialization init;
    init.controlPoints = controlPoints;
    if (controlPoints.size() == 1) {
        const NormalSample seedNormal = sampler.sampleNormal(controlPoints.front().volumePoint);
        init.seedTangent = initialTangentFromConfig(seedNormal, config);
        init.points = directNormalConstructedPoints(controlPoints.front().volumePoint,
                                                   init.seedTangent,
                                                   sampler,
                                                   config);
        init.spacingConstraints = fixedStepConstraints(init.points.size() - 1);
        init.fixedPointIndices = {config.segmentsPerSide};
        init.controlPoints.front().optimizedIndex = config.segmentsPerSide;
        return init;
    }

    if (auto existingLineInit = initializeFromExistingLine(controlPoints, sampler, config)) {
        return *std::move(existingLineInit);
    }

    const cv::Vec3d firstPoint = controlPoints.front().volumePoint;
    const cv::Vec3d secondPoint = controlPoints[1].volumePoint;
    const cv::Vec3d lastPoint = controlPoints.back().volumePoint;
    const cv::Vec3d beforeLastPoint = controlPoints[controlPoints.size() - 2].volumePoint;
    init.seedTangent = normalizedOrZero(secondPoint - firstPoint);
    if (length(init.seedTangent) <= kEpsilon) {
        const NormalSample firstNormal = sampler.sampleNormal(firstPoint);
        init.seedTangent = initialTangentFromConfig(firstNormal, config);
    }

    std::vector<std::array<double, 3>> backward;
    std::vector<std::array<double, 3>> forward;
    growNormalConstructedExtension(firstPoint, firstPoint - secondPoint, sampler, config, backward);
    growNormalConstructedExtension(lastPoint, lastPoint - beforeLastPoint, sampler, config, forward);

    init.points.reserve(backward.size() + controlPoints.size() + forward.size() + 16);
    init.spacingConstraints.reserve(init.points.capacity() > 0 ? init.points.capacity() - 1 : 0);
    for (auto it = backward.rbegin(); it != backward.rend(); ++it) {
        init.points.push_back(*it);
        init.spacingConstraints.push_back({SegmentSpacingMode::FixedStep, -1});
    }

    init.points.push_back(toArray(firstPoint));
    init.controlPoints.front().optimizedIndex = static_cast<int>(init.points.size() - 1);
    init.fixedPointIndices.push_back(init.controlPoints.front().optimizedIndex);

    int spanId = 0;
    for (size_t controlIndex = 0; controlIndex + 1 < controlPoints.size(); ++controlIndex) {
        const cv::Vec3d a = controlPoints[controlIndex].volumePoint;
        const cv::Vec3d b = controlPoints[controlIndex + 1].volumePoint;
        const double distance = length(b - a);
        const int segments = std::max(1, static_cast<int>(std::llround(distance / config.segmentLength)));
        for (int segment = 1; segment <= segments; ++segment) {
            const double t = static_cast<double>(segment) / static_cast<double>(segments);
            init.spacingConstraints.push_back({SegmentSpacingMode::None, spanId});
            init.points.push_back(toArray(lerp(a, b, t)));
        }
        init.controlPoints[controlIndex + 1].optimizedIndex =
            static_cast<int>(init.points.size() - 1);
        init.fixedPointIndices.push_back(init.controlPoints[controlIndex + 1].optimizedIndex);
        ++spanId;
    }

    for (const auto& point : forward) {
        init.spacingConstraints.push_back({SegmentSpacingMode::FixedStep, -1});
        init.points.push_back(point);
    }
    return init;
}

struct GlobalSolveResult {
    std::string name;
    std::vector<std::array<double, 3>> points;
    double initialCost = 0.0;
    double finalCost = 0.0;
    double initialRms = 0.0;
    double finalRms = 0.0;
    int residuals = 0;
    int iterations = 0;
    bool usable = false;
    double milliseconds = 0.0;
    double ceresSolveMs = 0.0;
    PrefetchTiming prefetchTiming;
    std::string report;
};

void fillPrefetchedNormalSamples(
    const std::vector<std::array<double, 3>>& points,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    PrefetchedNormalSamples& prefetched,
    PrefetchTiming* timing = nullptr,
    int activeStart = 0,
    int activeEnd = -1)
{
    const size_t totalSampleCount = points.size() < 2
        ? 0
        : (points.size() - 1) * static_cast<size_t>(config.samplesPerSegment + 1);
    prefetched.samples.clear();
    prefetched.samples.resize(totalSampleCount);
    if (totalSampleCount == 0) {
        return;
    }
    activeStart = std::clamp(activeStart, 0, static_cast<int>(points.size()) - 1);
    activeEnd = activeEnd < 0
        ? static_cast<int>(points.size()) - 1
        : std::clamp(activeEnd, activeStart, static_cast<int>(points.size()) - 1);
    if (activeEnd <= activeStart) {
        return;
    }

    std::vector<cv::Vec3d> samplePoints;
    const size_t localSampleCount =
        static_cast<size_t>(activeEnd - activeStart) *
        static_cast<size_t>(config.samplesPerSegment + 1);
    samplePoints.reserve(localSampleCount);
    for (int segment = activeStart; segment + 1 <= activeEnd; ++segment) {
        const cv::Vec3d a{points[static_cast<size_t>(segment)][0],
                          points[static_cast<size_t>(segment)][1],
                          points[static_cast<size_t>(segment)][2]};
        const cv::Vec3d b{points[static_cast<size_t>(segment + 1)][0],
                          points[static_cast<size_t>(segment + 1)][1],
                          points[static_cast<size_t>(segment + 1)][2]};
        for (int sample = 0; sample <= config.samplesPerSegment; ++sample) {
            const double t = static_cast<double>(sample) /
                             static_cast<double>(config.samplesPerSegment);
            samplePoints.push_back(lerp(a, b, t));
        }
    }

    std::vector<NormalSampleWithDerivative> localSamples;
    const NormalBatchReport batchReport =
        sampler.sampleNormalBatch(samplePoints,
                                  config.differentiableNormalSampling,
                                  localSamples);
    if (localSamples.size() != samplePoints.size()) {
        throw std::runtime_error("Normal sampler returned the wrong number of prefetched samples");
    }
    size_t localIndex = 0;
    for (int segment = activeStart; segment + 1 <= activeEnd; ++segment) {
        for (int sample = 0; sample <= config.samplesPerSegment; ++sample) {
            const size_t sampleIndex = static_cast<size_t>(segment) *
                                           static_cast<size_t>(config.samplesPerSegment + 1) +
                                       static_cast<size_t>(sample);
            prefetched.samples[sampleIndex] = localSamples[localIndex++];
        }
    }
    if (timing) {
        ++timing->calls;
        timing->chunkPrefetchMs += batchReport.prefetchMs;
        timing->materializeMs += batchReport.materializeMs;
        timing->requestedChunks += batchReport.prefetch.requestedChunks;
        timing->chunksRead += batchReport.prefetch.chunksRead;
    }
}

class LineNormalIterationPrefetchCallback final : public ceres::IterationCallback {
public:
    LineNormalIterationPrefetchCallback(
        const std::vector<std::array<double, 3>>& points,
        const NormalSampler& sampler,
        const LineOptimizationConfig& config,
        PrefetchedNormalSamples& prefetched,
        PrefetchTiming& timing,
        int activeStart,
        int activeEnd)
        : points_(points)
        , sampler_(sampler)
        , config_(config)
        , prefetched_(prefetched)
        , timing_(timing)
        , activeStart_(activeStart)
        , activeEnd_(activeEnd)
    {
    }

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& /*summary*/) override
    {
        fillPrefetchedNormalSamples(points_,
                                    sampler_,
                                    config_,
                                    prefetched_,
                                    &timing_,
                                    activeStart_,
                                    activeEnd_);
        return ceres::SOLVER_CONTINUE;
    }

private:
    const std::vector<std::array<double, 3>>& points_;
    const NormalSampler& sampler_;
    const LineOptimizationConfig& config_;
    PrefetchedNormalSamples& prefetched_;
    PrefetchTiming& timing_;
    int activeStart_ = 0;
    int activeEnd_ = -1;
};

[[nodiscard]] GlobalSolveResult solveGlobalCandidate(
    std::string name,
    std::vector<std::array<double, 3>> initialPoints,
    std::vector<SegmentSpacingConstraint> spacingConstraints,
    std::vector<int> fixedPointIndices,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    const cv::Vec3d& seedTangent,
    bool useSeedGuides,
    Clock::time_point chainStart,
    int activeStart = 0,
    int activeEnd = -1)
{
    if (!initialPoints.empty()) {
        activeStart = std::clamp(activeStart, 0, static_cast<int>(initialPoints.size()) - 1);
        activeEnd = activeEnd < 0
            ? static_cast<int>(initialPoints.size()) - 1
            : std::clamp(activeEnd, activeStart, static_cast<int>(initialPoints.size()) - 1);
    }
    int seedIndex = fixedPointIndices.empty() ? config.segmentsPerSide : fixedPointIndices.front();

    PrefetchedNormalSamples prefetchedSamples;
    PrefetchTiming prefetchTiming;
    ceres::Problem problem;
    for (int index = activeStart; index <= activeEnd; ++index) {
        problem.AddParameterBlock(initialPoints[static_cast<size_t>(index)].data(), 3);
    }
    std::sort(fixedPointIndices.begin(), fixedPointIndices.end());
    fixedPointIndices.erase(std::unique(fixedPointIndices.begin(), fixedPointIndices.end()),
                            fixedPointIndices.end());
    for (const int index : fixedPointIndices) {
        if (index >= activeStart && index <= activeEnd) {
            problem.SetParameterBlockConstant(initialPoints[static_cast<size_t>(index)].data());
        }
    }
    addResiduals(problem,
                 initialPoints,
                 spacingConstraints,
                 sampler,
                 config,
                 seedIndex,
                 seedTangent,
                 useSeedGuides,
                 &prefetchedSamples,
                 activeStart,
                 activeEnd);
    const int residuals = problem.NumResiduals();

    fillPrefetchedNormalSamples(initialPoints,
                                sampler,
                                config,
                                prefetchedSamples,
                                &prefetchTiming,
                                activeStart,
                                activeEnd);
    double initialCost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions{}, &initialCost, nullptr, nullptr, nullptr);

    ceres::Solver::Summary summary;
    LineNormalIterationPrefetchCallback prefetchCallback(initialPoints,
                                                         sampler,
                                                         config,
                                                         prefetchedSamples,
                                                         prefetchTiming,
                                                         activeStart,
                                                         activeEnd);
    ceres::Solver::Options options = solverOptions(config, config.printSolverProgress);
    options.update_state_every_iteration = true;
    options.callbacks.push_back(&prefetchCallback);
    const double normalMsBeforeSolve = prefetchTiming.chunkPrefetchMs + prefetchTiming.materializeMs;
    const auto solveStart = Clock::now();
    ceres::Solve(options, &problem, &summary);
    const auto solveEnd = Clock::now();
    const double normalMsAfterSolve = prefetchTiming.chunkPrefetchMs + prefetchTiming.materializeMs;
    const double normalCallbackMs = std::max(0.0, normalMsAfterSolve - normalMsBeforeSolve);
    const double ceresWallMs = elapsedMs(solveStart, solveEnd);

    GlobalSolveResult result;
    result.name = std::move(name);
    result.points = std::move(initialPoints);
    result.initialCost = initialCost;
    result.finalCost = summary.final_cost;
    result.initialRms = ceresResidualRms(initialCost, residuals);
    result.finalRms = ceresResidualRms(summary.final_cost, residuals);
    result.residuals = residuals;
    result.iterations = static_cast<int>(summary.iterations.size());
    result.usable = summary.IsSolutionUsable();
    result.milliseconds = elapsedMs(chainStart, Clock::now());
    result.ceresSolveMs = std::max(0.0, ceresWallMs - normalCallbackMs);
    result.prefetchTiming = prefetchTiming;
    result.report = summary.FullReport();
    return result;
}

struct LineDifference {
    double rms = 0.0;
    double max = 0.0;
};

[[nodiscard]] LineDifference compareLines(
    const std::vector<std::array<double, 3>>& a,
    const std::vector<std::array<double, 3>>& b)
{
    LineDifference diff;
    if (a.size() != b.size() || a.empty()) {
        diff.rms = std::numeric_limits<double>::quiet_NaN();
        diff.max = std::numeric_limits<double>::quiet_NaN();
        return diff;
    }
    double sumSq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const cv::Vec3d delta = toVec3d(a[i]) - toVec3d(b[i]);
        const double d = length(delta);
        sumSq += d * d;
        diff.max = std::max(diff.max, d);
    }
    diff.rms = std::sqrt(sumSq / static_cast<double>(a.size()));
    return diff;
}

[[nodiscard]] std::string comparisonReport(const std::vector<GlobalSolveResult>& results)
{
    if (results.empty()) {
        return {};
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::scientific << std::setprecision(3);
    out << "Line annotation Lasagna candidate comparison:\n"
        << "candidate                   ms  iters     init_rms    final_rms   rms_vs_inc   max_vs_inc\n";
    const auto& reference = results.front();
    for (const auto& result : results) {
        const LineDifference diff = compareLines(reference.points, result.points);
        out << std::left << std::setw(24) << result.name
            << std::right << std::setw(10) << result.milliseconds
            << std::setw(7) << result.iterations
            << std::setw(13) << result.initialRms
            << std::setw(13) << result.finalRms
            << std::setw(13) << diff.rms
            << std::setw(13) << diff.max
            << '\n';
    }
    return out.str();
}

[[nodiscard]] const LineReinitializationCandidateReport* spanCandidateReport(
    const LineReinitializationSpanReport& report,
    const char* name)
{
    for (const auto& candidate : report.candidates) {
        if (candidate.name == name) {
            return &candidate;
        }
    }
    return nullptr;
}

[[nodiscard]] double initialRmsForCandidate(const LineReinitializationSpanReport& report,
                                            const char* name)
{
    const auto* candidate = spanCandidateReport(report, name);
    return candidate ? candidate->initialRms : 0.0;
}

[[nodiscard]] double finalRmsForCandidate(const LineReinitializationSpanReport& report,
                                          const char* name)
{
    const auto* candidate = spanCandidateReport(report, name);
    return candidate ? candidate->finalRms : 0.0;
}

[[nodiscard]] OptimizedControlSpan optimizedControlSpan(
    const cv::Vec3d& leftPoint,
    const cv::Vec3d& rightPoint,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    int segmentIndex,
    int leftControlIndex,
    int rightControlIndex,
    std::string candidatePrefix,
    std::optional<cv::Vec3d> leftContinuationDirection = std::nullopt,
    std::optional<cv::Vec3d> rightContinuationDirection = std::nullopt)
{
    if (length(rightPoint - leftPoint) <= kEpsilon) {
        throw std::invalid_argument("Adjacent reinitialization control points are coincident");
    }

    const SpanRolloutResult leftRollout =
        spanRolloutCandidate(leftPoint, rightPoint, sampler, config);

    SpanRolloutResult rightRollout =
        spanRolloutCandidate(rightPoint, leftPoint, sampler, config);
    std::reverse(rightRollout.points.begin(), rightRollout.points.end());
    if (!rightRollout.points.empty()) {
        rightRollout.points.front() = leftPoint;
        rightRollout.points.back() = rightPoint;
    }

    LineOptimizationConfig spanConfig = config;
    spanConfig.printSolverProgress = false;
    const cv::Vec3d seedTangent = projectedChordTangentAtStart(leftPoint, rightPoint, sampler);
    const auto spanStart = Clock::now();
    const auto solveRollout = [&](std::string name, const SpanRolloutResult& rollout) {
        return solveGlobalCandidate(std::move(name),
                                    toArrayPoints(rollout.points),
                                    evenStepSpanConstraints(rollout.points.size() - 1,
                                                            segmentIndex),
                                    {0, static_cast<int>(rollout.points.size()) - 1},
                                    sampler,
                                    spanConfig,
                                    seedTangent,
                                    false,
                                    spanStart);
    };

    GlobalSolveResult leftSolved =
        solveRollout(candidatePrefix + "-left-" + std::to_string(segmentIndex), leftRollout);
    GlobalSolveResult rightSolved =
        solveRollout(candidatePrefix + "-right-" + std::to_string(segmentIndex), rightRollout);

    std::optional<SpanRolloutResult> continueLeftRollout;
    std::optional<GlobalSolveResult> continueLeftSolved;
    if (leftContinuationDirection.has_value() &&
        length(normalizedOrZero(*leftContinuationDirection)) > kEpsilon) {
        continueLeftRollout = directedSpanRolloutCandidate(leftPoint,
                                                           rightPoint,
                                                           *leftContinuationDirection,
                                                           sampler,
                                                           config);
        continueLeftSolved =
            solveRollout(candidatePrefix + "-continue-left-" + std::to_string(segmentIndex),
                         *continueLeftRollout);
    }

    std::optional<SpanRolloutResult> continueRightRollout;
    std::optional<GlobalSolveResult> continueRightSolved;
    if (rightContinuationDirection.has_value() &&
        length(normalizedOrZero(*rightContinuationDirection)) > kEpsilon) {
        continueRightRollout = directedSpanRolloutCandidate(rightPoint,
                                                            leftPoint,
                                                            *rightContinuationDirection,
                                                            sampler,
                                                            config);
        std::reverse(continueRightRollout->points.begin(), continueRightRollout->points.end());
        if (!continueRightRollout->points.empty()) {
            continueRightRollout->points.front() = leftPoint;
            continueRightRollout->points.back() = rightPoint;
        }
        continueRightSolved =
            solveRollout(candidatePrefix + "-continue-right-" + std::to_string(segmentIndex),
                         *continueRightRollout);
    }

    const SpanDeviationMetrics leftMetrics =
        spanDeviationMetrics(leftSolved.points, sampler, config);
    const SpanDeviationMetrics rightMetrics =
        spanDeviationMetrics(rightSolved.points, sampler, config);
    std::optional<SpanDeviationMetrics> continueLeftMetrics;
    if (continueLeftSolved.has_value()) {
        continueLeftMetrics = spanDeviationMetrics(continueLeftSolved->points, sampler, config);
    }
    std::optional<SpanDeviationMetrics> continueRightMetrics;
    if (continueRightSolved.has_value()) {
        continueRightMetrics = spanDeviationMetrics(continueRightSolved->points, sampler, config);
    }

    const auto normalizedReference =
        [](const std::optional<cv::Vec3d>& direction) -> std::optional<cv::Vec3d> {
        if (!direction.has_value()) {
            return std::nullopt;
        }
        const cv::Vec3d normalized = normalizedOrZero(*direction);
        if (length(normalized) <= kEpsilon) {
            return std::nullopt;
        }
        return normalized;
    };
    const std::optional<cv::Vec3d> leftReferenceDirection =
        normalizedReference(leftContinuationDirection);
    const std::optional<cv::Vec3d> rightReferenceDirection =
        normalizedReference(rightContinuationDirection);
    const auto leftReferenceDot = [&](const GlobalSolveResult& candidate) {
        return leftReferenceDirection.has_value()
            ? endpointDirectionDot(candidate.points,
                                   SpanCandidateAnchorSide::Left,
                                   *leftReferenceDirection)
            : std::numeric_limits<double>::quiet_NaN();
    };
    const auto rightReferenceDot = [&](const GlobalSolveResult& candidate) {
        return rightReferenceDirection.has_value()
            ? endpointDirectionDot(candidate.points,
                                   SpanCandidateAnchorSide::Right,
                                   *rightReferenceDirection)
            : std::numeric_limits<double>::quiet_NaN();
    };
    const auto matchesReferenceDirections =
        [&](const GlobalSolveResult& candidate, double leftDot, double rightDot) {
        if (!candidate.usable) {
            return false;
        }
        if (leftReferenceDirection.has_value() && !(leftDot > 0.0)) {
            return false;
        }
        if (rightReferenceDirection.has_value() && !(rightDot > 0.0)) {
            return false;
        }
        return true;
    };

    const double leftCandidateLeftDot = leftReferenceDot(leftSolved);
    const double leftCandidateRightDot = rightReferenceDot(leftSolved);
    const double rightCandidateLeftDot = leftReferenceDot(rightSolved);
    const double rightCandidateRightDot = rightReferenceDot(rightSolved);
    const double continueLeftCandidateLeftDot =
        continueLeftSolved.has_value()
            ? leftReferenceDot(*continueLeftSolved)
            : std::numeric_limits<double>::quiet_NaN();
    const double continueLeftCandidateRightDot =
        continueLeftSolved.has_value()
            ? rightReferenceDot(*continueLeftSolved)
            : std::numeric_limits<double>::quiet_NaN();
    const double continueRightCandidateLeftDot =
        continueRightSolved.has_value()
            ? leftReferenceDot(*continueRightSolved)
            : std::numeric_limits<double>::quiet_NaN();
    const double continueRightCandidateRightDot =
        continueRightSolved.has_value()
            ? rightReferenceDot(*continueRightSolved)
            : std::numeric_limits<double>::quiet_NaN();

    const bool leftSelectable =
        matchesReferenceDirections(leftSolved, leftCandidateLeftDot, leftCandidateRightDot);
    const bool rightSelectable =
        matchesReferenceDirections(rightSolved, rightCandidateLeftDot, rightCandidateRightDot);
    const bool continueLeftSelectable =
        continueLeftSolved.has_value() &&
        matchesReferenceDirections(*continueLeftSolved,
                                   continueLeftCandidateLeftDot,
                                   continueLeftCandidateRightDot);
    const bool continueRightSelectable =
        continueRightSolved.has_value() &&
        matchesReferenceDirections(*continueRightSolved,
                                   continueRightCandidateLeftDot,
                                   continueRightCandidateRightDot);

    const GlobalSolveResult* chosen = nullptr;
    const char* chosenName = "";
    double minFinalCost = std::numeric_limits<double>::infinity();
    double maxFinalCost = -std::numeric_limits<double>::infinity();
    double minFinalRms = std::numeric_limits<double>::infinity();
    double maxFinalRms = -std::numeric_limits<double>::infinity();
    double minAlignmentScore = std::numeric_limits<double>::infinity();
    double maxAlignmentScore = -std::numeric_limits<double>::infinity();
    double chosenAlignmentScore = std::numeric_limits<double>::infinity();
    const auto includeCandidate = [&](const GlobalSolveResult& candidate,
                                      const char* name,
                                      const SpanDeviationMetrics& metrics,
                                      bool selectable) {
        if (!selectable) {
            return;
        }
        const double score = alignmentChoiceScore(metrics);
        minFinalCost = std::min(minFinalCost, candidate.finalCost);
        maxFinalCost = std::max(maxFinalCost, candidate.finalCost);
        minFinalRms = std::min(minFinalRms, candidate.finalRms);
        maxFinalRms = std::max(maxFinalRms, candidate.finalRms);
        minAlignmentScore = std::min(minAlignmentScore, score);
        maxAlignmentScore = std::max(maxAlignmentScore, score);
        if (chosen == nullptr || score < chosenAlignmentScore) {
            chosen = &candidate;
            chosenName = name;
            chosenAlignmentScore = score;
        }
    };
    includeCandidate(leftSolved, "left", leftMetrics, leftSelectable);
    includeCandidate(rightSolved, "right", rightMetrics, rightSelectable);
    if (continueLeftSolved.has_value() && continueLeftMetrics.has_value()) {
        includeCandidate(*continueLeftSolved,
                         "continue-left",
                         *continueLeftMetrics,
                         continueLeftSelectable);
    }
    if (continueRightSolved.has_value() && continueRightMetrics.has_value()) {
        includeCandidate(*continueRightSolved,
                         "continue-right",
                         *continueRightMetrics,
                         continueRightSelectable);
    }

    const bool spanFailed = chosen == nullptr;
    std::string failureReason;
    if (spanFailed) {
        const auto appendCandidateDots =
            [&](std::ostringstream& error,
                const char* name,
                double leftDot,
                double rightDot) {
            error << ' ' << name << '=';
            if (leftReferenceDirection.has_value() && rightReferenceDirection.has_value()) {
                error << "(left_ref=" << leftDot << ",right_ref=" << rightDot << ')';
            } else if (leftReferenceDirection.has_value()) {
                error << leftDot;
            } else if (rightReferenceDirection.has_value()) {
                error << rightDot;
            } else {
                error << "no_reference";
            }
        };
        std::ostringstream error;
        error.imbue(std::locale::classic());
        error << std::scientific << std::setprecision(3)
              << "No reinitialization rollout candidate for span " << segmentIndex
              << " matches the adjacent span direction within 90 degrees"
              << " dots:";
        appendCandidateDots(error, "left", leftCandidateLeftDot, leftCandidateRightDot);
        appendCandidateDots(error, "right", rightCandidateLeftDot, rightCandidateRightDot);
        if (continueLeftSolved.has_value()) {
            appendCandidateDots(error,
                                "continue-left",
                                continueLeftCandidateLeftDot,
                                continueLeftCandidateRightDot);
        }
        if (continueRightSolved.has_value()) {
            appendCandidateDots(error,
                                "continue-right",
                                continueRightCandidateLeftDot,
                                continueRightCandidateRightDot);
        }
        failureReason = error.str();
    }

    OptimizedControlSpan result;
    result.failed = spanFailed;
    result.failureReason = failureReason;
    if (!spanFailed) {
        result.points = chosen->points;
    }
    result.candidateFinalCostDiff =
        (std::isfinite(maxFinalCost) && std::isfinite(minFinalCost))
            ? maxFinalCost - minFinalCost
            : 0.0;
    result.candidateFinalRmsDiff =
        (std::isfinite(maxFinalRms) && std::isfinite(minFinalRms))
            ? maxFinalRms - minFinalRms
            : 0.0;
    result.candidateAlignmentScoreDiff =
        (std::isfinite(maxAlignmentScore) && std::isfinite(minAlignmentScore))
            ? maxAlignmentScore - minAlignmentScore
            : 0.0;
    result.report.segmentIndex = segmentIndex;
    result.report.leftControlIndex = leftControlIndex;
    result.report.rightControlIndex = rightControlIndex;
    result.report.points = spanFailed ? 0 : static_cast<int>(chosen->points.size());
    result.report.candLeftRolloutSteps = leftRollout.rolloutSteps;
    result.report.candLeftTruncatedPoints = leftRollout.truncatedPoints;
    result.report.candRightRolloutSteps = rightRollout.rolloutSteps;
    result.report.candRightTruncatedPoints = rightRollout.truncatedPoints;
    if (continueLeftRollout.has_value()) {
        result.continuationCandidateLines.push_back({
            candidatePrefix + "-continue-left-" + std::to_string(segmentIndex),
            continueLeftRollout->points,
        });
        result.report.candContinueLeftRolloutSteps = continueLeftRollout->rolloutSteps;
        result.report.candContinueLeftTruncatedPoints = continueLeftRollout->truncatedPoints;
        result.report.candContinueLeftClosestTargetDistance =
            continueLeftRollout->closestTargetDistance;
    }
    if (continueRightRollout.has_value()) {
        result.continuationCandidateLines.push_back({
            candidatePrefix + "-continue-right-" + std::to_string(segmentIndex),
            continueRightRollout->points,
        });
        result.report.candContinueRightRolloutSteps = continueRightRollout->rolloutSteps;
        result.report.candContinueRightTruncatedPoints = continueRightRollout->truncatedPoints;
        result.report.candContinueRightClosestTargetDistance =
            continueRightRollout->closestTargetDistance;
    }
    if (spanFailed) {
        const auto solvedPolyline = [](const GlobalSolveResult& solved) {
            std::vector<cv::Vec3d> line;
            line.reserve(solved.points.size());
            for (const auto& point : solved.points) {
                line.push_back(toVec3d(point));
            }
            return line;
        };
        result.continuationCandidateLines.push_back({
            candidatePrefix + "-failed-left-" + std::to_string(segmentIndex),
            solvedPolyline(leftSolved),
        });
        result.continuationCandidateLines.push_back({
            candidatePrefix + "-failed-right-" + std::to_string(segmentIndex),
            solvedPolyline(rightSolved),
        });
        if (continueLeftSolved.has_value()) {
            result.continuationCandidateLines.push_back({
                candidatePrefix + "-failed-continue-left-" + std::to_string(segmentIndex),
                solvedPolyline(*continueLeftSolved),
            });
        }
        if (continueRightSolved.has_value()) {
            result.continuationCandidateLines.push_back({
                candidatePrefix + "-failed-continue-right-" + std::to_string(segmentIndex),
                solvedPolyline(*continueRightSolved),
            });
        }
    }
    result.report.candLeftSelectedSign = leftRollout.selectedSign;
    result.report.candRightSelectedSign = rightRollout.selectedSign;
    result.report.candLeftClosestTargetDistance = leftRollout.closestTargetDistance;
    result.report.candRightClosestTargetDistance = rightRollout.closestTargetDistance;
    result.report.candLeftInitialCost = leftSolved.initialCost;
    result.report.candLeftFinalCost = leftSolved.finalCost;
    result.report.candRightInitialCost = rightSolved.initialCost;
    result.report.candRightFinalCost = rightSolved.finalCost;
    if (continueLeftSolved.has_value()) {
        result.report.candContinueLeftInitialCost = continueLeftSolved->initialCost;
        result.report.candContinueLeftFinalCost = continueLeftSolved->finalCost;
    }
    if (continueRightSolved.has_value()) {
        result.report.candContinueRightInitialCost = continueRightSolved->initialCost;
        result.report.candContinueRightFinalCost = continueRightSolved->finalCost;
    }
    result.report.candidates.reserve(2 +
                                     (continueLeftSolved.has_value() ? 1 : 0) +
                                     (continueRightSolved.has_value() ? 1 : 0));
    const auto addCandidateReport = [&](const char* name,
                                        const SpanRolloutResult& rollout,
                                        const GlobalSolveResult& solved,
                                        const SpanDeviationMetrics& deviations,
                                        bool selectable) {
        LineReinitializationCandidateReport candidate;
        const double alignmentScore = alignmentChoiceScore(deviations);
        candidate.name = name;
        candidate.rolloutSteps = rollout.rolloutSteps;
        candidate.truncatedPoints = rollout.truncatedPoints;
        candidate.selectedSign = rollout.selectedSign;
        candidate.points = static_cast<int>(solved.points.size());
        candidate.residuals = solved.residuals;
        candidate.iterations = solved.iterations;
        candidate.usable = selectable;
        candidate.chosen = candidate.name == chosenName;
        candidate.closestTargetDistance = rollout.closestTargetDistance;
        candidate.initialCost = solved.initialCost;
        candidate.finalCost = solved.finalCost;
        candidate.initialRms = solved.initialRms;
        candidate.finalRms = solved.finalRms;
        candidate.finalRmsDelta = std::max(0.0, solved.finalRms - minFinalRms);
        candidate.finalCostDelta = std::max(0.0, solved.finalCost - minFinalCost);
        candidate.avgNormalAlignmentAbs = deviations.avgNormalAlignmentAbs;
        candidate.p95NormalAlignmentAbs = deviations.p95NormalAlignmentAbs;
        candidate.maxNormalAlignmentAbs = deviations.maxNormalAlignmentAbs;
        candidate.alignmentChoiceScore = alignmentScore;
        candidate.alignmentChoiceScoreDelta =
            (std::isfinite(alignmentScore) && std::isfinite(minAlignmentScore))
                ? std::max(0.0, alignmentScore - minAlignmentScore)
                : 0.0;
        result.report.candidates.push_back(std::move(candidate));
    };
    addCandidateReport("left", leftRollout, leftSolved, leftMetrics, leftSelectable);
    addCandidateReport("right", rightRollout, rightSolved, rightMetrics, rightSelectable);
    if (continueLeftRollout.has_value() && continueLeftSolved.has_value() &&
        continueLeftMetrics.has_value()) {
        addCandidateReport("continue-left",
                           *continueLeftRollout,
                           *continueLeftSolved,
                           *continueLeftMetrics,
                           continueLeftSelectable);
    }
    if (continueRightRollout.has_value() && continueRightSolved.has_value() &&
        continueRightMetrics.has_value()) {
        addCandidateReport("continue-right",
                           *continueRightRollout,
                           *continueRightSolved,
                           *continueRightMetrics,
                           continueRightSelectable);
    }
    if (!spanFailed) {
        const SpanDeviationMetrics deviations = spanDeviationMetrics(chosen->points, sampler, config);
        result.report.chosenMaxEvenStepDeviation = deviations.maxEvenStepDeviation;
        result.report.chosenMaxTangentSmoothDeviation = deviations.maxTangentSmoothDeviation;
        result.report.chosenMaxNormalSmoothDeviation = deviations.maxNormalSmoothDeviation;
        result.report.chosenMaxNormalAlignmentAbs = deviations.maxNormalAlignmentAbs;
        result.report.chosen = chosenName;
    } else {
        result.report.chosen = "failed";
    }
    return result;
}

[[nodiscard]] LineModel buildLineModel(
    const std::vector<std::array<double, 3>>& points,
    const NormalSampler& sampler,
    std::vector<LineSegmentSamples> segmentSamples,
    int displayFrameAnchorIndex)
{
    LineModel model;
    model.displayFrameAnchorIndex = displayFrameAnchorIndex;
    model.points.reserve(points.size());
    for (const auto& point : points) {
        LinePoint linePoint;
        linePoint.position = {point[0], point[1], point[2]};
        linePoint.sampledNormal = sampler.sampleNormal(linePoint.position);
        linePoint.sampledNormal.normal = normalizedOrZero(linePoint.sampledNormal.normal);
        linePoint.sampledNormal.valid =
            linePoint.sampledNormal.valid && length(linePoint.sampledNormal.normal) > kEpsilon;
        linePoint.valid = linePoint.sampledNormal.valid;
        model.points.push_back(std::move(linePoint));
    }
    model.segmentSamples = std::move(segmentSamples);
    return model;
}

[[nodiscard]] int nearestPointIndex(
    const std::vector<std::array<double, 3>>& points,
    const cv::Vec3d& target)
{
    int bestIndex = -1;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < points.size(); ++i) {
        const cv::Vec3d delta = toVec3d(points[i]) - target;
        const double distance = length(delta);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

[[nodiscard]] std::vector<double> uniqueSortedPositions(std::vector<double> positions)
{
    std::sort(positions.begin(), positions.end());
    positions.erase(std::remove_if(positions.begin(),
                                   positions.end(),
                                   [](double position) {
                                       return !std::isfinite(position);
                                   }),
                    positions.end());
    positions.erase(std::unique(positions.begin(),
                                positions.end(),
                                [](double a, double b) {
                                    return std::abs(a - b) <= 1.0e-6;
                                }),
                    positions.end());
    return positions;
}

[[nodiscard]] std::pair<int, int> activeRangeAroundControlSpans(
    const std::vector<LineControlPoint>& controlPoints,
    int changedControlIndex,
    int pointCount,
    int spanRadius)
{
    if (pointCount <= 0) {
        return {-1, -1};
    }
    const int maxIndex = pointCount - 1;
    if (controlPoints.empty() || changedControlIndex < 0) {
        return {0, maxIndex};
    }

    changedControlIndex = std::clamp(changedControlIndex,
                                     0,
                                     static_cast<int>(controlPoints.size()) - 1);
    const int lastControlIndex = static_cast<int>(controlPoints.size()) - 1;
    int leftControl = changedControlIndex;
    bool includeLeftOpenEnd = false;
    for (int span = 0; span < spanRadius; ++span) {
        if (leftControl == 0) {
            includeLeftOpenEnd = true;
            break;
        }
        --leftControl;
    }
    int rightControl = changedControlIndex;
    bool includeRightOpenEnd = false;
    for (int span = 0; span < spanRadius; ++span) {
        if (rightControl == lastControlIndex) {
            includeRightOpenEnd = true;
            break;
        }
        ++rightControl;
    }

    int activeStart = includeLeftOpenEnd
        ? 0
        : std::clamp(controlPoints[static_cast<size_t>(leftControl)].optimizedIndex,
                     0,
                     maxIndex);
    int activeEnd = includeRightOpenEnd
        ? maxIndex
        : std::clamp(controlPoints[static_cast<size_t>(rightControl)].optimizedIndex,
                     0,
                     maxIndex);
    if (activeEnd < activeStart) {
        std::swap(activeStart, activeEnd);
    }
    return {activeStart, activeEnd};
}

} // namespace

LineControlPointUpdateResult updateExistingLineControlPoint(
    std::vector<cv::Vec3d> linePoints,
    std::vector<LineControlPoint> controlPoints,
    size_t changedControlIndex,
    double segmentLength)
{
    if (linePoints.size() < 2) {
        throw std::invalid_argument("Existing line update requires at least two samples");
    }
    if (changedControlIndex >= controlPoints.size()) {
        throw std::invalid_argument("Changed control index is out of range");
    }

    segmentLength = std::max(kEpsilon, segmentLength);
    const LineControlPoint changedBeforeSort = controlPoints[changedControlIndex];
    std::vector<std::pair<LineControlPoint, bool>> taggedControls;
    taggedControls.reserve(controlPoints.size());
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        taggedControls.push_back({controlPoints[i], i == changedControlIndex});
    }
    std::stable_sort(taggedControls.begin(),
                     taggedControls.end(),
                     [](const auto& a, const auto& b) {
                         return a.first.linePosition < b.first.linePosition;
                     });

    int changedSortedIndex = -1;
    controlPoints.clear();
    controlPoints.reserve(taggedControls.size());
    for (size_t i = 0; i < taggedControls.size(); ++i) {
        taggedControls[i].first.linePosition = std::clamp(
            taggedControls[i].first.linePosition,
            0.0,
            static_cast<double>(linePoints.size() - 1));
        if (taggedControls[i].second) {
            changedSortedIndex = static_cast<int>(i);
        }
        controlPoints.push_back(taggedControls[i].first);
    }
    if (changedSortedIndex < 0) {
        throw std::invalid_argument("Changed control point was not found after sorting");
    }
    (void)changedBeforeSort;

    const auto controlDelta = [&](const LineControlPoint& control) {
        return control.volumePoint - interpolateInitialLinePoint(linePoints, control.linePosition);
    };
    const auto resampleSpan = [&](const LineControlPoint& left,
                                  const LineControlPoint& right) {
        const double leftPosition = left.linePosition;
        const double rightPosition = right.linePosition;
        std::vector<double> positions;
        positions.push_back(leftPosition);
        const int firstInteger = static_cast<int>(std::floor(leftPosition)) + 1;
        const int lastInteger = static_cast<int>(std::ceil(rightPosition)) - 1;
        for (int position = firstInteger; position <= lastInteger; ++position) {
            if (position > leftPosition + kEpsilon &&
                position < rightPosition - kEpsilon) {
                positions.push_back(static_cast<double>(position));
            }
        }
        positions.push_back(rightPosition);
        positions = uniqueSortedPositions(std::move(positions));

        const cv::Vec3d leftDelta = controlDelta(left);
        const cv::Vec3d rightDelta = controlDelta(right);
        std::vector<cv::Vec3d> provisional;
        provisional.reserve(positions.size());
        for (const double position : positions) {
            const double denom = rightPosition - leftPosition;
            const double t = std::abs(denom) <= kEpsilon
                ? 0.0
                : std::clamp((position - leftPosition) / denom, 0.0, 1.0);
            provisional.push_back(interpolateInitialLinePoint(linePoints, position) +
                                  lerp(leftDelta, rightDelta, t));
        }

        double spanLength = 0.0;
        for (size_t i = 0; i + 1 < provisional.size(); ++i) {
            spanLength += length(provisional[i + 1] - provisional[i]);
        }
        const int segments = std::max(1, static_cast<int>(std::llround(spanLength / segmentLength)));
        std::vector<cv::Vec3d> resampled;
        resampled.reserve(static_cast<size_t>(segments + 1));
        if (spanLength <= kEpsilon || provisional.size() < 2) {
            for (int segment = 0; segment <= segments; ++segment) {
                const double t = static_cast<double>(segment) / static_cast<double>(segments);
                resampled.push_back(lerp(left.volumePoint, right.volumePoint, t));
            }
            return resampled;
        }

        std::vector<double> cumulative(provisional.size(), 0.0);
        for (size_t i = 1; i < provisional.size(); ++i) {
            cumulative[i] = cumulative[i - 1] + length(provisional[i] - provisional[i - 1]);
        }
        for (int segment = 0; segment <= segments; ++segment) {
            const double target = spanLength * static_cast<double>(segment) /
                                  static_cast<double>(segments);
            auto upper = std::lower_bound(cumulative.begin(), cumulative.end(), target);
            if (upper == cumulative.begin()) {
                resampled.push_back(provisional.front());
            } else if (upper == cumulative.end()) {
                resampled.push_back(provisional.back());
            } else {
                const size_t index = static_cast<size_t>(std::distance(cumulative.begin(), upper));
                const double a = cumulative[index - 1];
                const double b = cumulative[index];
                const double t = std::abs(b - a) <= kEpsilon ? 0.0 : (target - a) / (b - a);
                resampled.push_back(lerp(provisional[index - 1], provisional[index], t));
            }
        }
        resampled.front() = left.volumePoint;
        resampled.back() = right.volumePoint;
        return resampled;
    };

    const bool hasLeft = changedSortedIndex > 0;
    const bool hasRight = changedSortedIndex + 1 < static_cast<int>(controlPoints.size());
    if (!hasLeft && !hasRight) {
        LineControlPointUpdateResult unchanged;
        unchanged.linePoints = std::move(linePoints);
        unchanged.controlPoints = std::move(controlPoints);
        unchanged.changedControlIndex = changedSortedIndex;
        unchanged.activeStart = 0;
        unchanged.activeEnd = static_cast<int>(unchanged.linePoints.size()) - 1;
        return unchanged;
    }

    const double replaceStartPosition = hasLeft
        ? controlPoints[static_cast<size_t>(changedSortedIndex - 1)].linePosition
        : controlPoints[static_cast<size_t>(changedSortedIndex)].linePosition;
    const double replaceEndPosition = hasRight
        ? controlPoints[static_cast<size_t>(changedSortedIndex + 1)].linePosition
        : controlPoints[static_cast<size_t>(changedSortedIndex)].linePosition;

    std::vector<cv::Vec3d> replacement;
    if (hasLeft) {
        replacement = resampleSpan(controlPoints[static_cast<size_t>(changedSortedIndex - 1)],
                                   controlPoints[static_cast<size_t>(changedSortedIndex)]);
    }
    if (hasRight) {
        std::vector<cv::Vec3d> rightSpan =
            resampleSpan(controlPoints[static_cast<size_t>(changedSortedIndex)],
                         controlPoints[static_cast<size_t>(changedSortedIndex + 1)]);
        if (!replacement.empty() && !rightSpan.empty()) {
            replacement.insert(replacement.end(), rightSpan.begin() + 1, rightSpan.end());
        } else {
            replacement = std::move(rightSpan);
        }
    }

    const int eraseStart = std::clamp(static_cast<int>(std::ceil(replaceStartPosition)),
                                      0,
                                      static_cast<int>(linePoints.size()) - 1);
    const int eraseEnd = std::clamp(static_cast<int>(std::floor(replaceEndPosition)),
                                    eraseStart,
                                    static_cast<int>(linePoints.size()) - 1);

    std::vector<cv::Vec3d> updatedLine;
    updatedLine.reserve(linePoints.size() + replacement.size());
    updatedLine.insert(updatedLine.end(), linePoints.begin(), linePoints.begin() + eraseStart);
    updatedLine.insert(updatedLine.end(), replacement.begin(), replacement.end());
    updatedLine.insert(updatedLine.end(),
                       linePoints.begin() + static_cast<std::ptrdiff_t>(eraseEnd + 1),
                       linePoints.end());

    for (auto& control : controlPoints) {
        int bestIndex = 0;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < updatedLine.size(); ++i) {
            const double distance = length(updatedLine[i] - control.volumePoint);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = static_cast<int>(i);
            }
        }
        control.optimizedIndex = bestIndex;
        control.linePosition = static_cast<double>(bestIndex);
        control.volumePoint = updatedLine[static_cast<size_t>(bestIndex)];
    }

    LineControlPointUpdateResult result;
    result.linePoints = std::move(updatedLine);
    result.controlPoints = std::move(controlPoints);
    result.changedControlIndex = changedSortedIndex;
    const auto activeRange = activeRangeAroundControlSpans(result.controlPoints,
                                                          result.changedControlIndex,
                                                          static_cast<int>(result.linePoints.size()),
                                                          3);
    result.activeStart = activeRange.first;
    result.activeEnd = activeRange.second;
    return result;
}

LineControlPointUpdateResult updateExistingLineControlPoint(
    std::vector<cv::Vec3d> linePoints,
    std::vector<LineControlPoint> controlPoints,
    size_t changedControlIndex,
    const NormalSampler& sampler,
    const LineOptimizationConfig& rawConfig)
{
    const LineOptimizationConfig config = sanitizedConfig(rawConfig);
    if (linePoints.size() < 2) {
        throw std::invalid_argument("Existing line update requires at least two samples");
    }
    if (changedControlIndex >= controlPoints.size()) {
        throw std::invalid_argument("Changed control index is out of range");
    }

    std::vector<std::pair<LineControlPoint, bool>> taggedControls;
    taggedControls.reserve(controlPoints.size());
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        taggedControls.push_back({controlPoints[i], i == changedControlIndex});
    }
    std::stable_sort(taggedControls.begin(),
                     taggedControls.end(),
                     [](const auto& a, const auto& b) {
                         return a.first.linePosition < b.first.linePosition;
                     });

    int changedSortedIndex = -1;
    controlPoints.clear();
    controlPoints.reserve(taggedControls.size());
    const int inputMaxIndex = static_cast<int>(linePoints.size()) - 1;
    for (size_t i = 0; i < taggedControls.size(); ++i) {
        taggedControls[i].first.linePosition = std::clamp(
            taggedControls[i].first.linePosition,
            0.0,
            static_cast<double>(inputMaxIndex));
        if (taggedControls[i].second) {
            changedSortedIndex = static_cast<int>(i);
        }
        controlPoints.push_back(taggedControls[i].first);
    }
    if (changedSortedIndex < 0) {
        throw std::invalid_argument("Changed control point was not found after sorting");
    }

    const bool hasLeft = changedSortedIndex > 0;
    const bool hasRight = changedSortedIndex + 1 < static_cast<int>(controlPoints.size());
    if (!hasLeft && !hasRight) {
        LineControlPointUpdateResult unchanged;
        unchanged.linePoints = std::move(linePoints);
        unchanged.controlPoints = std::move(controlPoints);
        unchanged.changedControlIndex = changedSortedIndex;
        unchanged.activeStart = 0;
        unchanged.activeEnd = static_cast<int>(unchanged.linePoints.size()) - 1;
        return unchanged;
    }

    const double replaceStartPosition = hasLeft
        ? controlPoints[static_cast<size_t>(changedSortedIndex - 1)].linePosition
        : controlPoints[static_cast<size_t>(changedSortedIndex)].linePosition;
    const double replaceEndPosition = hasRight
        ? controlPoints[static_cast<size_t>(changedSortedIndex + 1)].linePosition
        : controlPoints[static_cast<size_t>(changedSortedIndex)].linePosition;

    std::vector<cv::Vec3d> replacement;
    std::vector<LineReinitializationSpanReport> initializedSpans;
    if (hasLeft) {
        const int leftIndex = changedSortedIndex - 1;
        OptimizedControlSpan leftSpan =
            optimizedControlSpan(controlPoints[static_cast<size_t>(leftIndex)].volumePoint,
                                 controlPoints[static_cast<size_t>(changedSortedIndex)].volumePoint,
                                 sampler,
                                 config,
                                 leftIndex,
                                 leftIndex,
                                 changedSortedIndex,
                                 "control-update-span");
        initializedSpans.push_back(leftSpan.report);
        replacement.reserve(leftSpan.points.size());
        for (const auto& point : leftSpan.points) {
            replacement.push_back(toVec3d(point));
        }
    }
    if (hasRight) {
        const int rightIndex = changedSortedIndex + 1;
        OptimizedControlSpan rightSpan =
            optimizedControlSpan(controlPoints[static_cast<size_t>(changedSortedIndex)].volumePoint,
                                 controlPoints[static_cast<size_t>(rightIndex)].volumePoint,
                                 sampler,
                                 config,
                                 changedSortedIndex,
                                 changedSortedIndex,
                                 rightIndex,
                                 "control-update-span");
        initializedSpans.push_back(rightSpan.report);
        if (!replacement.empty()) {
            for (size_t i = 1; i < rightSpan.points.size(); ++i) {
                replacement.push_back(toVec3d(rightSpan.points[i]));
            }
        } else {
            replacement.reserve(rightSpan.points.size());
            for (const auto& point : rightSpan.points) {
                replacement.push_back(toVec3d(point));
            }
        }
    }

    const int eraseStart = std::clamp(static_cast<int>(std::ceil(replaceStartPosition)),
                                      0,
                                      inputMaxIndex);
    const int eraseEnd = std::clamp(static_cast<int>(std::floor(replaceEndPosition)),
                                    eraseStart,
                                    inputMaxIndex);

    std::vector<cv::Vec3d> updatedLine;
    updatedLine.reserve(linePoints.size() + replacement.size());
    updatedLine.insert(updatedLine.end(), linePoints.begin(), linePoints.begin() + eraseStart);
    updatedLine.insert(updatedLine.end(), replacement.begin(), replacement.end());
    updatedLine.insert(updatedLine.end(),
                       linePoints.begin() + static_cast<std::ptrdiff_t>(eraseEnd + 1),
                       linePoints.end());

    for (auto& control : controlPoints) {
        int bestIndex = 0;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < updatedLine.size(); ++i) {
            const double distance = length(updatedLine[i] - control.volumePoint);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = static_cast<int>(i);
            }
        }
        control.optimizedIndex = bestIndex;
        control.linePosition = static_cast<double>(bestIndex);
        control.volumePoint = updatedLine[static_cast<size_t>(bestIndex)];
    }

    LineControlPointUpdateResult result;
    result.linePoints = std::move(updatedLine);
    result.controlPoints = std::move(controlPoints);
    result.initializedSpans = std::move(initializedSpans);
    result.changedControlIndex = changedSortedIndex;

    const int changed = result.changedControlIndex;
    const int lastControl = static_cast<int>(result.controlPoints.size()) - 1;
    if (changed == 0 && result.controlPoints.size() >= 2) {
        const LineControlPoint& first = result.controlPoints.front();
        const int firstIndex = std::clamp(first.optimizedIndex,
                                          0,
                                          static_cast<int>(result.linePoints.size()) - 1);
        if (firstIndex + 1 < static_cast<int>(result.linePoints.size())) {
            const cv::Vec3d outward = first.volumePoint -
                                      result.linePoints[static_cast<size_t>(firstIndex + 1)];
            std::vector<std::array<double, 3>> grown;
            growNormalConstructedExtension(first.volumePoint,
                                           projectedDirectionAtStart(first.volumePoint,
                                                                     outward,
                                                                     sampler),
                                           sampler,
                                           config,
                                           grown);

            std::vector<cv::Vec3d> expanded;
            expanded.reserve(grown.size() + result.linePoints.size() - static_cast<size_t>(firstIndex));
            for (auto it = grown.rbegin(); it != grown.rend(); ++it) {
                expanded.push_back(toVec3d(*it));
            }
            expanded.insert(expanded.end(),
                            result.linePoints.begin() + static_cast<std::ptrdiff_t>(firstIndex),
                            result.linePoints.end());

            const int shift = static_cast<int>(grown.size()) - firstIndex;
            for (auto& control : result.controlPoints) {
                control.optimizedIndex += shift;
                control.linePosition = static_cast<double>(control.optimizedIndex);
            }
            result.linePoints = std::move(expanded);
        }
    } else if (changed == lastControl && result.controlPoints.size() >= 2) {
        const LineControlPoint& last = result.controlPoints.back();
        const int lastIndex = std::clamp(last.optimizedIndex,
                                         0,
                                         static_cast<int>(result.linePoints.size()) - 1);
        if (lastIndex > 0) {
            const cv::Vec3d outward = last.volumePoint -
                                      result.linePoints[static_cast<size_t>(lastIndex - 1)];
            std::vector<std::array<double, 3>> grown;
            growNormalConstructedExtension(last.volumePoint,
                                           projectedDirectionAtStart(last.volumePoint,
                                                                     outward,
                                                                     sampler),
                                           sampler,
                                           config,
                                           grown);

            std::vector<cv::Vec3d> expanded;
            expanded.reserve(static_cast<size_t>(lastIndex + 1) + grown.size());
            expanded.insert(expanded.end(),
                            result.linePoints.begin(),
                            result.linePoints.begin() + static_cast<std::ptrdiff_t>(lastIndex + 1));
            for (const auto& point : grown) {
                expanded.push_back(toVec3d(point));
            }
            result.linePoints = std::move(expanded);
        }
    }

    const auto activeRange = activeRangeAroundControlSpans(result.controlPoints,
                                                          result.changedControlIndex,
                                                          static_cast<int>(result.linePoints.size()),
                                                          3);
    result.activeStart = activeRange.first;
    result.activeEnd = activeRange.second;
    return result;
}

LineOptimizer::LineOptimizer(const NormalSampler& normalSampler)
    : normalSampler_(normalSampler)
{
}

LineOptimizationResult LineOptimizer::optimizeFromSeed(
    const cv::Vec3d& seedPoint,
    const LineOptimizationConfig& rawConfig) const
{
    const LineOptimizationConfig config = sanitizedConfig(rawConfig);
    const NormalSample seedNormal = normalSampler_.sampleNormal(seedPoint);
    const cv::Vec3d tangent = initialTangentFromConfig(seedNormal, config);

    const int seedIndex = config.segmentsPerSide;

    const auto directNormalStart = Clock::now();
    auto directNormalInit = directNormalConstructedPoints(seedPoint, tangent, normalSampler_, config);
    auto spacingConstraints = fixedStepConstraints(directNormalInit.size() - 1);
    if (!config.runGlobalOptimization) {
        int finalValidSamples = 0;
        int finalInvalidSamples = 0;
        auto finalSamples = sampleSegments(
            directNormalInit,
            normalSampler_,
            config.samplesPerSegment,
            &finalValidSamples,
            &finalInvalidSamples);

        LineOptimizationResult result;
        result.line = buildLineModel(directNormalInit,
                                     normalSampler_,
                                     std::move(finalSamples),
                                     seedIndex);
        result.report.finalLosses = evaluateLosses(directNormalInit,
                                                   spacingConstraints,
                                                   normalSampler_,
                                                   config,
                                                   seedIndex,
                                                   tangent,
                                                   true);
        result.report.initialCost = totalWeightedCost(result.report.finalLosses);
        result.report.finalCost = result.report.initialCost;
        result.report.residuals = totalResidualCount(result.report.finalLosses);
        result.report.initialRms = ceresResidualRms(result.report.initialCost,
                                                   result.report.residuals);
        result.report.finalRms = result.report.initialRms;
        result.report.iterations = 0;
        result.report.validNormalSamples = finalValidSamples;
        result.report.invalidNormalSamples = finalInvalidSamples;
        result.report.converged = true;
        result.report.normalPrefetchCalls = 0;
        result.report.ceresSolveMs = 0.0;
        result.report.normalChunkPrefetchMs = 0.0;
        result.report.normalMaterializeMs = 0.0;
        result.report.totalMs = elapsedMs(directNormalStart, Clock::now());
        result.report.normalPrefetchRequestedChunks = 0;
        result.report.normalPrefetchChunksRead = 0;
        std::ostringstream message;
        message.imbue(std::locale::classic());
        message << std::scientific << std::setprecision(3)
                << "Line annotation Lasagna selected candidate:\n"
                << "candidate                   ms  iters     init_rms    final_rms\n"
                << std::left << std::setw(24) << "normal-construct-init"
                << std::right << std::setw(10) << elapsedMs(directNormalStart, Clock::now())
                << std::setw(7) << 0
                << std::setw(13) << result.report.initialRms
                << std::setw(13) << result.report.finalRms
                << "\n\nGlobal optimization disabled; returning normal-transport initialization.";
        result.report.message = message.str();
        return result;
    }

    const GlobalSolveResult selected = solveGlobalCandidate("normal-construct+global",
                                                            std::move(directNormalInit),
                                                            std::move(spacingConstraints),
                                                            {seedIndex},
                                                            normalSampler_,
                                                            config,
                                                            tangent,
                                                            true,
                                                            directNormalStart);

    int finalValidSamples = 0;
    int finalInvalidSamples = 0;
    auto finalSamples = sampleSegments(
        selected.points,
        normalSampler_,
        config.samplesPerSegment,
        &finalValidSamples,
        &finalInvalidSamples);

    LineOptimizationResult result;
    result.line = buildLineModel(selected.points,
                                 normalSampler_,
                                 std::move(finalSamples),
                                 seedIndex);
    result.report.initialCost = selected.initialCost;
    result.report.finalCost = selected.finalCost;
    result.report.initialRms = selected.initialRms;
    result.report.finalRms = selected.finalRms;
    result.report.residuals = selected.residuals;
    result.report.iterations = selected.iterations;
    result.report.validNormalSamples = finalValidSamples;
    result.report.invalidNormalSamples = finalInvalidSamples;
    result.report.converged = selected.usable;
    result.report.normalPrefetchCalls = selected.prefetchTiming.calls;
    result.report.ceresSolveMs = selected.ceresSolveMs;
    result.report.normalChunkPrefetchMs = selected.prefetchTiming.chunkPrefetchMs;
    result.report.normalMaterializeMs = selected.prefetchTiming.materializeMs;
    result.report.totalMs = selected.milliseconds;
    result.report.normalPrefetchRequestedChunks = selected.prefetchTiming.requestedChunks;
    result.report.normalPrefetchChunksRead = selected.prefetchTiming.chunksRead;
    std::ostringstream message;
    message.imbue(std::locale::classic());
    message << std::scientific << std::setprecision(3)
            << "Line annotation Lasagna selected candidate:\n"
            << "candidate                   ms  iters     init_rms    final_rms\n"
            << std::left << std::setw(24) << selected.name
            << std::right << std::setw(10) << selected.milliseconds
            << std::setw(7) << selected.iterations
            << std::setw(13) << selected.initialRms
            << std::setw(13) << selected.finalRms
            << "\n\nNormal prefetch/materialization:\n"
            << "calls=" << selected.prefetchTiming.calls
            << " ceres_solve_ms=" << selected.ceresSolveMs
            << " chunk_prefetch_ms=" << selected.prefetchTiming.chunkPrefetchMs
            << " materialize_ms=" << selected.prefetchTiming.materializeMs
            << " requested_chunks=" << selected.prefetchTiming.requestedChunks
            << " chunks_read=" << selected.prefetchTiming.chunksRead
            << " total_ms=" << (selected.prefetchTiming.chunkPrefetchMs +
                                selected.prefetchTiming.materializeMs)
            << "\n\nSelected candidate report:\n"
            << selected.report;
    result.report.message = message.str();
    result.report.finalLosses = evaluateLosses(selected.points,
                                               fixedStepConstraints(selected.points.size() - 1),
                                               normalSampler_,
                                               config,
                                               seedIndex,
                                               tangent,
                                               true);

    return result;
}

LineOptimizationResult LineOptimizer::optimizeFromSeeds(
    const std::vector<cv::Vec3d>& seedPoints,
    const LineOptimizationConfig& config) const
{
    if (seedPoints.empty()) {
        throw std::invalid_argument("LineOptimizer requires at least one seed point");
    }
    if (seedPoints.size() == 1) {
        return optimizeFromSeed(seedPoints.front(), config);
    }
    std::vector<LineControlPoint> controls;
    controls.reserve(seedPoints.size());
    for (size_t i = 0; i < seedPoints.size(); ++i) {
        controls.push_back({static_cast<double>(i), seedPoints[i], i == 0, -1});
    }
    return optimizeFromControlPoints(std::move(controls), config);
}

LineOptimizationResult LineOptimizer::optimizeExistingLine(
    std::vector<cv::Vec3d> linePoints,
    std::vector<int> fixedPointIndices,
    int displayFrameAnchorIndex,
    const LineOptimizationConfig& rawConfig,
    int activeStart,
    int activeEnd,
    std::string candidateName) const
{
    const LineOptimizationConfig config = sanitizedConfig(rawConfig);
    if (linePoints.size() < 2) {
        throw std::invalid_argument("Existing line optimization requires at least two samples");
    }
    if (candidateName.empty()) {
        candidateName = "existing-line+global";
    }

    std::vector<std::array<double, 3>> points;
    points.reserve(linePoints.size());
    for (const auto& point : linePoints) {
        points.push_back(toArray(point));
    }

    const int maxIndex = static_cast<int>(points.size()) - 1;
    activeStart = activeStart < 0 ? 0 : std::clamp(activeStart, 0, maxIndex);
    activeEnd = activeEnd < 0 ? maxIndex : std::clamp(activeEnd, activeStart, maxIndex);
    displayFrameAnchorIndex = std::clamp(displayFrameAnchorIndex, 0, maxIndex);

    fixedPointIndices.erase(std::remove_if(fixedPointIndices.begin(),
                                           fixedPointIndices.end(),
                                           [maxIndex](int index) {
                                               return index < 0 || index > maxIndex;
                                           }),
                            fixedPointIndices.end());
    if (fixedPointIndices.empty()) {
        fixedPointIndices.push_back(displayFrameAnchorIndex);
    }
    std::vector<int> controlAnchorIndices = fixedPointIndices;
    std::sort(controlAnchorIndices.begin(), controlAnchorIndices.end());
    controlAnchorIndices.erase(std::unique(controlAnchorIndices.begin(), controlAnchorIndices.end()),
                               controlAnchorIndices.end());

    if (activeStart > 0 || activeEnd < maxIndex) {
        fixedPointIndices.push_back(activeStart);
        fixedPointIndices.push_back(activeEnd);
    }
    std::sort(fixedPointIndices.begin(), fixedPointIndices.end());
    fixedPointIndices.erase(std::unique(fixedPointIndices.begin(), fixedPointIndices.end()),
                            fixedPointIndices.end());

    std::vector<SegmentSpacingConstraint> spacing(
        points.size() - 1,
        SegmentSpacingConstraint{SegmentSpacingMode::None, -1});
    if (controlAnchorIndices.empty()) {
        spacing = fixedStepConstraints(points.size() - 1);
    } else {
        const int firstAnchor = controlAnchorIndices.front();
        const int lastAnchor = controlAnchorIndices.back();
        for (int segment = 0; segment + 1 < static_cast<int>(points.size()); ++segment) {
            if (segment < firstAnchor || segment >= lastAnchor) {
                spacing[static_cast<size_t>(segment)] = {SegmentSpacingMode::FixedStep, -1};
            }
        }
        for (size_t spanIndex = 0; spanIndex + 1 < controlAnchorIndices.size(); ++spanIndex) {
            const int spanStart = controlAnchorIndices[spanIndex];
            const int spanEnd = controlAnchorIndices[spanIndex + 1];
            if (spanEnd <= spanStart) {
                continue;
            }
            const int spanId = static_cast<int>(spanIndex);
            for (int segment = spanStart; segment < spanEnd; ++segment) {
                spacing[static_cast<size_t>(segment)] = {
                    SegmentSpacingMode::EvenStep,
                    spanId,
                };
            }
        }
    }

    const int tangentLeft = std::max(0, displayFrameAnchorIndex - 1);
    const int tangentRight = std::min(maxIndex, displayFrameAnchorIndex + 1);
    cv::Vec3d seedTangent = tangentRight > tangentLeft
        ? normalizedOrZero(linePoints[static_cast<size_t>(tangentRight)] -
                           linePoints[static_cast<size_t>(tangentLeft)])
        : cv::Vec3d{1.0, 0.0, 0.0};
    if (length(seedTangent) <= kEpsilon) {
        seedTangent = {1.0, 0.0, 0.0};
    }

    const auto start = Clock::now();
    const GlobalSolveResult selected = solveGlobalCandidate(std::move(candidateName),
                                                            std::move(points),
                                                            spacing,
                                                            fixedPointIndices,
                                                            normalSampler_,
                                                            config,
                                                            seedTangent,
                                                            false,
                                                            start,
                                                            activeStart,
                                                            activeEnd);

    int finalValidSamples = 0;
    int finalInvalidSamples = 0;
    auto finalSamples = sampleSegments(selected.points,
                                       normalSampler_,
                                       config.samplesPerSegment,
                                       &finalValidSamples,
                                       &finalInvalidSamples);

    LineOptimizationResult result;
    result.line = buildLineModel(selected.points,
                                 normalSampler_,
                                 std::move(finalSamples),
                                 displayFrameAnchorIndex);
    result.report.initialCost = selected.initialCost;
    result.report.finalCost = selected.finalCost;
    result.report.initialRms = selected.initialRms;
    result.report.finalRms = selected.finalRms;
    result.report.residuals = selected.residuals;
    result.report.iterations = selected.iterations;
    result.report.validNormalSamples = finalValidSamples;
    result.report.invalidNormalSamples = finalInvalidSamples;
    result.report.converged = selected.usable;
    result.report.normalPrefetchCalls = selected.prefetchTiming.calls;
    result.report.ceresSolveMs = selected.ceresSolveMs;
    result.report.normalChunkPrefetchMs = selected.prefetchTiming.chunkPrefetchMs;
    result.report.normalMaterializeMs = selected.prefetchTiming.materializeMs;
    result.report.totalMs = selected.milliseconds;
    result.report.normalPrefetchRequestedChunks = selected.prefetchTiming.requestedChunks;
    result.report.normalPrefetchChunksRead = selected.prefetchTiming.chunksRead;
    result.report.finalLosses = evaluateLosses(selected.points,
                                               spacing,
                                               normalSampler_,
                                               config,
                                               displayFrameAnchorIndex,
                                               seedTangent,
                                               false);

    std::ostringstream message;
    message.imbue(std::locale::classic());
    message << std::scientific << std::setprecision(3)
            << "Line annotation Lasagna selected candidate:\n"
            << "candidate                   ms  iters     init_rms    final_rms\n"
            << std::left << std::setw(24) << selected.name
            << std::right << std::setw(10) << selected.milliseconds
            << std::setw(7) << selected.iterations
            << std::setw(13) << selected.initialRms
            << std::setw(13) << selected.finalRms
            << "\n\nExisting-line optimization:\n"
            << "active_points=[" << activeStart << ", " << activeEnd << "]"
            << " total_points=" << selected.points.size()
            << "\nfixed sample indices:";
    for (const int index : fixedPointIndices) {
        message << ' ' << index;
    }
    message << "\n\nNormal prefetch/materialization:\n"
            << "calls=" << selected.prefetchTiming.calls
            << " ceres_solve_ms=" << selected.ceresSolveMs
            << " chunk_prefetch_ms=" << selected.prefetchTiming.chunkPrefetchMs
            << " materialize_ms=" << selected.prefetchTiming.materializeMs
            << " requested_chunks=" << selected.prefetchTiming.requestedChunks
            << " chunks_read=" << selected.prefetchTiming.chunksRead
            << " total_ms=" << (selected.prefetchTiming.chunkPrefetchMs +
                                selected.prefetchTiming.materializeMs)
            << "\n\nSelected candidate report:\n"
            << selected.report;
    result.report.message = message.str();
    return result;
}

LineReinitializationOptimizationResult LineOptimizer::reinitializeAndOptimizeExistingLine(
    std::vector<cv::Vec3d> linePoints,
    std::vector<LineControlPoint> controlPoints,
    std::vector<int> fixedControlAnchorIndices,
    int displayFrameAnchorIndex,
    const LineOptimizationConfig& rawConfig) const
{
    const LineOptimizationConfig config = sanitizedConfig(rawConfig);
    if (linePoints.size() < 2) {
        throw std::invalid_argument("Existing line reinitialization requires at least two samples");
    }
    if (controlPoints.size() < 2) {
        throw std::invalid_argument("Existing line reinitialization requires at least two control points");
    }

    const int inputMaxIndex = static_cast<int>(linePoints.size()) - 1;
    fixedControlAnchorIndices.erase(
        std::remove_if(fixedControlAnchorIndices.begin(),
                       fixedControlAnchorIndices.end(),
                       [inputMaxIndex](int index) {
                           return index < 0 || index > inputMaxIndex;
                       }),
        fixedControlAnchorIndices.end());
    std::sort(fixedControlAnchorIndices.begin(), fixedControlAnchorIndices.end());
    fixedControlAnchorIndices.erase(std::unique(fixedControlAnchorIndices.begin(),
                                                fixedControlAnchorIndices.end()),
                                    fixedControlAnchorIndices.end());

    std::vector<bool> usedFixedAnchors(fixedControlAnchorIndices.size(), false);
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        auto& control = controlPoints[i];
        int matchedAnchor = -1;
        double matchedAnchorDistance = std::numeric_limits<double>::infinity();
        for (size_t anchorIndex = 0; anchorIndex < fixedControlAnchorIndices.size(); ++anchorIndex) {
            if (usedFixedAnchors[anchorIndex]) {
                continue;
            }
            const int lineIndex = fixedControlAnchorIndices[anchorIndex];
            const double candidateDistance =
                length(linePoints[static_cast<size_t>(lineIndex)] - control.volumePoint);
            if (candidateDistance < matchedAnchorDistance) {
                matchedAnchorDistance = candidateDistance;
                matchedAnchor = static_cast<int>(anchorIndex);
            }
        }
        if (matchedAnchor >= 0) {
            usedFixedAnchors[static_cast<size_t>(matchedAnchor)] = true;
            control.optimizedIndex = fixedControlAnchorIndices[static_cast<size_t>(matchedAnchor)];
            control.linePosition = static_cast<double>(control.optimizedIndex);
        } else if (control.optimizedIndex >= 0) {
            control.optimizedIndex = std::clamp(control.optimizedIndex, 0, inputMaxIndex);
            control.linePosition = static_cast<double>(control.optimizedIndex);
        } else {
            control.linePosition = std::clamp(control.linePosition, 0.0, static_cast<double>(inputMaxIndex));
            control.optimizedIndex = static_cast<int>(std::lround(control.linePosition));
        }
    }

    std::vector<std::pair<LineControlPoint, int>> sortedControls;
    sortedControls.reserve(controlPoints.size());
    for (size_t i = 0; i < controlPoints.size(); ++i) {
        sortedControls.push_back({controlPoints[i], static_cast<int>(i)});
    }
    std::stable_sort(sortedControls.begin(),
                     sortedControls.end(),
                     [](const auto& a, const auto& b) {
                         return a.first.linePosition < b.first.linePosition;
                     });

    controlPoints.clear();
    controlPoints.reserve(sortedControls.size());
    std::vector<int> originalControlIndices;
    originalControlIndices.reserve(sortedControls.size());
    for (const auto& tagged : sortedControls) {
        controlPoints.push_back(tagged.first);
        originalControlIndices.push_back(tagged.second);
    }

    std::vector<std::vector<std::array<double, 3>>> optimizedSpans(controlPoints.size() - 1);
    std::vector<LineReinitializationSpanReport> spanReports(controlPoints.size() - 1);
    std::vector<LineDebugPolyline> continuationCandidateLines;
    double maxCandidateFinalDiff = 0.0;
    double maxCandidateFinalRmsDiff = 0.0;
    double maxCandidateAlignmentScoreDiff = 0.0;

    int rolloutAnchorControlIndex = -1;
    for (size_t controlIndex = 0; controlIndex < controlPoints.size(); ++controlIndex) {
        if (controlPoints[controlIndex].isSeed) {
            rolloutAnchorControlIndex = static_cast<int>(controlIndex);
            break;
        }
    }
    if (rolloutAnchorControlIndex < 0) {
        double bestAnchorDistance = std::numeric_limits<double>::infinity();
        for (size_t controlIndex = 0; controlIndex < controlPoints.size(); ++controlIndex) {
            const double distance = std::abs(controlPoints[controlIndex].linePosition -
                                             static_cast<double>(displayFrameAnchorIndex));
            if (distance < bestAnchorDistance) {
                bestAnchorDistance = distance;
                rolloutAnchorControlIndex = static_cast<int>(controlIndex);
            }
        }
    }
    rolloutAnchorControlIndex = std::clamp(rolloutAnchorControlIndex,
                                           0,
                                           static_cast<int>(controlPoints.size()) - 1);

    const auto directionFromLeftSpan = [](const std::vector<std::array<double, 3>>& span)
        -> std::optional<cv::Vec3d> {
        if (span.size() < 2) {
            return std::nullopt;
        }
        return toVec3d(span.back()) - toVec3d(span[span.size() - 2]);
    };
    const auto directionFromRightSpan = [](const std::vector<std::array<double, 3>>& span)
        -> std::optional<cv::Vec3d> {
        if (span.size() < 2) {
            return std::nullopt;
        }
        return toVec3d(span.front()) - toVec3d(span[1]);
    };
    const auto buildPartialFailureResult =
        [&](size_t failedSpanIndex, const OptimizedControlSpan& failedSpan) {
            std::vector<bool> includeControl(controlPoints.size(), false);
            for (size_t spanIndex = 0; spanIndex < optimizedSpans.size(); ++spanIndex) {
                if (optimizedSpans[spanIndex].size() >= 2) {
                    includeControl[spanIndex] = true;
                    includeControl[spanIndex + 1] = true;
                }
            }
            includeControl[failedSpanIndex] = true;
            includeControl[failedSpanIndex + 1] = true;

            std::vector<cv::Vec3d> debugControlPoints;
            debugControlPoints.reserve(controlPoints.size());
            for (size_t controlIndex = 0; controlIndex < controlPoints.size(); ++controlIndex) {
                if (includeControl[controlIndex]) {
                    debugControlPoints.push_back(controlPoints[controlIndex].volumePoint);
                }
            }

            std::vector<cv::Vec3d> partialPoints;
            std::vector<int> partialFixedPointIndices;
            int previousSpanIndex = -1;
            for (size_t spanIndex = 0; spanIndex < optimizedSpans.size(); ++spanIndex) {
                const auto& span = optimizedSpans[spanIndex];
                if (span.size() < 2) {
                    continue;
                }
                const bool contiguous = previousSpanIndex >= 0 &&
                                        static_cast<size_t>(previousSpanIndex + 1) == spanIndex;
                if (partialPoints.empty()) {
                    partialFixedPointIndices.push_back(0);
                    for (const auto& point : span) {
                        partialPoints.push_back(toVec3d(point));
                    }
                } else if (contiguous) {
                    for (size_t pointIndex = 1; pointIndex < span.size(); ++pointIndex) {
                        partialPoints.push_back(toVec3d(span[pointIndex]));
                    }
                } else {
                    for (const auto& point : span) {
                        partialPoints.push_back(toVec3d(point));
                    }
                }
                partialFixedPointIndices.push_back(static_cast<int>(partialPoints.size()) - 1);
                previousSpanIndex = static_cast<int>(spanIndex);
            }
            if (partialPoints.size() < 2) {
                partialPoints = {
                    controlPoints[failedSpanIndex].volumePoint,
                    controlPoints[failedSpanIndex + 1].volumePoint,
                };
                partialFixedPointIndices = {0, 1};
            }

            std::vector<LineReinitializationSpanReport> partialReports;
            partialReports.reserve(spanReports.size());
            for (size_t spanIndex = 0; spanIndex < spanReports.size(); ++spanIndex) {
                if (optimizedSpans[spanIndex].size() >= 2 || spanIndex == failedSpanIndex) {
                    partialReports.push_back(spanReports[spanIndex]);
                }
            }

            LineReinitializationOptimizationResult partial;
            partial.failed = true;
            partial.failedSegmentIndex = static_cast<int>(failedSpanIndex);
            partial.failureReason = failedSpan.failureReason;
            partial.spans = std::move(partialReports);
            partial.maxSegmentCandidateFinalCostDiff = maxCandidateFinalDiff;
            partial.maxSegmentCandidateFinalRmsDiff = maxCandidateFinalRmsDiff;
            partial.maxSegmentCandidateAlignmentScoreDiff = maxCandidateAlignmentScoreDiff;
            partial.fixedPointIndices = std::move(partialFixedPointIndices);
            partial.debugControlPoints = std::move(debugControlPoints);
            partial.continuationCandidateLines = continuationCandidateLines;
            partial.optimization.line =
                buildLineModel(toArrayPoints(partialPoints),
                               normalSampler_,
                               {},
                               partial.fixedPointIndices.empty()
                                   ? 0
                                   : partial.fixedPointIndices[partial.fixedPointIndices.size() / 2]);
            partial.optimization.report.converged = false;
            partial.optimization.report.message = failedSpan.failureReason;
            return partial;
        };
    const auto initializeSpan = [&](size_t controlIndex,
                                    std::optional<cv::Vec3d> leftContinuationDirection,
                                    std::optional<cv::Vec3d> rightContinuationDirection)
        -> std::optional<LineReinitializationOptimizationResult> {
        const LineControlPoint& left = controlPoints[controlIndex];
        const LineControlPoint& right = controlPoints[controlIndex + 1];
        OptimizedControlSpan span =
            optimizedControlSpan(left.volumePoint,
                                 right.volumePoint,
                                 normalSampler_,
                                 config,
                                 static_cast<int>(controlIndex),
                                 originalControlIndices[controlIndex],
                                 originalControlIndices[controlIndex + 1],
                                 "reinit-span",
                                 leftContinuationDirection,
                                 rightContinuationDirection);
        maxCandidateFinalDiff = std::max(maxCandidateFinalDiff,
                                         span.candidateFinalCostDiff);
        maxCandidateFinalRmsDiff = std::max(maxCandidateFinalRmsDiff,
                                            span.candidateFinalRmsDiff);
        maxCandidateAlignmentScoreDiff = std::max(maxCandidateAlignmentScoreDiff,
                                                  span.candidateAlignmentScoreDiff);
        spanReports[controlIndex] = span.report;
        for (auto& candidateLine : span.continuationCandidateLines) {
            continuationCandidateLines.push_back(std::move(candidateLine));
        }
        if (span.failed) {
            return buildPartialFailureResult(controlIndex, span);
        }
        optimizedSpans[controlIndex] = std::move(span.points);
        return std::nullopt;
    };

    for (size_t controlIndex = static_cast<size_t>(rolloutAnchorControlIndex);
         controlIndex + 1 < controlPoints.size();
         ++controlIndex) {
        std::optional<cv::Vec3d> leftContinuationDirection;
        if (controlIndex > 0) {
            leftContinuationDirection = directionFromLeftSpan(optimizedSpans[controlIndex - 1]);
        }
        if (auto failure = initializeSpan(controlIndex, leftContinuationDirection, std::nullopt)) {
            return *failure;
        }
    }

    for (size_t controlIndex = static_cast<size_t>(rolloutAnchorControlIndex);
         controlIndex > 0;
         --controlIndex) {
        const size_t spanIndex = controlIndex - 1;
        std::optional<cv::Vec3d> rightContinuationDirection;
        if (spanIndex + 1 < optimizedSpans.size()) {
            rightContinuationDirection = directionFromRightSpan(optimizedSpans[spanIndex + 1]);
        }
        if (auto failure = initializeSpan(spanIndex, std::nullopt, rightContinuationDirection)) {
            return *failure;
        }
    }

    std::vector<cv::Vec3d> stitchedInternal;
    std::vector<int> internalControlIndices;
    internalControlIndices.reserve(controlPoints.size());
    for (size_t spanIndex = 0; spanIndex < optimizedSpans.size(); ++spanIndex) {
        const auto& span = optimizedSpans[spanIndex];
        if (span.size() < 2) {
            throw std::runtime_error("Existing line reinitialization produced a degenerate control span");
        }
        if (spanIndex == 0) {
            internalControlIndices.push_back(0);
            stitchedInternal.reserve(span.size() * optimizedSpans.size());
            for (const auto& point : span) {
                stitchedInternal.push_back(toVec3d(point));
            }
        } else {
            for (size_t pointIndex = 1; pointIndex < span.size(); ++pointIndex) {
                stitchedInternal.push_back(toVec3d(span[pointIndex]));
            }
        }
        internalControlIndices.push_back(static_cast<int>(stitchedInternal.size()) - 1);
    }

    if (stitchedInternal.size() < 2 ||
        internalControlIndices.size() != controlPoints.size()) {
        throw std::runtime_error("Existing line reinitialization produced fewer than two stitched samples");
    }

    std::vector<cv::Vec3d> leftOpenExtension;
    {
        std::vector<std::array<double, 3>> grown;
        const int firstControlIndex = internalControlIndices.front();
        const cv::Vec3d outward = stitchedInternal[static_cast<size_t>(firstControlIndex)] -
                                  stitchedInternal[static_cast<size_t>(firstControlIndex + 1)];
        growNormalConstructedExtension(stitchedInternal[static_cast<size_t>(firstControlIndex)],
                                       projectedDirectionAtStart(stitchedInternal[static_cast<size_t>(firstControlIndex)],
                                                                 outward,
                                                                 normalSampler_),
                                       normalSampler_,
                                       config,
                                       grown);
        leftOpenExtension.reserve(grown.size());
        for (auto it = grown.rbegin(); it != grown.rend(); ++it) {
            leftOpenExtension.push_back(toVec3d(*it));
        }
    }

    std::vector<cv::Vec3d> rightOpenExtension;
    {
        std::vector<std::array<double, 3>> grown;
        const int lastControlIndex = internalControlIndices.back();
        const cv::Vec3d outward = stitchedInternal[static_cast<size_t>(lastControlIndex)] -
                                  stitchedInternal[static_cast<size_t>(lastControlIndex - 1)];
        growNormalConstructedExtension(stitchedInternal[static_cast<size_t>(lastControlIndex)],
                                       projectedDirectionAtStart(stitchedInternal[static_cast<size_t>(lastControlIndex)],
                                                                 outward,
                                                                 normalSampler_),
                                       normalSampler_,
                                       config,
                                       grown);
        rightOpenExtension.reserve(grown.size());
        for (const auto& point : grown) {
            rightOpenExtension.push_back(toVec3d(point));
        }
    }

    std::vector<cv::Vec3d> stitched;
    stitched.reserve(leftOpenExtension.size() + stitchedInternal.size() + rightOpenExtension.size());
    stitched.insert(stitched.end(), leftOpenExtension.begin(), leftOpenExtension.end());
    const int internalOffset = static_cast<int>(stitched.size());
    stitched.insert(stitched.end(), stitchedInternal.begin(), stitchedInternal.end());
    stitched.insert(stitched.end(), rightOpenExtension.begin(), rightOpenExtension.end());

    std::vector<int> stitchedControlIndices;
    stitchedControlIndices.reserve(internalControlIndices.size());
    for (const int index : internalControlIndices) {
        stitchedControlIndices.push_back(internalOffset + index);
    }

    if (stitched.size() < 2) {
        throw std::runtime_error("Existing line reinitialization produced fewer than two samples");
    }

    int displayControlIndex = 0;
    double displayControlDistance = std::numeric_limits<double>::infinity();
    displayFrameAnchorIndex = std::clamp(displayFrameAnchorIndex, 0, inputMaxIndex);
    for (size_t controlIndex = 0; controlIndex < controlPoints.size(); ++controlIndex) {
        const double candidate = std::abs(controlPoints[controlIndex].linePosition -
                                          static_cast<double>(displayFrameAnchorIndex));
        if (candidate < displayControlDistance) {
            displayControlDistance = candidate;
            displayControlIndex = static_cast<int>(controlIndex);
        }
    }
    const int stitchedDisplayFrameAnchorIndex =
        stitchedControlIndices.empty()
            ? 0
            : stitchedControlIndices[static_cast<size_t>(
                  std::clamp(displayControlIndex, 0, static_cast<int>(stitchedControlIndices.size()) - 1))];

    LineOptimizationResult finalOptimization =
        optimizeExistingLine(stitched,
                             stitchedControlIndices,
                             stitchedDisplayFrameAnchorIndex,
                             config,
                             -1,
                             -1,
                             "reinit-reopt+global");

    std::ostringstream message;
    message.imbue(std::locale::classic());
    message << std::scientific << std::setprecision(1)
            << "Reinitialized control spans:\n"
            << std::right
            << std::setw(3) << "seg"
            << std::setw(4) << "lcp"
            << std::setw(4) << "rcp"
            << std::setw(4) << "pts"
            << std::setw(5) << "lsgn"
            << std::setw(8) << "lnear"
            << std::setw(5) << "rsgn"
            << std::setw(8) << "rnear"
            << std::setw(8) << "clnear"
            << std::setw(8) << "crnear"
            << std::setw(9) << "lirms"
            << std::setw(9) << "lfrms"
            << std::setw(9) << "rirms"
            << std::setw(9) << "rfrms"
            << std::setw(9) << "clfrms"
            << std::setw(9) << "crfrms"
            << std::setw(8) << "mxstep"
            << std::setw(8) << "mxtan"
            << std::setw(8) << "mxnorm"
            << std::setw(8) << "mxalign"
            << ' ' << "pick" << '\n';
    for (const auto& report : spanReports) {
        message << std::right << std::setw(3) << report.segmentIndex
                << std::setw(4) << report.leftControlIndex
                << std::setw(4) << report.rightControlIndex
                << std::setw(4) << report.points
                << std::setw(5) << report.candLeftSelectedSign
                << std::setw(8) << report.candLeftClosestTargetDistance
                << std::setw(5) << report.candRightSelectedSign
                << std::setw(8) << report.candRightClosestTargetDistance
                << std::setw(8) << report.candContinueLeftClosestTargetDistance
                << std::setw(8) << report.candContinueRightClosestTargetDistance
                << std::setw(9) << initialRmsForCandidate(report, "left")
                << std::setw(9) << finalRmsForCandidate(report, "left")
                << std::setw(9) << initialRmsForCandidate(report, "right")
                << std::setw(9) << finalRmsForCandidate(report, "right")
                << std::setw(9) << finalRmsForCandidate(report, "continue-left")
                << std::setw(9) << finalRmsForCandidate(report, "continue-right")
                << std::setw(8) << report.chosenMaxEvenStepDeviation
                << std::setw(8) << report.chosenMaxTangentSmoothDeviation
                << std::setw(8) << report.chosenMaxNormalSmoothDeviation
                << std::setw(8) << report.chosenMaxNormalAlignmentAbs
                << ' ' << report.chosen << '\n';
    }
    message << "max_segment_candidate_alignment_score_diff="
            << maxCandidateAlignmentScoreDiff << "\n\n"
            << finalOptimization.report.message;
    finalOptimization.report.message = message.str();

    LineReinitializationOptimizationResult result;
    result.optimization = std::move(finalOptimization);
    result.spans = std::move(spanReports);
    result.maxSegmentCandidateFinalCostDiff = maxCandidateFinalDiff;
    result.maxSegmentCandidateFinalRmsDiff = maxCandidateFinalRmsDiff;
    result.maxSegmentCandidateAlignmentScoreDiff = maxCandidateAlignmentScoreDiff;
    result.fixedPointIndices = std::move(stitchedControlIndices);
    result.continuationCandidateLines = std::move(continuationCandidateLines);
    return result;
}

LineOptimizationResult LineOptimizer::optimizeFromControlPoints(
    std::vector<LineControlPoint> controlPoints,
    const LineOptimizationConfig& rawConfig) const
{
    const LineOptimizationConfig config = sanitizedConfig(rawConfig);
    if (controlPoints.empty()) {
        throw std::invalid_argument("LineOptimizer requires at least one control point");
    }
    if (controlPoints.size() == 1) {
        return optimizeFromSeed(controlPoints.front().volumePoint, config);
    }

    const auto start = Clock::now();
    ControlPointInitialization init = initializeFromControlPoints(std::move(controlPoints), normalSampler_, config);
    const std::vector<SegmentSpacingConstraint> finalSpacing = init.spacingConstraints;
    const std::vector<int> fixedPointIndices = init.fixedPointIndices;
    int seedIndex = fixedPointIndices.empty() ? config.segmentsPerSide : fixedPointIndices.front();
    for (const auto& control : init.controlPoints) {
        if (control.isSeed && control.optimizedIndex >= 0) {
            seedIndex = control.optimizedIndex;
            break;
        }
    }
    const cv::Vec3d seedTangent = init.seedTangent;

    GlobalSolveResult selected;
    bool usedLocalWindow = false;
    int localProblemStart = 0;
    int localProblemEnd = static_cast<int>(init.points.size()) - 1;
    int leftBoundaryControlIndex = 0;
    int rightBoundaryControlIndex = static_cast<int>(init.controlPoints.size()) - 1;

    if (canInitializeFromExistingLine(init.controlPoints, config)) {
        int minMovedControlIndex = static_cast<int>(init.controlPoints.size());
        int maxMovedControlIndex = -1;
        for (size_t controlIndex = 0; controlIndex < init.controlPoints.size(); ++controlIndex) {
            const auto& control = init.controlPoints[controlIndex];
            const cv::Vec3d base = interpolateInitialLinePoint(config.initialLinePoints,
                                                               control.linePosition);
            if (length(control.volumePoint - base) > kMovedControlDistanceThreshold) {
                const int index = static_cast<int>(controlIndex);
                minMovedControlIndex = std::min(minMovedControlIndex, index);
                maxMovedControlIndex = std::max(maxMovedControlIndex, index);
            }
        }

        if (maxMovedControlIndex >= 0) {
            leftBoundaryControlIndex = std::max(
                0,
                minMovedControlIndex - kLocalControlOptimizationSegments);
            rightBoundaryControlIndex = std::min(
                static_cast<int>(init.controlPoints.size()) - 1,
                maxMovedControlIndex + kLocalControlOptimizationSegments);
            const int leftIndex =
                init.controlPoints[static_cast<size_t>(leftBoundaryControlIndex)].optimizedIndex;
            const int rightIndex =
                init.controlPoints[static_cast<size_t>(rightBoundaryControlIndex)].optimizedIndex;
            if (leftIndex >= 0 && rightIndex >= leftIndex) {
                localProblemStart = std::max(0, leftIndex - 1);
                localProblemEnd = std::min(static_cast<int>(init.points.size()) - 1,
                                           rightIndex + 1);
                usedLocalWindow = localProblemEnd > localProblemStart;
            }
        }
    }

    if (usedLocalWindow) {
        std::vector<int> localFixedIndices{localProblemStart, localProblemEnd};
        for (const auto& control : init.controlPoints) {
            if (control.optimizedIndex >= localProblemStart &&
                control.optimizedIndex <= localProblemEnd) {
                localFixedIndices.push_back(control.optimizedIndex);
            }
        }
        std::sort(localFixedIndices.begin(), localFixedIndices.end());
        localFixedIndices.erase(std::unique(localFixedIndices.begin(), localFixedIndices.end()),
                                localFixedIndices.end());

        selected = solveGlobalCandidate("control-points-local",
                                        std::move(init.points),
                                        std::move(init.spacingConstraints),
                                        localFixedIndices,
                                        normalSampler_,
                                        config,
                                        seedTangent,
                                        false,
                                        start,
                                        localProblemStart,
                                        localProblemEnd);
    } else {
        selected = solveGlobalCandidate("control-points+global",
                                        std::move(init.points),
                                        std::move(init.spacingConstraints),
                                        fixedPointIndices,
                                        normalSampler_,
                                        config,
                                        seedTangent,
                                        false,
                                        start);
    }

    int finalValidSamples = 0;
    int finalInvalidSamples = 0;
    auto finalSamples = sampleSegments(
        selected.points,
        normalSampler_,
        config.samplesPerSegment,
        &finalValidSamples,
        &finalInvalidSamples);

    LineOptimizationResult result;
    int displayFrameAnchorIndex = seedIndex;
    for (const auto& control : init.controlPoints) {
        if (control.isSeed && control.optimizedIndex >= 0) {
            displayFrameAnchorIndex = control.optimizedIndex;
            break;
        }
    }
    result.line = buildLineModel(selected.points,
                                 normalSampler_,
                                 std::move(finalSamples),
                                 displayFrameAnchorIndex);
    result.report.initialCost = selected.initialCost;
    result.report.finalCost = selected.finalCost;
    result.report.initialRms = selected.initialRms;
    result.report.finalRms = selected.finalRms;
    result.report.residuals = selected.residuals;
    result.report.iterations = selected.iterations;
    result.report.validNormalSamples = finalValidSamples;
    result.report.invalidNormalSamples = finalInvalidSamples;
    result.report.converged = selected.usable;
    result.report.normalPrefetchCalls = selected.prefetchTiming.calls;
    result.report.ceresSolveMs = selected.ceresSolveMs;
    result.report.normalChunkPrefetchMs = selected.prefetchTiming.chunkPrefetchMs;
    result.report.normalMaterializeMs = selected.prefetchTiming.materializeMs;
    result.report.totalMs = selected.milliseconds;
    result.report.normalPrefetchRequestedChunks = selected.prefetchTiming.requestedChunks;
    result.report.normalPrefetchChunksRead = selected.prefetchTiming.chunksRead;

    std::ostringstream message;
    message.imbue(std::locale::classic());
    message << std::scientific << std::setprecision(3)
            << "Line annotation Lasagna selected candidate:\n"
            << "candidate                   ms  iters     init_rms    final_rms\n"
            << std::left << std::setw(24) << selected.name
            << std::right << std::setw(10) << selected.milliseconds
            << std::setw(7) << selected.iterations
            << std::setw(13) << selected.initialRms
            << std::setw(13) << selected.finalRms
            << "\n\nFixed control point indices:";
    for (const int index : fixedPointIndices) {
        message << ' ' << index;
    }
    if (usedLocalWindow) {
        message << "\n\nLocal control-point window:\n"
                << "control_indices=[" << leftBoundaryControlIndex << ", "
                << rightBoundaryControlIndex << "]"
                << " point_indices=[" << localProblemStart << ", "
                << localProblemEnd << "]"
                << " active_points=" << (localProblemEnd - localProblemStart + 1)
                << " total_points=" << selected.points.size();
    }
    message << "\n\nNormal prefetch/materialization:\n"
            << "calls=" << selected.prefetchTiming.calls
            << " ceres_solve_ms=" << selected.ceresSolveMs
            << " chunk_prefetch_ms=" << selected.prefetchTiming.chunkPrefetchMs
            << " materialize_ms=" << selected.prefetchTiming.materializeMs
            << " requested_chunks=" << selected.prefetchTiming.requestedChunks
            << " chunks_read=" << selected.prefetchTiming.chunksRead
            << " total_ms=" << (selected.prefetchTiming.chunkPrefetchMs +
                                selected.prefetchTiming.materializeMs)
            << "\n\nSelected candidate report:\n"
            << selected.report;
    result.report.message = message.str();
    result.report.finalLosses = evaluateLosses(selected.points,
                                               finalSpacing,
                                               normalSampler_,
                                               config,
                                               seedIndex,
                                               seedTangent,
                                               false);

    return result;
}

} // namespace vc::lasagna
