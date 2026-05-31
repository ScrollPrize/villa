#include "vc/lasagna/LineOptimizer.hpp"

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vc::lasagna {
namespace {

constexpr double kEpsilon = 1.0e-12;

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
    config.normalAlignmentWeight = std::max(0.0, config.normalAlignmentWeight);
    config.distanceWeight = std::max(0.0, config.distanceWeight);
    config.initialTangentWeight = std::max(0.0, config.initialTangentWeight);
    config.tangentGuideWeight = std::max(0.0, config.tangentGuideWeight);
    config.samplesPerSegment = std::max(1, config.samplesPerSegment);
    config.maxIterations = std::max(0, config.maxIterations);
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

struct StraightnessResidual {
    explicit StraightnessResidual(double weight)
        : weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const prev, const T* const point, const T* const next, T* residuals) const
    {
        for (int axis = 0; axis < 3; ++axis) {
            residuals[axis] = T(weight) * (prev[axis] - T(2.0) * point[axis] + next[axis]);
        }
        return true;
    }

    double weight = 1.0;
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

[[nodiscard]] std::vector<LineOptimizationLossReport> evaluateLosses(
    const std::vector<std::array<double, 3>>& points,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    int seedIndex,
    const cv::Vec3d& seedTangent)
{
    LossAccumulator distance{"distance", config.distanceWeight};
    LossAccumulator straightness{"straightness", config.straightnessWeight};
    LossAccumulator normalAlignment{"normal_alignment", config.normalAlignmentWeight};
    LossAccumulator initialDirection{"initial_direction", config.initialTangentWeight};
    LossAccumulator tangentGuide{"tangent_guide", config.tangentGuideWeight};
    const bool useTangentGuide = config.tangentGuideMode != LineOptimizationConfig::TangentGuideMode::None;
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
        distance.add(std::sqrt((b - a).dot(b - a) + kEpsilon) - config.segmentLength);
    }

    for (size_t i = 1; i + 1 < points.size(); ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            straightness.add(points[i - 1][axis] - 2.0 * points[i][axis] + points[i + 1][axis]);
        }
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

    if (config.useInitialTangent &&
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
    losses.reserve(4);
    losses.push_back(toReport(distance));
    losses.push_back(toReport(straightness));
    losses.push_back(toReport(normalAlignment));
    losses.push_back(toReport(initialDirection));
    losses.push_back(toReport(tangentGuide));
    return losses;
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
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    int seedIndex,
    const cv::Vec3d& seedTangent)
{
    for (size_t i = 0; i + 1 < points.size(); ++i) {
        auto* cost = new ceres::AutoDiffCostFunction<DistanceResidual, 1, 3, 3>(
            new DistanceResidual(config.segmentLength, config.distanceWeight));
        problem.AddResidualBlock(cost, nullptr, points[i].data(), points[i + 1].data());
    }

    for (size_t i = 1; i + 1 < points.size(); ++i) {
        auto* cost = new ceres::AutoDiffCostFunction<StraightnessResidual, 3, 3, 3, 3>(
            new StraightnessResidual(config.straightnessWeight));
        problem.AddResidualBlock(cost, nullptr, points[i - 1].data(), points[i].data(), points[i + 1].data());
    }

    for (size_t segment = 0; segment + 1 < points.size(); ++segment) {
        for (int sample = 0; sample <= config.samplesPerSegment; ++sample) {
            const double t = static_cast<double>(sample) /
                             static_cast<double>(config.samplesPerSegment);
            auto* cost = new LiveNormalAlignmentResidual(sampler, t, config.normalAlignmentWeight);
            problem.AddResidualBlock(cost, nullptr, points[segment].data(), points[segment + 1].data());

            const bool seedAdjacent = static_cast<int>(segment) == seedIndex ||
                                      static_cast<int>(segment + 1) == seedIndex;
            if (seedAdjacent &&
                config.tangentGuideMode != LineOptimizationConfig::TangentGuideMode::None &&
                config.tangentGuideWeight > 0.0) {
                auto* guideCost = new LiveTangentGuideResidual(sampler, t, config);
                problem.AddResidualBlock(guideCost, nullptr, points[segment].data(), points[segment + 1].data());
            }
        }
    }

    if (config.useInitialTangent &&
        config.initialTangentWeight > 0.0 &&
        length(seedTangent) > kEpsilon &&
        seedIndex > 0 &&
        seedIndex + 1 < static_cast<int>(points.size())) {
        auto* prevCost = new ceres::AutoDiffCostFunction<DirectionResidual, 3, 3, 3>(
            new DirectionResidual(seedTangent, config.initialTangentWeight));
        problem.AddResidualBlock(prevCost, nullptr, points[seedIndex - 1].data(), points[seedIndex].data());

        auto* nextCost = new ceres::AutoDiffCostFunction<DirectionResidual, 3, 3, 3>(
            new DirectionResidual(seedTangent, config.initialTangentWeight));
        problem.AddResidualBlock(nextCost, nullptr, points[seedIndex].data(), points[seedIndex + 1].data());
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

    if (previous != nullptr && config.straightnessWeight > 0.0) {
        auto* straightnessCost = new ceres::AutoDiffCostFunction<StraightnessResidual, 3, 3, 3, 3>(
            new StraightnessResidual(config.straightnessWeight));
        problem.AddResidualBlock(straightnessCost, nullptr, previous->data(), fixed.data(), moving.data());
    }

    for (int sample = 0; sample <= config.samplesPerSegment; ++sample) {
        const double t = static_cast<double>(sample) /
                         static_cast<double>(config.samplesPerSegment);
        auto* normalCost = new LiveNormalAlignmentResidual(sampler, t, config.normalAlignmentWeight);
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
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = progress;
    options.logging_type = progress ? ceres::PER_MINIMIZER_ITERATION : ceres::SILENT;
    options.num_threads = 1;
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
        for (int i = 0; i < config.segmentsPerSide; ++i) {
            const cv::Vec3d predicted = point + direction * config.segmentLength;
            const NormalSample sample = sampler.sampleNormal(predicted);
            const cv::Vec3d normal = normalizedOrZero(sample.normal);
            if (sample.valid && length(normal) > kEpsilon) {
                direction = projectDirectionToNormalPlane(direction, normal);
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

struct GlobalSolveResult {
    std::string name;
    std::vector<std::array<double, 3>> points;
    double initialCost = 0.0;
    double finalCost = 0.0;
    int iterations = 0;
    bool usable = false;
    double milliseconds = 0.0;
    std::string report;
};

[[nodiscard]] GlobalSolveResult solveGlobalCandidate(
    std::string name,
    std::vector<std::array<double, 3>> initialPoints,
    const NormalSampler& sampler,
    const LineOptimizationConfig& config,
    const cv::Vec3d& seedTangent,
    Clock::time_point chainStart)
{
    const int seedIndex = config.segmentsPerSide;

    ceres::Problem problem;
    for (auto& point : initialPoints) {
        problem.AddParameterBlock(point.data(), 3);
    }
    problem.SetParameterBlockConstant(initialPoints[static_cast<size_t>(seedIndex)].data());
    addResiduals(problem, initialPoints, sampler, config, seedIndex, seedTangent);

    double initialCost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions{}, &initialCost, nullptr, nullptr, nullptr);

    ceres::Solver::Summary summary;
    ceres::Solve(solverOptions(config, true), &problem, &summary);

    GlobalSolveResult result;
    result.name = std::move(name);
    result.points = std::move(initialPoints);
    result.initialCost = initialCost;
    result.finalCost = summary.final_cost;
    result.iterations = static_cast<int>(summary.iterations.size());
    result.usable = summary.IsSolutionUsable();
    result.milliseconds = elapsedMs(chainStart, Clock::now());
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
        << "candidate                   ms  iters    init_cost   final_cost   rms_vs_inc   max_vs_inc\n";
    const auto& reference = results.front();
    for (const auto& result : results) {
        const LineDifference diff = compareLines(reference.points, result.points);
        out << std::left << std::setw(24) << result.name
            << std::right << std::setw(10) << result.milliseconds
            << std::setw(7) << result.iterations
            << std::setw(13) << result.initialCost
            << std::setw(13) << result.finalCost
            << std::setw(13) << diff.rms
            << std::setw(13) << diff.max
            << '\n';
    }
    return out.str();
}

[[nodiscard]] LineModel buildLineModel(
    const std::vector<std::array<double, 3>>& points,
    const NormalSampler& sampler,
    std::vector<LineSegmentSamples> segmentSamples)
{
    LineModel model;
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

} // namespace

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

    std::vector<GlobalSolveResult> candidates;
    candidates.reserve(3);

    const auto sequentialStart = Clock::now();
    SequentialSolveResult sequential = growSequentialLine(seedPoint, tangent, normalSampler_, config);
    candidates.push_back(solveGlobalCandidate("incremental+global",
                                              sequential.points,
                                              normalSampler_,
                                              config,
                                              tangent,
                                              sequentialStart));

    const auto straightStart = Clock::now();
    auto straightInit = straightLinePoints(seedPoint, tangent, config);
    candidates.push_back(solveGlobalCandidate("straight+global",
                                              std::move(straightInit),
                                              normalSampler_,
                                              config,
                                              tangent,
                                              straightStart));

    const auto directNormalStart = Clock::now();
    auto directNormalInit = directNormalConstructedPoints(seedPoint, tangent, normalSampler_, config);
    candidates.push_back(solveGlobalCandidate("normal-construct+global",
                                              std::move(directNormalInit),
                                              normalSampler_,
                                              config,
                                              tangent,
                                              directNormalStart));

    const auto& selected = candidates[2];

    int finalValidSamples = 0;
    int finalInvalidSamples = 0;
    auto finalSamples = sampleSegments(
        selected.points,
        normalSampler_,
        config.samplesPerSegment,
        &finalValidSamples,
        &finalInvalidSamples);

    LineOptimizationResult result;
    result.line = buildLineModel(selected.points, normalSampler_, std::move(finalSamples));
    result.report.initialCost = selected.initialCost;
    result.report.finalCost = selected.finalCost;
    result.report.iterations = selected.iterations;
    result.report.validNormalSamples = finalValidSamples;
    result.report.invalidNormalSamples = finalInvalidSamples;
    result.report.converged = selected.usable;
    result.report.message = comparisonReport(candidates) + "\nSelected candidate report:\n" + selected.report;
    result.report.finalLosses = evaluateLosses(selected.points, normalSampler_, config, seedIndex, tangent);

    return result;
}

LineOptimizationResult LineOptimizer::optimizeFromSeeds(
    const std::vector<cv::Vec3d>& seedPoints,
    const LineOptimizationConfig& config) const
{
    if (seedPoints.size() != 1) {
        throw std::invalid_argument("LineOptimizer V1 requires exactly one seed point");
    }
    return optimizeFromSeed(seedPoints.front(), config);
}

} // namespace vc::lasagna
