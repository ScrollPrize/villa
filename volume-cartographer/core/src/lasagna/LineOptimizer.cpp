#include "vc/lasagna/LineOptimizer.hpp"

#include <ceres/ceres.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
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

    const int pointCount = config.segmentsPerSide * 2 + 1;
    const int seedIndex = config.segmentsPerSide;

    std::vector<std::array<double, 3>> points;
    points.reserve(static_cast<size_t>(pointCount));
    for (int i = 0; i < pointCount; ++i) {
        const double offset = static_cast<double>(i - seedIndex) * config.segmentLength;
        const cv::Vec3d point = seedPoint + tangent * offset;
        points.push_back({point[0], point[1], point[2]});
    }

    ceres::Problem problem;
    for (auto& point : points) {
        problem.AddParameterBlock(point.data(), 3);
    }
    problem.SetParameterBlockConstant(points[seedIndex].data());
    addResiduals(problem, points, normalSampler_, config, seedIndex, tangent);

    double initialCost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions{}, &initialCost, nullptr, nullptr, nullptr);

    ceres::Solver::Options options;
    options.max_num_iterations = config.maxIterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    options.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    int finalValidSamples = 0;
    int finalInvalidSamples = 0;
    auto finalSamples = sampleSegments(
        points,
        normalSampler_,
        config.samplesPerSegment,
        &finalValidSamples,
        &finalInvalidSamples);

    LineOptimizationResult result;
    result.line = buildLineModel(points, normalSampler_, std::move(finalSamples));
    result.report.initialCost = initialCost;
    result.report.finalCost = summary.final_cost;
    result.report.iterations = static_cast<int>(summary.iterations.size());
    result.report.validNormalSamples = finalValidSamples;
    result.report.invalidNormalSamples = finalInvalidSamples;
    result.report.converged = summary.IsSolutionUsable();
    result.report.message = summary.BriefReport();

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
