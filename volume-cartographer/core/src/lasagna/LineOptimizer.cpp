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
    config.samplesPerSegment = std::max(1, config.samplesPerSegment);
    config.maxIterations = std::max(0, config.maxIterations);
    return config;
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

struct NormalAlignmentResidual {
    NormalAlignmentResidual(cv::Vec3d normal, double weight)
        : normal(normalizedOrZero(normal))
        , weight(weight)
    {
    }

    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const
    {
        const T dx = b[0] - a[0];
        const T dy = b[1] - a[1];
        const T dz = b[2] - a[2];
        const T invLength = T(1.0) / ceres::sqrt(dx * dx + dy * dy + dz * dz + T(kEpsilon));
        const T dot = (dx * T(normal[0]) + dy * T(normal[1]) + dz * T(normal[2])) * invLength;
        residual[0] = T(weight) * dot;
        return true;
    }

    cv::Vec3d normal{0.0, 0.0, 0.0};
    double weight = 1.0;
};

void addResiduals(
    ceres::Problem& problem,
    std::vector<std::array<double, 3>>& points,
    const std::vector<LineSegmentSamples>& initialSamples,
    const LineOptimizationConfig& config)
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

    for (size_t segment = 0; segment < initialSamples.size(); ++segment) {
        for (const auto& sample : initialSamples[segment].samples) {
            if (!sample.sampledNormal.valid) {
                continue;
            }

            auto* cost = new ceres::AutoDiffCostFunction<NormalAlignmentResidual, 1, 3, 3>(
                new NormalAlignmentResidual(sample.sampledNormal.normal, config.normalAlignmentWeight));
            problem.AddResidualBlock(cost, nullptr, points[segment].data(), points[segment + 1].data());
        }
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
    const cv::Vec3d tangent = deterministicTangentFromNormal(seedNormal);

    const int pointCount = config.segmentsPerSide * 2 + 1;
    const int seedIndex = config.segmentsPerSide;

    std::vector<std::array<double, 3>> points;
    points.reserve(static_cast<size_t>(pointCount));
    for (int i = 0; i < pointCount; ++i) {
        const double offset = static_cast<double>(i - seedIndex) * config.segmentLength;
        const cv::Vec3d point = seedPoint + tangent * offset;
        points.push_back({point[0], point[1], point[2]});
    }

    const auto initialSamples = sampleSegments(
        points,
        normalSampler_,
        config.samplesPerSegment,
        nullptr,
        nullptr);

    ceres::Problem problem;
    for (auto& point : points) {
        problem.AddParameterBlock(point.data(), 3);
    }
    problem.SetParameterBlockConstant(points[seedIndex].data());
    addResiduals(problem, points, initialSamples, config);

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
