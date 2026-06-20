#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineOptimizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

class ConstantNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit ConstantNormalSampler(cv::Vec3d normal)
        : normal_(normal)
    {
    }

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& /*volumePoint*/) const override
    {
        return {normal_, true, {}};
    }

private:
    cv::Vec3d normal_;
};

class MissingNormalSampler final : public vc::lasagna::NormalSampler {
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& /*volumePoint*/) const override
    {
        return {{0.0, 0.0, 0.0}, false, "missing"};
    }
};

class CountingNormalSampler final : public vc::lasagna::NormalSampler {
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& volumePoint) const override
    {
        ++calls;
        sampledPoints.push_back(volumePoint);
        return {{0.0, 0.0, 1.0}, true, {}};
    }

    mutable int calls = 0;
    mutable std::vector<cv::Vec3d> sampledPoints;
};

class StepXThenZNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit StepXThenZNormalSampler(double xThreshold)
        : xThreshold_(xThreshold)
    {
    }

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& volumePoint) const override
    {
        if (volumePoint[0] < xThreshold_) {
            return {{1.0, 0.0, 0.0}, true, {}};
        }
        return {{0.0, 0.0, 1.0}, true, {}};
    }

private:
    double xThreshold_ = 0.0;
};

class BatchOnlyZNormalSampler final : public vc::lasagna::NormalSampler {
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& /*volumePoint*/) const override
    {
        return {{1.0, 0.0, 0.0}, true, {}};
    }

    vc::lasagna::NormalBatchReport sampleNormalBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        bool /*withDerivative*/,
        std::vector<vc::lasagna::NormalSampleWithDerivative>& samples) const override
    {
        ++batchCalls;
        batchSampledPoints.insert(batchSampledPoints.end(), volumePoints.begin(), volumePoints.end());
        samples.assign(volumePoints.size(),
                       vc::lasagna::NormalSampleWithDerivative{
                           {{0.0, 0.0, 1.0}, true, {}},
                           cv::Matx33d::zeros(),
                           false});
        return {};
    }

    mutable int batchCalls = 0;
    mutable std::vector<cv::Vec3d> batchSampledPoints;
};

class SeedThenTiltedNormalSampler final : public vc::lasagna::NormalSampler {
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& volumePoint) const override
    {
        if (std::sqrt(volumePoint.dot(volumePoint)) < 1.0e-9) {
            return {{0.0, 0.0, 1.0}, true, {}};
        }
        const double invSqrt2 = 1.0 / std::sqrt(2.0);
        return {{invSqrt2, 0.0, invSqrt2}, true, {}};
    }
};

double norm(const cv::Vec3d& vector)
{
    return std::sqrt(vector.dot(vector));
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& vector)
{
    const double n = norm(vector);
    if (n <= 1.0e-12 ||
        !std::isfinite(vector[0]) ||
        !std::isfinite(vector[1]) ||
        !std::isfinite(vector[2])) {
        return {0.0, 0.0, 0.0};
    }
    return vector * (1.0 / n);
}

double dot(const cv::Vec3d& a, const cv::Vec3d& b)
{
    return a.dot(b);
}

double angleDegrees(const cv::Vec3d& a, const cv::Vec3d& b)
{
    const double an = norm(a);
    const double bn = norm(b);
    if (an <= 1.0e-12 || bn <= 1.0e-12) {
        return 0.0;
    }
    return std::acos(std::clamp(dot(a, b) / (an * bn), -1.0, 1.0)) * 180.0 / 3.14159265358979323846;
}

double nearestLinePosition(const std::vector<cv::Vec3d>& points, const cv::Vec3d& query)
{
    if (points.size() < 2) {
        return 0.0;
    }
    double bestPosition = 0.0;
    double bestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < points.size(); ++i) {
        const cv::Vec3d a = points[i];
        const cv::Vec3d b = points[i + 1];
        const cv::Vec3d ab = b - a;
        const double denom = ab.dot(ab);
        const double t = denom <= 1.0e-12
            ? 0.0
            : std::clamp((query - a).dot(ab) / denom, 0.0, 1.0);
        const cv::Vec3d closest = a + ab * t;
        const double distance = norm(query - closest);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestPosition = static_cast<double>(i) + t;
        }
    }
    return bestPosition;
}

struct LocalJumpMetrics {
    int controlIndex = -1;
    double nearestControlDistance = std::numeric_limits<double>::infinity();
    double maxLengthRatio = 1.0;
    double maxTurnDegrees = 0.0;
};

LocalJumpMetrics localJumpMetrics(const vc::lasagna::LineModel& line, const cv::Vec3d& controlPoint)
{
    LocalJumpMetrics metrics;
    for (size_t i = 0; i < line.points.size(); ++i) {
        const double distance = norm(line.points[i].position - controlPoint);
        if (distance < metrics.nearestControlDistance) {
            metrics.nearestControlDistance = distance;
            metrics.controlIndex = static_cast<int>(i);
        }
    }
    if (metrics.controlIndex < 0) {
        return metrics;
    }

    for (int segment = metrics.controlIndex - 4; segment <= metrics.controlIndex + 3; ++segment) {
        if (segment <= 0 || segment + 1 >= static_cast<int>(line.points.size())) {
            continue;
        }
        const cv::Vec3d before = line.points[static_cast<size_t>(segment)].position -
                                 line.points[static_cast<size_t>(segment - 1)].position;
        const cv::Vec3d after = line.points[static_cast<size_t>(segment + 1)].position -
                                line.points[static_cast<size_t>(segment)].position;
        const double beforeLength = norm(before);
        const double afterLength = norm(after);
        if (beforeLength > 1.0e-12 && afterLength > 1.0e-12) {
            metrics.maxLengthRatio = std::max(metrics.maxLengthRatio,
                                              std::max(beforeLength / afterLength,
                                                       afterLength / beforeLength));
            metrics.maxTurnDegrees = std::max(metrics.maxTurnDegrees, angleDegrees(before, after));
        }
    }
    return metrics;
}

std::vector<cv::Vec3d> readVec3Array(const nlohmann::json& root, const char* key)
{
    std::vector<cv::Vec3d> points;
    for (const auto& item : root.at(key)) {
        points.push_back({item.at(0).get<double>(),
                          item.at(1).get<double>(),
                          item.at(2).get<double>()});
    }
    return points;
}

vc::lasagna::LineModel lineModelFromPoints(const std::vector<cv::Vec3d>& points)
{
    vc::lasagna::LineModel line;
    line.points.reserve(points.size());
    for (const auto& point : points) {
        vc::lasagna::LinePoint linePoint;
        linePoint.position = point;
        line.points.push_back(linePoint);
    }
    return line;
}

const vc::lasagna::LineOptimizationLossReport& lossByName(
    const vc::lasagna::LineOptimizationReport& report,
    const std::string& name)
{
    for (const auto& loss : report.finalLosses) {
        if (loss.name == name) {
            return loss;
        }
    }
    throw std::runtime_error("missing loss " + name);
}

} // namespace

TEST_CASE("LineOptimizer grows a centered line tangent to sampled normals")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 10;
    config.segmentLength = 50.0;
    config.samplesPerSegment = 4;

    const auto result = optimizer.optimizeFromSeed({100.0, 200.0, 30.0}, config);

    REQUIRE(result.line.points.size() == 21);
    CHECK(result.line.displayFrameAnchorIndex == 10);
    REQUIRE(result.line.segmentSamples.size() == 20);
    CHECK(result.report.converged);
    CHECK(result.report.validNormalSamples == 20 * 5);
    CHECK(result.report.invalidNormalSamples == 0);
    REQUIRE(result.report.finalLosses.size() == 7);
    double weightedCost = 0.0;
    for (const auto& loss : result.report.finalLosses) {
        CHECK(loss.residuals >= 0);
        CHECK(loss.rawCost >= 0.0);
        CHECK(loss.weightedCost >= 0.0);
        weightedCost += loss.weightedCost;
    }
    CHECK(weightedCost == doctest::Approx(result.report.finalCost).epsilon(1.0e-8));

    const auto& seedPoint = result.line.points[10].position;
    CHECK(seedPoint[0] == doctest::Approx(100.0));
    CHECK(seedPoint[1] == doctest::Approx(200.0));
    CHECK(seedPoint[2] == doctest::Approx(30.0));

    for (size_t i = 0; i + 1 < result.line.points.size(); ++i) {
        const cv::Vec3d delta = result.line.points[i + 1].position - result.line.points[i].position;
        CHECK(norm(delta) == doctest::Approx(50.0).epsilon(1.0e-8));
        CHECK(delta.dot(cv::Vec3d{0.0, 0.0, 1.0}) == doctest::Approx(0.0).epsilon(1.0e-8));
    }

    for (const auto& segment : result.line.segmentSamples) {
        REQUIRE(segment.samples.size() == 5);
        CHECK(segment.samples.front().t == doctest::Approx(0.0));
        CHECK(segment.samples.back().t == doctest::Approx(1.0));
        CHECK(segment.samples[1].t == doctest::Approx(0.25));
        CHECK(segment.samples[2].t == doctest::Approx(0.5));
        CHECK(segment.samples[3].t == doctest::Approx(0.75));
    }
}

TEST_CASE("LineOptimizer follows live tangent guide inside sampled tangent plane")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 10.0;
    config.straightnessWeight = 0.0;
    config.normalAlignmentWeight = 0.0;
    config.tangentGuideMode = vc::lasagna::LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane;
    config.tangentGuideVector = {0.0, 1.0, 0.0};
    config.tangentGuideWeight = 10.0;
    config.useInitialTangent = true;
    config.initialTangent = {1.0, 0.0, 0.0};
    config.initialTangentWeight = 0.0;

    const auto result = optimizer.optimizeFromSeed({0.0, 0.0, 0.0}, config);

    REQUIRE(result.line.points.size() == 3);
    const cv::Vec3d prevDelta = result.line.points[1].position - result.line.points[0].position;
    const cv::Vec3d nextDelta = result.line.points[2].position - result.line.points[1].position;
    CHECK(std::abs(prevDelta[0]) < 1.0e-3);
    CHECK(std::abs(nextDelta[0]) < 1.0e-3);
    CHECK(std::abs(prevDelta[1]) == doctest::Approx(10.0).epsilon(1.0e-5));
    CHECK(std::abs(nextDelta[1]) == doctest::Approx(10.0).epsilon(1.0e-5));
}

TEST_CASE("LineOptimizer completes with missing normals and reports invalid samples")
{
    MissingNormalSampler sampler;
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 2;
    config.segmentLength = 50.0;
    config.samplesPerSegment = 4;

    const auto result = optimizer.optimizeFromSeed({0.0, 0.0, 0.0}, config);

    REQUIRE(result.line.points.size() == 5);
    REQUIRE(result.line.segmentSamples.size() == 4);
    CHECK(result.report.validNormalSamples == 0);
    CHECK(result.report.invalidNormalSamples == 4 * 5);
    CHECK(result.report.converged);
}

TEST_CASE("LineOptimizer supports multiple fixed control points")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 10.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 5;
    config.useInitialTangent = true;
    config.initialTangent = {0.0, 1.0, 0.0};
    config.tangentGuideMode = vc::lasagna::LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane;
    config.tangentGuideVector = {0.0, 1.0, 0.0};

    std::vector<vc::lasagna::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, true, -1},
        {3.0, {30.0, 0.0, 0.0}, false, -1},
    };
    const auto result = optimizer.optimizeFromControlPoints(controls, config);

    REQUIRE(result.line.points.size() == 6);
    CHECK(result.line.displayFrameAnchorIndex == 1);
    CHECK(result.line.points[1].position[0] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.line.points[1].position[1] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.line.points[1].position[2] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.line.points[4].position[0] == doctest::Approx(30.0).epsilon(1.0e-9));
    CHECK(result.line.points[4].position[1] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.line.points[4].position[2] == doctest::Approx(0.0).epsilon(1.0e-9));

    CHECK(lossByName(result.report, "step_distance").residuals == 2);
    CHECK(lossByName(result.report, "even_step").residuals == 0);
    CHECK(lossByName(result.report, "tangent_straightness").residuals > 0);
    CHECK(lossByName(result.report, "normal_straightness").residuals > 0);
    CHECK(lossByName(result.report, "initial_direction").residuals == 0);
    CHECK(lossByName(result.report, "tangent_guide").residuals == 0);

    const auto fromSeeds = optimizer.optimizeFromSeeds({{0.0, 0.0, 0.0}, {30.0, 0.0, 0.0}}, config);
    REQUIRE(fromSeeds.line.points.size() == result.line.points.size());
    CHECK_THROWS_AS(optimizer.optimizeFromSeeds({}), std::invalid_argument);
}

TEST_CASE("LineOptimizer reoptimizes only a local existing-line window for added controls")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 10.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 20;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 1.0;
    config.normalStraightnessWeight = 1.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    for (int i = 0; i <= 20; ++i) {
        config.initialLinePoints.push_back({static_cast<double>(i) * 10.0, 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls;
    for (int i = 0; i <= 20; i += 2) {
        controls.push_back({static_cast<double>(i),
                            config.initialLinePoints[static_cast<size_t>(i)],
                            i == 10,
                            -1});
    }
    controls[5].volumePoint += cv::Vec3d{0.0, 0.0, 5.0};
    auto baselineConfig = config;
    baselineConfig.maxIterations = 0;
    const auto baseline = optimizer.optimizeFromControlPoints(controls, baselineConfig);
    const auto result = optimizer.optimizeFromControlPoints(std::move(controls), config);

    REQUIRE(result.line.points.size() == baseline.line.points.size());
    CHECK(result.report.message.find("control-points-local") != std::string::npos);
    CHECK(result.report.message.find("control_indices=[2, 8]") != std::string::npos);
    CHECK(result.line.points[11].position[2] == doctest::Approx(5.0).epsilon(1.0e-9));

    for (int i = 0; i < 4; ++i) {
        CHECK(norm(result.line.points[static_cast<size_t>(i)].position -
                   baseline.line.points[static_cast<size_t>(i)].position) ==
              doctest::Approx(0.0).epsilon(1.0e-12));
    }
    for (size_t i = 19; i < result.line.points.size(); ++i) {
        CHECK(norm(result.line.points[static_cast<size_t>(i)].position -
                   baseline.line.points[static_cast<size_t>(i)].position) ==
              doctest::Approx(0.0).epsilon(1.0e-12));
    }
}

TEST_CASE("LineOptimizer resamples fractional control insertion spans evenly")
{
    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 10; ++i) {
        linePoints.push_back({static_cast<double>(i), 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls{
        {0.0, linePoints[0], true, 0},
        {5.0, linePoints[5], false, 5},
        {10.0, linePoints[10], false, 10},
    };
    controls.push_back({2.25, {2.25, 0.0, 1.0}, false, -1});

    const auto update = vc::lasagna::updateExistingLineControlPoint(linePoints,
                                                                    std::move(controls),
                                                                    3,
                                                                    1.0);

    REQUIRE(update.linePoints.size() == linePoints.size());
    REQUIRE(update.controlPoints.size() == 4);
    CHECK(update.controlPoints[1].linePosition == doctest::Approx(2.0));
    CHECK(update.linePoints[2][0] == doctest::Approx(2.25));
    CHECK(update.linePoints[2][2] == doctest::Approx(1.0));
    CHECK(update.linePoints[1][0] == doctest::Approx(1.125));
    CHECK(update.linePoints[1][2] == doctest::Approx(0.5));
    CHECK(update.linePoints[3][0] - update.linePoints[2][0] == doctest::Approx((5.0 - 2.25) / 3.0));
    for (size_t i = 5; i < linePoints.size(); ++i) {
        CHECK(norm(update.linePoints[i] - linePoints[i]) == doctest::Approx(0.0));
    }
}

TEST_CASE("LineOptimizer local update range covers three neighboring control spans")
{
    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 40; ++i) {
        linePoints.push_back({static_cast<double>(i), 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls;
    for (int i = 0; i <= 40; i += 5) {
        controls.push_back({static_cast<double>(i),
                            linePoints[static_cast<size_t>(i)],
                            i == 20,
                            i});
    }
    controls[4].volumePoint = {20.0, 1.0, 0.0};

    const auto update = vc::lasagna::updateExistingLineControlPoint(linePoints,
                                                                    std::move(controls),
                                                                    4,
                                                                    1.0);

    REQUIRE(update.controlPoints.size() == 9);
    CHECK(update.changedControlIndex == 4);
    CHECK(update.activeStart == update.controlPoints[1].optimizedIndex);
    CHECK(update.activeEnd == update.controlPoints[7].optimizedIndex);
}

TEST_CASE("LineOptimizer normal-aware control update uses two-sided span trials")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 40; ++i) {
        linePoints.push_back({static_cast<double>(i), 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls;
    for (int i = 0; i <= 40; i += 5) {
        controls.push_back({static_cast<double>(i),
                            linePoints[static_cast<size_t>(i)],
                            i == 20,
                            i});
    }
    controls[4].volumePoint = {20.0, 1.0, 0.0};

    vc::lasagna::LineOptimizationConfig config;
    config.segmentLength = 1.0;
    config.segmentsPerSide = 3;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    const auto update = vc::lasagna::updateExistingLineControlPoint(linePoints,
                                                                    std::move(controls),
                                                                    4,
                                                                    sampler,
                                                                    config);

    REQUIRE(update.controlPoints.size() == 9);
    REQUIRE(update.initializedSpans.size() == 2);
    CHECK(update.changedControlIndex == 4);
    CHECK(update.initializedSpans[0].leftControlIndex == 3);
    CHECK(update.initializedSpans[0].rightControlIndex == 4);
    CHECK(update.initializedSpans[1].leftControlIndex == 4);
    CHECK(update.initializedSpans[1].rightControlIndex == 5);
    for (const auto& span : update.initializedSpans) {
        CHECK(span.candLeftSelectedSign != 0);
        CHECK(span.candRightSelectedSign != 0);
        CHECK(std::isfinite(span.candLeftClosestTargetDistance));
        CHECK(std::isfinite(span.candRightClosestTargetDistance));
        CHECK(span.points >= 2);
        CHECK(span.candidates.size() >= 2);
    }
    CHECK(update.activeStart == update.controlPoints[1].optimizedIndex);
    CHECK(update.activeEnd == update.controlPoints[7].optimizedIndex);
    CHECK(norm(update.linePoints[static_cast<size_t>(update.controlPoints[4].optimizedIndex)] -
               cv::Vec3d{20.0, 1.0, 0.0}) == doctest::Approx(0.0).epsilon(1.0e-12));
}

TEST_CASE("LineOptimizer open-end control update grows a new extension")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 10; ++i) {
        const double staleY = i > 7 ? 100.0 : 0.0;
        linePoints.push_back({static_cast<double>(i), staleY, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls{
        {5.0, linePoints[5], true, 5},
        {7.0, {7.0, 0.0, 0.0}, false, -1},
    };

    vc::lasagna::LineOptimizationConfig config;
    config.segmentLength = 1.0;
    config.segmentsPerSide = 3;

    const auto update = vc::lasagna::updateExistingLineControlPoint(linePoints,
                                                                    std::move(controls),
                                                                    1,
                                                                    sampler,
                                                                    config);

    REQUIRE(update.linePoints.size() == 11);
    REQUIRE(update.controlPoints.size() == 2);
    CHECK(update.changedControlIndex == 1);
    CHECK(update.controlPoints[1].optimizedIndex == 7);
    CHECK(update.activeStart == 0);
    CHECK(update.activeEnd == static_cast<int>(update.linePoints.size()) - 1);
    CHECK(update.linePoints[8][0] == doctest::Approx(8.0));
    CHECK(update.linePoints[8][1] == doctest::Approx(0.0));
    CHECK(update.linePoints[9][1] == doctest::Approx(0.0));
    CHECK(update.linePoints[10][1] == doctest::Approx(0.0));
}

TEST_CASE("LineOptimizer two-control first-control update optimizes both open ends")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 10; ++i) {
        linePoints.push_back({static_cast<double>(i), 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls{
        {3.0, {3.0, 0.0, 0.0}, false, -1},
        {5.0, linePoints[5], true, 5},
    };

    vc::lasagna::LineOptimizationConfig config;
    config.segmentLength = 1.0;
    config.segmentsPerSide = 3;

    const auto update = vc::lasagna::updateExistingLineControlPoint(linePoints,
                                                                    std::move(controls),
                                                                    0,
                                                                    sampler,
                                                                    config);

    REQUIRE(update.linePoints.size() == linePoints.size());
    REQUIRE(update.controlPoints.size() == 2);
    CHECK(update.changedControlIndex == 0);
    CHECK(update.activeStart == 0);
    CHECK(update.activeEnd == static_cast<int>(update.linePoints.size()) - 1);
}

TEST_CASE("LineOptimizer full existing-line solve uses current samples directly")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentLength = 1.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.25},
        {2.0, 0.0, 0.0},
        {3.0, 0.0, -0.25},
        {4.0, 0.0, 0.0},
    };
    const auto result = optimizer.optimizeExistingLine(linePoints,
                                                       {1, 3},
                                                       1,
                                                       config,
                                                       -1,
                                                       -1,
                                                       "existing-line+global");

    REQUIRE(result.line.points.size() == linePoints.size());
    CHECK(result.report.message.find("existing-line+global") != std::string::npos);
    CHECK(result.report.message.find("control-points+global") == std::string::npos);
    CHECK(result.report.message.find("normal-construct") == std::string::npos);
    CHECK(lossByName(result.report, "step_distance").residuals == 2);
    CHECK(lossByName(result.report, "even_step").residuals == 1);
    for (size_t i = 0; i < linePoints.size(); ++i) {
        CHECK(norm(result.line.points[i].position - linePoints[i]) == doctest::Approx(0.0));
    }
}

TEST_CASE("LineOptimizer reinit reopt reports rollout candidates and preserves fixed controls")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 2;
    config.segmentLength = 100.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 10; ++i) {
        linePoints.push_back({static_cast<double>(i) * 50.0, 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls{
        {5.0, {250.0, 0.0, 0.0}, true, 5},
        {0.0, {0.0, 0.0, 0.0}, false, 0},
        {10.0, {500.0, 0.0, 0.0}, false, 10},
    };

    const auto result = optimizer.reinitializeAndOptimizeExistingLine(linePoints,
                                                                      controls,
                                                                      {0, 5, 10},
                                                                      5,
                                                                      config);

    REQUIRE(result.spans.size() == 2);
    CHECK(result.initialSeedSpanIndex == 1);
    CHECK(result.spans[0].candLeftRolloutSteps == 10);
    CHECK(result.spans[0].candRightRolloutSteps == 10);
    CHECK(result.spans[0].candLeftTruncatedPoints == 3);
    CHECK(result.spans[0].candRightTruncatedPoints == 3);
    CHECK(result.spans[0].candLeftSelectedSign == 1);
    CHECK(result.spans[0].candRightSelectedSign == 1);
    CHECK(result.spans[0].candLeftClosestTargetDistance == doctest::Approx(50.0));
    CHECK(result.spans[0].candRightClosestTargetDistance == doctest::Approx(50.0));
    CHECK(result.spans[0].points == 4);
    CHECK(result.spans[0].leftControlIndex == 1);
    CHECK(result.spans[0].rightControlIndex == 0);
    CHECK(result.spans[0].chosen == "left");
    REQUIRE(result.spans[0].candidates.size() >= 2);
    CHECK(result.spans[0].candidates[0].name == "left");
    CHECK(result.spans[0].candidates[1].name == "right");
    CHECK(std::any_of(result.spans[0].candidates.begin(),
                      result.spans[0].candidates.end(),
                      [](const auto& candidate) {
                          return candidate.name == "existing";
                      }));
    int chosenCandidateCount = 0;
    double bestAlignmentScore = std::numeric_limits<double>::infinity();
    double chosenAlignmentScore = std::numeric_limits<double>::infinity();
    for (const auto& candidate : result.spans[0].candidates) {
        chosenCandidateCount += candidate.chosen ? 1 : 0;
        if (candidate.usable) {
            bestAlignmentScore = std::min(bestAlignmentScore, candidate.alignmentChoiceScore);
        } else {
            CHECK_FALSE(candidate.chosen);
        }
        if (candidate.chosen) {
            CHECK(candidate.usable);
            chosenAlignmentScore = candidate.alignmentChoiceScore;
        }
        CHECK(candidate.points >= 2);
        CHECK(std::isfinite(candidate.finalCost));
        CHECK(candidate.finalCostDelta >= 0.0);
        CHECK(std::isfinite(candidate.finalRms));
        CHECK(candidate.finalRmsDelta >= 0.0);
        CHECK(std::isfinite(candidate.avgNormalAlignmentAbs));
        CHECK(std::isfinite(candidate.p95NormalAlignmentAbs));
        CHECK(std::isfinite(candidate.maxNormalAlignmentAbs));
        CHECK(std::isfinite(candidate.alignmentChoiceScore));
        CHECK(candidate.alignmentChoiceScoreDelta >= 0.0);
        CHECK(candidate.avgNormalAlignmentAbs <= candidate.maxNormalAlignmentAbs + 1.0e-12);
        CHECK(candidate.p95NormalAlignmentAbs <= candidate.maxNormalAlignmentAbs + 1.0e-12);
        CHECK(candidate.alignmentChoiceScore ==
              doctest::Approx((candidate.avgNormalAlignmentAbs +
                               candidate.p95NormalAlignmentAbs +
                               candidate.maxNormalAlignmentAbs) / 3.0));
    }
    CHECK(chosenCandidateCount == 1);
    CHECK(chosenAlignmentScore == doctest::Approx(bestAlignmentScore));
    CHECK(result.maxSegmentCandidateFinalCostDiff >= 0.0);

    REQUIRE(result.fixedPointIndices.size() == 3);
    CHECK(result.fixedPointIndices[0] == 2);
    CHECK(result.fixedPointIndices[1] == 5);
    CHECK(result.fixedPointIndices[2] == 8);

    REQUIRE(result.optimization.line.points.size() == 11);
    CHECK(result.optimization.line.displayFrameAnchorIndex == 5);
    CHECK(norm(result.optimization.line.points[2].position - controls[1].volumePoint) ==
          doctest::Approx(0.0).epsilon(1.0e-12));
    CHECK(norm(result.optimization.line.points[5].position - controls[0].volumePoint) ==
          doctest::Approx(0.0).epsilon(1.0e-12));
    CHECK(norm(result.optimization.line.points[8].position - controls[2].volumePoint) ==
          doctest::Approx(0.0).epsilon(1.0e-12));
    CHECK(result.optimization.line.points.front().position[0] < controls[1].volumePoint[0]);
    CHECK(result.optimization.line.points.back().position[0] > controls[2].volumePoint[0]);
    for (size_t i = 1; i < result.optimization.line.points.size(); ++i) {
        CHECK(norm(result.optimization.line.points[i].position -
                   result.optimization.line.points[i - 1].position) > 1.0e-9);
    }
    CHECK(result.optimization.report.message.find("Reinitialized control spans") != std::string::npos);
    CHECK(result.optimization.report.message.find("lsgn") != std::string::npos);
    CHECK(result.optimization.report.message.find("reinit-reopt+global") != std::string::npos);
}

TEST_CASE("LineOptimizer reinitialization adds continuation candidates on both sides of seed")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 2;
    config.segmentLength = 100.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 15; ++i) {
        linePoints.push_back({static_cast<double>(i) * 50.0, 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, false, 0},
        {5.0, {250.0, 0.0, 0.0}, true, 5},
        {10.0, {500.0, 0.0, 0.0}, false, 10},
        {15.0, {750.0, 0.0, 0.0}, false, 15},
    };

    const auto result = optimizer.reinitializeAndOptimizeExistingLine(linePoints,
                                                                      controls,
                                                                      {0, 5, 10, 15},
                                                                      5,
                                                                      config);

    REQUIRE(result.spans.size() == 3);
    CHECK(result.initialSeedSpanIndex == 1);
    CHECK(result.spans[0].candContinueLeftRolloutSteps == 0);
    CHECK(result.spans[0].candContinueRightRolloutSteps > 0);
    CHECK(result.spans[1].candContinueLeftRolloutSteps > 0);
    CHECK(result.spans[1].candContinueRightRolloutSteps == 0);
    CHECK(result.spans[2].candContinueLeftRolloutSteps > 0);
    CHECK(result.spans[2].candContinueRightRolloutSteps == 0);
    CHECK(std::isfinite(result.spans[0].candContinueRightClosestTargetDistance));
    CHECK(std::isfinite(result.spans[1].candContinueLeftClosestTargetDistance));
    CHECK(std::isfinite(result.spans[2].candContinueLeftClosestTargetDistance));
    CHECK(result.spans[0].candidates.size() == 4);
    CHECK(result.spans[1].candidates.size() == 4);
    CHECK(result.spans[2].candidates.size() == 4);
    CHECK(result.spans[0].candidates[2].name == "existing");
    CHECK(result.spans[1].candidates[2].name == "existing");
    CHECK(result.spans[2].candidates[2].name == "existing");
    CHECK(result.spans[0].candidates.back().name == "continue-right");
    CHECK(result.spans[1].candidates.back().name == "continue-left");
    CHECK(result.spans[2].candidates.back().name == "continue-left");
    REQUIRE(result.continuationCandidateLines.size() == 3);
    CHECK(result.continuationCandidateLines[0].name == "reinit-span-continue-left-1");
    CHECK(result.continuationCandidateLines[1].name == "reinit-span-continue-left-2");
    CHECK(result.continuationCandidateLines[2].name == "reinit-span-continue-right-0");
    for (const auto& line : result.continuationCandidateLines) {
        CHECK(line.points.size() >= 2);
    }
    for (const auto& span : result.spans) {
        for (const auto& candidate : span.candidates) {
            if (candidate.chosen) {
                CHECK(candidate.usable);
            }
        }
    }
}

TEST_CASE("LineOptimizer seedless reinitialization scans for first low-alignment seed span")
{
    StepXThenZNormalSampler sampler(100.0);
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 2;
    config.segmentLength = 100.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 6; ++i) {
        linePoints.push_back({static_cast<double>(i) * 50.0, 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, false, 0},
        {2.0, {100.0, 0.0, 0.0}, false, 2},
        {4.0, {200.0, 0.0, 0.0}, false, 4},
        {6.0, {300.0, 0.0, 0.0}, false, 6},
    };

    const auto result = optimizer.reinitializeAndOptimizeExistingLine(linePoints,
                                                                      controls,
                                                                      {0, 2, 4, 6},
                                                                      0,
                                                                      config);

    CHECK_FALSE(result.failed);
    REQUIRE(result.spans.size() == 3);
    CHECK(result.initialSeedSpanIndex == 1);
    CHECK(result.spans[0].chosenMaxNormalAlignmentAbs > 0.9);
    CHECK(result.spans[1].chosenMaxNormalAlignmentAbs < 1.0e-9);
}

TEST_CASE("LineOptimizer seedless reinitialization fails when no seed span passes alignment gate")
{
    ConstantNormalSampler sampler({1.0, 0.0, 0.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 2;
    config.segmentLength = 100.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 4; ++i) {
        linePoints.push_back({static_cast<double>(i) * 50.0, 0.0, 0.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, false, 0},
        {2.0, {100.0, 0.0, 0.0}, false, 2},
        {4.0, {200.0, 0.0, 0.0}, false, 4},
    };

    const auto result = optimizer.reinitializeAndOptimizeExistingLine(linePoints,
                                                                      controls,
                                                                      {0, 2, 4},
                                                                      0,
                                                                      config);

    CHECK(result.failed);
    CHECK(result.initialSeedSpanIndex == -1);
    CHECK(result.failedSegmentIndex == 0);
    CHECK(result.failureReason.find("No reinitialization seed span") != std::string::npos);
    CHECK(result.optimization.report.converged == false);
}

TEST_CASE("LineOptimizer reinit reopt handles degenerate projected rollout tangents")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 100.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i <= 10; ++i) {
        linePoints.push_back({0.0, 0.0, static_cast<double>(i) * 50.0});
    }

    std::vector<vc::lasagna::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, false, 0},
        {5.0, {0.0, 0.0, 250.0}, true, 5},
        {10.0, {0.0, 0.0, 500.0}, false, 10},
    };

    const auto result = optimizer.reinitializeAndOptimizeExistingLine(linePoints,
                                                                      controls,
                                                                      {0, 5, 10},
                                                                      5,
                                                                      config);

    REQUIRE(result.spans.size() == 2);
    for (const auto& span : result.spans) {
        CHECK(span.candLeftSelectedSign == 1);
        CHECK(span.candRightSelectedSign == 1);
        CHECK(std::isfinite(span.candLeftClosestTargetDistance));
        CHECK(std::isfinite(span.candRightClosestTargetDistance));
    }
    REQUIRE(result.fixedPointIndices.size() == 3);
    for (size_t controlIndex = 0; controlIndex < controls.size(); ++controlIndex) {
        const int fixedIndex = result.fixedPointIndices[controlIndex];
        REQUIRE(fixedIndex >= 0);
        REQUIRE(fixedIndex < static_cast<int>(result.optimization.line.points.size()));
        CHECK(norm(result.optimization.line.points[static_cast<size_t>(fixedIndex)].position -
                   controls[controlIndex].volumePoint) ==
              doctest::Approx(0.0).epsilon(1.0e-12));
    }
    for (const auto& point : result.optimization.line.points) {
        CHECK(std::isfinite(point.position[0]));
        CHECK(std::isfinite(point.position[1]));
        CHECK(std::isfinite(point.position[2]));
    }
}

TEST_CASE("LineOptimizer reproduces las008 seed then added-control edit without local jump")
{
    const fs::path manifestPath =
        "/home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json";
    if (!fs::exists(manifestPath)) {
        MESSAGE("las008 manifest missing; skipping dataset-specific line edit regression");
        return;
    }

    const cv::Vec3d addedControl{17011.396484375, 15631.43359375, 32125.755859375};
    const cv::Vec3d seed{17965.669921875, 15138.587890625, 37891.5};

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 200;
    config.segmentLength = 32.0;
    config.straightnessWeight = 0.1;
    config.tangentStraightnessWeight = 5.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 1000;
    config.differentiableNormalSampling = true;
    const auto seedNormal = sampler.sampleNormal(seed);
    const cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
    const cv::Vec3d normal = seedNormal.valid ? normalizedOrZero(seedNormal.normal) : cv::Vec3d{0.0, 0.0, 0.0};
    config.initialTangent = normalizedOrZero(sourceSliceNormal - normal * sourceSliceNormal.dot(normal));
    config.useInitialTangent = norm(config.initialTangent) > 1.0e-12;
    config.tangentGuideVector = sourceSliceNormal;
    config.tangentGuideWeight = 1.0;
    config.tangentGuideMode =
        vc::lasagna::LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane;
    config.printSolverProgress = false;

    const auto seedResult = optimizer.optimizeFromSeed(seed, config);
    REQUIRE(seedResult.line.points.size() > static_cast<size_t>(config.segmentsPerSide));

    std::vector<cv::Vec3d> initialLinePoints;
    initialLinePoints.reserve(seedResult.line.points.size());
    for (const auto& point : seedResult.line.points) {
        initialLinePoints.push_back(point.position);
    }

    const double addedLinePosition = nearestLinePosition(initialLinePoints, addedControl);
    auto editConfig = config;
    editConfig.initialLinePoints = initialLinePoints;

    std::vector<vc::lasagna::LineControlPoint> controls{
        {addedLinePosition, addedControl, false, -1},
        {static_cast<double>(config.segmentsPerSide),
         initialLinePoints[static_cast<size_t>(config.segmentsPerSide)],
         true,
         config.segmentsPerSide},
    };

    auto initOnlyConfig = editConfig;
    initOnlyConfig.maxIterations = 0;
    const auto initOnly = optimizer.optimizeFromControlPoints(controls, initOnlyConfig);
    const LocalJumpMetrics initMetrics = localJumpMetrics(initOnly.line, addedControl);

    const auto edited = optimizer.optimizeFromControlPoints(std::move(controls), editConfig);
    const LocalJumpMetrics finalMetrics = localJumpMetrics(edited.line, addedControl);

    REQUIRE(initMetrics.controlIndex >= 3);
    REQUIRE(initMetrics.controlIndex + 3 < static_cast<int>(initOnly.line.points.size()));
    REQUIRE(finalMetrics.controlIndex >= 3);
    REQUIRE(finalMetrics.controlIndex + 3 < static_cast<int>(edited.line.points.size()));
    CHECK(initMetrics.nearestControlDistance < 1.0e-4);
    CHECK(finalMetrics.nearestControlDistance < 1.0e-4);

    MESSAGE("added_line_position=" << addedLinePosition
            << " init_control_index=" << initMetrics.controlIndex
            << " init_nearest_control_distance=" << initMetrics.nearestControlDistance
            << " init_max_local_length_ratio=" << initMetrics.maxLengthRatio
            << " init_max_local_turn_deg=" << initMetrics.maxTurnDegrees
            << " final_control_index=" << finalMetrics.controlIndex
            << " final_nearest_control_distance=" << finalMetrics.nearestControlDistance
            << " final_max_local_length_ratio=" << finalMetrics.maxLengthRatio
            << " final_max_local_turn_deg=" << finalMetrics.maxTurnDegrees);
    CHECK(initMetrics.maxLengthRatio < 2.0);
    CHECK(initMetrics.maxTurnDegrees < 60.0);
    CHECK(finalMetrics.maxLengthRatio < 2.0);
    CHECK(finalMetrics.maxTurnDegrees < 60.0);
}

TEST_CASE("LineOptimizer detects jump in saved las008 fiber 5 and reoptimizes it")
{
    const fs::path fiberPath = "/home/hendrik/business/aiconsulting/vesuviuschallenge/villa/5.json";
    const fs::path manifestPath =
        "/home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json";
    if (!fs::exists(fiberPath) || !fs::exists(manifestPath)) {
        MESSAGE("5.json or las008 manifest missing; skipping saved-fiber reoptimization regression");
        return;
    }

    std::ifstream input(fiberPath);
    REQUIRE(input.good());
    const nlohmann::json root = nlohmann::json::parse(input);
    const std::vector<cv::Vec3d> controlPoints = readVec3Array(root, "control_points");
    const std::vector<cv::Vec3d> linePoints = readVec3Array(root, "line_points");
    REQUIRE(controlPoints.size() == 2);
    REQUIRE(linePoints.size() > 10);

    const vc::lasagna::LineModel savedLine = lineModelFromPoints(linePoints);
    const LocalJumpMetrics savedAddedMetrics = localJumpMetrics(savedLine, controlPoints[0]);
    const LocalJumpMetrics savedSeedMetrics = localJumpMetrics(savedLine, controlPoints[1]);
    MESSAGE("saved_added_control_index=" << savedAddedMetrics.controlIndex
            << " saved_added_dist=" << savedAddedMetrics.nearestControlDistance
            << " saved_added_length_ratio=" << savedAddedMetrics.maxLengthRatio
            << " saved_added_turn_deg=" << savedAddedMetrics.maxTurnDegrees
            << " saved_seed_control_index=" << savedSeedMetrics.controlIndex
            << " saved_seed_dist=" << savedSeedMetrics.nearestControlDistance
            << " saved_seed_length_ratio=" << savedSeedMetrics.maxLengthRatio
            << " saved_seed_turn_deg=" << savedSeedMetrics.maxTurnDegrees);
    CHECK(savedAddedMetrics.nearestControlDistance < 1.0e-4);
    CHECK(savedAddedMetrics.maxLengthRatio > 1.8);
    CHECK(savedAddedMetrics.maxTurnDegrees > 60.0);

    vc::lasagna::LasagnaDataset dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    vc::lasagna::LasagnaNormalSampler sampler(dataset);
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 200;
    config.segmentLength = 32.0;
    config.straightnessWeight = 0.1;
    config.tangentStraightnessWeight = 5.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 1000;
    config.differentiableNormalSampling = true;
    config.initialLinePoints = linePoints;
    config.printSolverProgress = false;

    const auto seedNormal = sampler.sampleNormal(controlPoints[1]);
    const cv::Vec3d sourceSliceNormal{0.0, 0.0, 1.0};
    const cv::Vec3d normal = seedNormal.valid ? normalizedOrZero(seedNormal.normal) : cv::Vec3d{0.0, 0.0, 0.0};
    config.initialTangent = normalizedOrZero(sourceSliceNormal - normal * sourceSliceNormal.dot(normal));
    config.useInitialTangent = norm(config.initialTangent) > 1.0e-12;
    config.tangentGuideVector = sourceSliceNormal;
    config.tangentGuideWeight = 1.0;
    config.tangentGuideMode =
        vc::lasagna::LineOptimizationConfig::TangentGuideMode::ProjectVectorOntoTangentPlane;

    std::vector<vc::lasagna::LineControlPoint> controls;
    controls.reserve(controlPoints.size());
    for (size_t controlIndex = 0; controlIndex < controlPoints.size(); ++controlIndex) {
        int nearestIndex = 0;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (size_t lineIndex = 0; lineIndex < linePoints.size(); ++lineIndex) {
            const double distance = norm(linePoints[lineIndex] - controlPoints[controlIndex]);
            if (distance < bestDistance) {
                bestDistance = distance;
                nearestIndex = static_cast<int>(lineIndex);
            }
        }
        controls.push_back({static_cast<double>(nearestIndex),
                            controlPoints[controlIndex],
                            controlIndex == 1,
                            nearestIndex});
    }

    const auto reoptimized = optimizer.optimizeFromControlPoints(std::move(controls), config);
    const LocalJumpMetrics reoptAddedMetrics = localJumpMetrics(reoptimized.line, controlPoints[0]);
    const LocalJumpMetrics reoptSeedMetrics = localJumpMetrics(reoptimized.line, controlPoints[1]);
    MESSAGE("reopt_added_control_index=" << reoptAddedMetrics.controlIndex
            << " reopt_added_dist=" << reoptAddedMetrics.nearestControlDistance
            << " reopt_added_length_ratio=" << reoptAddedMetrics.maxLengthRatio
            << " reopt_added_turn_deg=" << reoptAddedMetrics.maxTurnDegrees
            << " reopt_seed_control_index=" << reoptSeedMetrics.controlIndex
            << " reopt_seed_dist=" << reoptSeedMetrics.nearestControlDistance
            << " reopt_seed_length_ratio=" << reoptSeedMetrics.maxLengthRatio
            << " reopt_seed_turn_deg=" << reoptSeedMetrics.maxTurnDegrees);

    CHECK(reoptAddedMetrics.nearestControlDistance < 1.0e-4);
    CHECK(reoptAddedMetrics.maxLengthRatio < savedAddedMetrics.maxLengthRatio);
    CHECK(reoptAddedMetrics.maxTurnDegrees < savedAddedMetrics.maxTurnDegrees);
}

TEST_CASE("LineOptimizer sanitizes degenerate config values")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 0;
    config.segmentLength = -10.0;
    config.samplesPerSegment = 0;
    config.straightnessWeight = -1.0;
    config.normalAlignmentWeight = -1.0;
    config.distanceWeight = -1.0;
    config.maxIterations = -5;

    const auto result = optimizer.optimizeFromSeed({1.0, 2.0, 3.0}, config);

    CHECK(result.line.points.size() == 3);
    CHECK(result.line.segmentSamples.size() == 2);
    CHECK(result.line.segmentSamples.front().samples.size() == 2);
    CHECK(result.report.validNormalSamples == 4);
    CHECK(result.report.invalidNormalSamples == 0);
}

TEST_CASE("LineOptimizer handles invalid and zero seed normals with deterministic tangent")
{
    MissingNormalSampler missingSampler;
    vc::lasagna::LineOptimizer missingOptimizer(missingSampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 7.0;
    config.maxIterations = 0;

    const auto missingResult = missingOptimizer.optimizeFromSeed({10.0, 20.0, 30.0}, config);
    REQUIRE(missingResult.line.points.size() == 3);
    CHECK(missingResult.line.points.front().position[0] == doctest::Approx(3.0));
    CHECK(missingResult.line.points.back().position[0] == doctest::Approx(17.0));
    CHECK(missingResult.report.invalidNormalSamples == 2 * 2);

    ConstantNormalSampler zeroSampler({0.0, 0.0, 0.0});
    vc::lasagna::LineOptimizer zeroOptimizer(zeroSampler);
    const auto zeroResult = zeroOptimizer.optimizeFromSeed({10.0, 20.0, 30.0}, config);
    REQUIRE(zeroResult.line.points.size() == 3);
    CHECK(zeroResult.line.points.front().position[0] == doctest::Approx(3.0));
    CHECK(zeroResult.line.points.back().position[0] == doctest::Approx(17.0));
    CHECK(zeroResult.report.invalidNormalSamples == 2 * 2);
}

TEST_CASE("LineOptimizer accepts an initial tangent constrained to the sampled tangent plane")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 10.0;
    config.maxIterations = 0;
    config.useInitialTangent = true;
    config.initialTangent = {0.0, 1.0, 5.0};

    const auto result = optimizer.optimizeFromSeed({10.0, 20.0, 30.0}, config);

    REQUIRE(result.line.points.size() == 3);
    CHECK(result.line.points.front().position[0] == doctest::Approx(10.0));
    CHECK(result.line.points.front().position[1] == doctest::Approx(10.0));
    CHECK(result.line.points.front().position[2] == doctest::Approx(30.0));
    CHECK(result.line.points.back().position[0] == doctest::Approx(10.0));
    CHECK(result.line.points.back().position[1] == doctest::Approx(30.0));
    CHECK(result.line.points.back().position[2] == doctest::Approx(30.0));
}

TEST_CASE("LineOptimizer seed initialization transports tangent by normal rotation")
{
    SeedThenTiltedNormalSampler sampler;
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 1.0;
    config.samplesPerSegment = 1;
    config.runGlobalOptimization = false;
    config.useInitialTangent = true;
    config.initialTangent = {1.0, 1.0, 0.0};

    const auto result = optimizer.optimizeFromSeed({0.0, 0.0, 0.0}, config);

    REQUIRE(result.line.points.size() == 3);
    CHECK(result.report.iterations == 0);
    const cv::Vec3d forward = result.line.points[2].position - result.line.points[1].position;
    const cv::Vec3d direction = forward * (1.0 / norm(forward));
    CHECK(direction[0] == doctest::Approx(0.5).epsilon(1.0e-9));
    CHECK(direction[1] == doctest::Approx(1.0 / std::sqrt(2.0)).epsilon(1.0e-9));
    CHECK(direction[2] == doctest::Approx(-0.5).epsilon(1.0e-9));

    const double invSqrt2 = 1.0 / std::sqrt(2.0);
    CHECK(direction.dot(cv::Vec3d{invSqrt2, 0.0, invSqrt2}) ==
          doctest::Approx(0.0).epsilon(1.0e-9));
}

TEST_CASE("LineOptimizer resamples normals during solver residual evaluation")
{
    CountingNormalSampler sampler;
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 10.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 1;

    const auto result = optimizer.optimizeFromSeed({10.0, 20.0, 30.0}, config);

    REQUIRE(result.line.points.size() == 3);
    CHECK(sampler.calls > 8);
}

TEST_CASE("LineOptimizer split straightness reports match equal-weight 3D straightness")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 10.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.straightnessWeight = 2.0;
    config.tangentStraightnessWeight = -1.0;
    config.normalStraightnessWeight = -1.0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    std::vector<vc::lasagna::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, true, -1},
        {1.0, {10.0, 0.0, 3.0}, false, -1},
        {2.0, {20.0, 0.0, 0.0}, false, -1},
    };
    const auto result = optimizer.optimizeFromControlPoints(std::move(controls), config);

    const auto& tangent = lossByName(result.report, "tangent_straightness");
    const auto& normal = lossByName(result.report, "normal_straightness");
    CHECK(tangent.weight == doctest::Approx(2.0));
    CHECK(normal.weight == doctest::Approx(2.0));
    CHECK(tangent.residuals == normal.residuals * 3);

    double expectedWeightedCost = 0.0;
    for (size_t i = 1; i + 1 < result.line.points.size(); ++i) {
        const cv::Vec3d curvature = result.line.points[i - 1].position -
                                    2.0 * result.line.points[i].position +
                                    result.line.points[i + 1].position;
        expectedWeightedCost += 0.5 * 2.0 * 2.0 * curvature.dot(curvature);
    }
    CHECK(tangent.weightedCost + normal.weightedCost ==
          doctest::Approx(expectedWeightedCost).epsilon(1.0e-10));
}

TEST_CASE("LineOptimizer split straightness samples the normal sampler at stencil centers")
{
    CountingNormalSampler sampler;
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 10.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 1;
    config.straightnessWeight = 0.0;
    config.tangentStraightnessWeight = 1.0;
    config.normalStraightnessWeight = 1.0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    std::vector<vc::lasagna::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, true, -1},
        {1.0, {10.0, 0.0, 3.0}, false, -1},
        {2.0, {20.0, 0.0, 0.0}, false, -1},
    };
    const auto result = optimizer.optimizeFromControlPoints(std::move(controls), config);

    REQUIRE(lossByName(result.report, "tangent_straightness").residuals > 0);
    REQUIRE(lossByName(result.report, "normal_straightness").residuals > 0);

    int centerSamples = 0;
    for (const auto& sampled : sampler.sampledPoints) {
        for (size_t i = 1; i + 1 < result.line.points.size(); ++i) {
            if (norm(sampled - result.line.points[i].position) < 1.0e-9) {
                ++centerSamples;
                break;
            }
        }
    }
    CHECK(centerSamples > static_cast<int>(result.line.points.size() - 2));
}

TEST_CASE("LineOptimizer straightness reuses prefetched point normals")
{
    BatchOnlyZNormalSampler sampler;
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentLength = 10.0;
    config.samplesPerSegment = 1;
    config.maxIterations = 0;
    config.straightnessWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 1.0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 3.0},
        {20.0, 0.0, 0.0},
    };
    const auto result = optimizer.optimizeExistingLine(linePoints,
                                                       {0, 2},
                                                       1,
                                                       config,
                                                       -1,
                                                       -1,
                                                       "prefetched-straightness");

    CHECK(sampler.batchCalls > 0);
    CHECK(result.report.initialCost == doctest::Approx(18.0));
}

TEST_CASE("LineOptimizer supports zero iterations and single-seed optimizeFromSeeds")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    vc::lasagna::LineOptimizationConfig config;
    config.segmentsPerSide = 1;
    config.segmentLength = 25.0;
    config.maxIterations = 0;

    const auto result = optimizer.optimizeFromSeeds({{100.0, 200.0, 300.0}}, config);

    REQUIRE(result.line.points.size() == 3);
    CHECK(result.line.points[1].position[0] == doctest::Approx(100.0));
    CHECK(result.line.points[1].position[1] == doctest::Approx(200.0));
    CHECK(result.line.points[1].position[2] == doctest::Approx(300.0));
    CHECK(result.report.validNormalSamples == 2 * 2);
    CHECK(result.report.invalidNormalSamples == 0);
}
