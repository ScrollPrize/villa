#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/LineOptimizer.hpp"

#include <cmath>
#include <stdexcept>

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
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& /*volumePoint*/) const override
    {
        ++calls;
        return {{0.0, 0.0, 1.0}, true, {}};
    }

    mutable int calls = 0;
};

double norm(const cv::Vec3d& vector)
{
    return std::sqrt(vector.dot(vector));
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
    REQUIRE(result.line.segmentSamples.size() == 20);
    CHECK(result.report.converged);
    CHECK(result.report.validNormalSamples == 20 * 5);
    CHECK(result.report.invalidNormalSamples == 0);
    REQUIRE(result.report.finalLosses.size() == 6);
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
    CHECK(result.line.points[1].position[0] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.line.points[1].position[1] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.line.points[1].position[2] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.line.points[4].position[0] == doctest::Approx(30.0).epsilon(1.0e-9));
    CHECK(result.line.points[4].position[1] == doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.line.points[4].position[2] == doctest::Approx(0.0).epsilon(1.0e-9));

    auto lossByName = [&](const std::string& name) -> const vc::lasagna::LineOptimizationLossReport& {
        for (const auto& loss : result.report.finalLosses) {
            if (loss.name == name) {
                return loss;
            }
        }
        throw std::runtime_error("missing loss " + name);
    };
    CHECK(lossByName("step_distance").residuals == 2);
    CHECK(lossByName("even_step").residuals == 2);
    CHECK(lossByName("initial_direction").residuals == 0);
    CHECK(lossByName("tangent_guide").residuals == 0);

    const auto fromSeeds = optimizer.optimizeFromSeeds({{0.0, 0.0, 0.0}, {30.0, 0.0, 0.0}}, config);
    REQUIRE(fromSeeds.line.points.size() == result.line.points.size());
    CHECK_THROWS_AS(optimizer.optimizeFromSeeds({}), std::invalid_argument);
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
    CHECK(missingResult.report.invalidNormalSamples == 2 * 5);

    ConstantNormalSampler zeroSampler({0.0, 0.0, 0.0});
    vc::lasagna::LineOptimizer zeroOptimizer(zeroSampler);
    const auto zeroResult = zeroOptimizer.optimizeFromSeed({10.0, 20.0, 30.0}, config);
    REQUIRE(zeroResult.line.points.size() == 3);
    CHECK(zeroResult.line.points.front().position[0] == doctest::Approx(3.0));
    CHECK(zeroResult.line.points.back().position[0] == doctest::Approx(17.0));
    CHECK(zeroResult.report.invalidNormalSamples == 2 * 5);
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
    CHECK(result.report.validNormalSamples == 2 * 5);
    CHECK(result.report.invalidNormalSamples == 0);
}
