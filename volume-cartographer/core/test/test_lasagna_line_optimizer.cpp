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

TEST_CASE("LineOptimizer V1 rejects multi-seed requests explicitly")
{
    ConstantNormalSampler sampler({0.0, 0.0, 1.0});
    vc::lasagna::LineOptimizer optimizer(sampler);

    std::vector<cv::Vec3d> seeds{{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}};
    CHECK_THROWS_AS(optimizer.optimizeFromSeeds(seeds), std::invalid_argument);
    CHECK_THROWS_AS(optimizer.optimizeFromSeeds({}), std::invalid_argument);
}
