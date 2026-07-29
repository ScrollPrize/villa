#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/ChannelSampler.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace {

class StraightPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d&,
        const cv::Vec3d& referenceDirection) const override
    {
        vc::fiber_tracer::FiberPredictionSample out;
        out.options.push_back({
            referenceDirection[0] < 0.0
                ? cv::Vec3d{-1.0, 0.0, 0.0}
                : cv::Vec3d{1.0, 0.0, 0.0},
            1.0,
            true,
        });
        return out;
    }
};

class CountingPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    mutable int sampleCalls = 0;

    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d&,
        const cv::Vec3d& referenceDirection) const override
    {
        ++sampleCalls;
        vc::fiber_tracer::FiberPredictionSample out;
        out.options.push_back({referenceDirection, 1.0, true});
        return out;
    }
};

class InvalidStartPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d& point,
        const cv::Vec3d& referenceDirection) const override
    {
        vc::fiber_tracer::FiberPredictionSample out;
        if (point[0] < 1.0e-6 && std::abs(point[1]) < 1.0e-6) {
            out.options.push_back({});
            return out;
        }
        out.options.push_back({referenceDirection, 1.0, true});
        return out;
    }
};

class StartAndCurrentBranchPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d& point,
        const cv::Vec3d& referenceDirection) const override
    {
        vc::fiber_tracer::FiberPredictionSample out;
        if (point[0] < 1.0e-6 && std::abs(point[1]) < 1.0e-6) {
            out.options.push_back({{1.0, 0.0, 0.0}, 0.01, true});
            out.options.push_back({{0.0, 1.0, 0.0}, 1.0, true});
            return out;
        }
        if (std::abs(point[0] - 4.0) < 0.25 && std::abs(point[1]) < 0.25) {
            out.options.push_back({{1.0, 0.0, 0.0}, 0.05, true});
            out.options.push_back({{1.0, 1.0, 0.0}, 1.0, true});
            return out;
        }
        out.options.push_back({referenceDirection, 1.0, true});
        return out;
    }
};

class PruningPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d& point,
        const cv::Vec3d& referenceDirection) const override
    {
        vc::fiber_tracer::FiberPredictionSample out;
        double presence = 1.0;
        if (point[0] > 1.0e-6 && point[0] < 4.5) {
            if (std::abs(point[1]) < 0.1 && std::abs(point[2]) < 0.1)
                presence = 1.0;
            else if (point[1] > 0.0 && point[1] < 0.6 && std::abs(point[2]) < 0.6)
                presence = 0.99;
            else if (point[1] > 1.5 && std::abs(point[2]) < 0.6)
                presence = 0.98;
            else
                presence = 0.1;
        } else if (point[0] >= 4.5) {
            presence = point[1] > 1.0 ? 1.0 : 0.0;
        }
        out.options.push_back({referenceDirection, presence, true});
        return out;
    }
};

class DirectionProductPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d&,
        const cv::Vec3d& referenceDirection) const override
    {
        const cv::Vec3d ref = referenceDirection / cv::norm(referenceDirection);
        const double axialPresence =
            std::abs(ref.dot(cv::Vec3d{1.0, 0.0, 0.0})) > 0.999999
                ? 0.80
                : 1.0;
        vc::fiber_tracer::FiberPredictionSample out;
        out.options.push_back({ref, axialPresence, true});
        return out;
    }
};

class ConstantNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit ConstantNormalSampler(cv::Vec3d normal = {0.0, 1.0, 0.0})
        : normal_(normal)
    {
    }

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override
    {
        return {normal_, true, {}};
    }

private:
    cv::Vec3d normal_;
};

vc::lasagna::LasagnaChannelGroup makeGroup(
    std::string name,
    int scaledown,
    std::vector<std::string> channels)
{
    vc::lasagna::LasagnaChannelGroup group;
    group.name = std::move(name);
    group.scaledown = scaledown;
    group.channels = std::move(channels);
    return group;
}

cv::Matx33d outer(const cv::Vec3d& v)
{
    cv::Matx33d out = cv::Matx33d::zeros();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col)
            out(row, col) = v[row] * v[col];
    }
    return out;
}

} // namespace

TEST_CASE("native fiber tracer defaults match regular Trace2CP command")
{
    const vc::fiber_tracer::FiberTraceConfig config;

    CHECK(config.stepVoxels == doctest::Approx(4.0));
    CHECK(config.coneGridSize == 25);
    CHECK(config.beamWidth == 8);
    CHECK(config.beamPruneDistanceVoxels == doctest::Approx(1.0));
    CHECK(config.beamLookaheadSteps == 2);
    CHECK(config.smoothnessNormalWeight == doctest::Approx(0.1));
    CHECK(config.smoothnessTangentWeight == doctest::Approx(10.0));
    CHECK(config.smoothnessFreeAngleDegrees == doctest::Approx(0.0));
    CHECK(config.cumulativeSmoothnessSteps == 4);
    CHECK(config.cumulativeSmoothnessTangentWeight == doctest::Approx(2.0));
}

TEST_CASE("fiber prediction trace scales derive from existing manifest fields")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 4.0;
    manifest.groups.push_back(makeGroup("fiber", 2, {"presence", "nx", "ny"}));

    const auto scales =
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest);

    CHECK(scales.traceToBaseScale == doctest::Approx(4.0));
    CHECK(scales.predictionToBaseScale == doctest::Approx(16.0));
    CHECK(scales.predictionSpacingInTraceVoxels == doctest::Approx(4.0));
    CHECK(vc::fiber_tracer::inferFiberPredictionWorkingToBaseScale(manifest) ==
          doctest::Approx(4.0));
}

TEST_CASE("fiber prediction trace scales use inference scaledown power")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 1.0;
    manifest.groups.push_back(makeGroup("fiber", 4, {"presence", "nx", "ny"}));

    const auto scales =
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest);

    CHECK(scales.traceToBaseScale == doctest::Approx(4.0));
    CHECK(scales.predictionToBaseScale == doctest::Approx(16.0));
    CHECK(scales.predictionSpacingInTraceVoxels == doctest::Approx(4.0));

    const auto unscaledTrace =
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest, 0);
    CHECK(unscaledTrace.traceToBaseScale == doctest::Approx(16.0));
    CHECK(unscaledTrace.predictionToBaseScale == doctest::Approx(16.0));
    CHECK(unscaledTrace.predictionSpacingInTraceVoxels == doctest::Approx(1.0));
}

TEST_CASE("fiber prediction trace scales support multi-output manifest")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 4.0;
    manifest.groups.push_back(makeGroup(
        "fiber_option_000",
        2,
        {"option_000_presence", "option_000_nx", "option_000_ny"}));
    manifest.groups.push_back(makeGroup(
        "fiber_option_001",
        2,
        {"option_001_presence", "option_001_nx", "option_001_ny"}));

    const auto scales =
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest);

    CHECK(scales.traceToBaseScale == doctest::Approx(4.0));
    CHECK(scales.predictionToBaseScale == doctest::Approx(16.0));
    CHECK(scales.predictionSpacingInTraceVoxels == doctest::Approx(4.0));
}

TEST_CASE("fiber prediction trace scales reject missing prediction channels")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 1.0;
    manifest.groups.push_back(makeGroup("fiber", 2, {"presence", "nx"}));

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest),
        doctest::Contains("presence/nx/ny"),
        std::runtime_error);
}

TEST_CASE("fiber prediction trace scales reject mixed prediction channel scales")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 1.0;
    manifest.groups.push_back(makeGroup("presence", 2, {"presence"}));
    manifest.groups.push_back(makeGroup("directions", 3, {"nx", "ny"}));

    CHECK_THROWS_AS(
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest),
        std::runtime_error);
}

TEST_CASE("fiber prediction trace scales reject invalid source_to_base")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 0.0;
    manifest.groups.push_back(makeGroup("fiber", 2, {"presence", "nx", "ny"}));

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest),
        doctest::Contains("source_to_base"),
        std::runtime_error);
}

TEST_CASE("fiber prediction trace scales require manifest source_to_base field")
{
    const auto manifest = vc::lasagna::LasagnaDatasetManifest::parseText(R"({
        "version": 2,
        "groups": {
            "fiber": {
                "zarr": "fiber.zarr",
                "scaledown": 2,
                "channels": ["presence", "nx", "ny"]
            }
        }
    })");

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest),
        doctest::Contains("source_to_base"),
        std::runtime_error);
}

TEST_CASE("fiber prediction trace scales reject invalid inference scaledown power")
{
    vc::lasagna::LasagnaDatasetManifest manifest;
    manifest.sourceToBase = 1.0;
    manifest.groups.push_back(makeGroup("fiber", 4, {"presence", "nx", "ny"}));

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest, -1),
        doctest::Contains("scaledown power"),
        std::runtime_error);
    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::resolveFiberPredictionTraceScales(manifest, 31),
        doctest::Contains("scaledown power"),
        std::runtime_error);
}

TEST_CASE("native fiber tracer requires normals for normal-aware smoothness")
{
    StraightPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {16.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {1.0, 0.0, 0.0};
    request.budgetSpanVoxels = 16.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr),
        doctest::Contains("Lasagna normal sampler"),
        std::invalid_argument);

    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;
    CHECK_NOTHROW(
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr));
}

TEST_CASE("native fiber tracer prunes beams by Python score and generation order")
{
    using vc::fiber_tracer::testing::BeamDebugState;
    const std::vector<BeamDebugState> states = {
        {.loss = 1.0, .depth = 2, .tracedLength = 50.0, .point = {0.0, 0.0, 0.0}},
        {.loss = 1.0, .depth = 2, .tracedLength = 1.0, .point = {10.0, 0.0, 0.0}},
        {.loss = 1.0, .depth = 1, .tracedLength = 100.0, .point = {20.0, 0.0, 0.0}},
        {.loss = 1.0, .depth = 2, .tracedLength = 0.0, .point = {30.0, 0.0, 0.0}},
    };

    const auto kept =
        vc::fiber_tracer::testing::debugPruneBeamStateIndices(states, 3, 0.0);

    REQUIRE(kept.size() == 3);
    CHECK(kept[0] == 2);
    CHECK(kept[1] == 0);
    CHECK(kept[2] == 1);
}

TEST_CASE("native fiber tracer spatial beam pruning preserves Python tensor order ties")
{
    using vc::fiber_tracer::testing::BeamDebugState;
    const std::vector<BeamDebugState> states = {
        {.loss = 0.5, .depth = 1, .tracedLength = 100.0, .point = {0.0, 0.0, 0.0}},
        {.loss = 0.5, .depth = 1, .tracedLength = 0.0, .point = {0.5, 0.0, 0.0}},
        {.loss = 0.5, .depth = 1, .tracedLength = 50.0, .point = {2.0, 0.0, 0.0}},
    };

    const auto kept =
        vc::fiber_tracer::testing::debugPruneBeamStateIndices(states, 2, 1.0);

    REQUIRE(kept.size() == 2);
    CHECK(kept[0] == 0);
    CHECK(kept[1] == 2);
}

TEST_CASE("native fiber tracer reached-state selection uses first minimum loss only")
{
    using vc::fiber_tracer::testing::BeamDebugState;
    const std::vector<BeamDebugState> states = {
        {.loss = 1.0, .depth = 3, .tracedLength = 100.0, .point = {0.0, 0.0, 0.0}, .reached = true},
        {.loss = 1.0, .depth = 3, .tracedLength = 0.0, .point = {1.0, 0.0, 0.0}, .reached = true},
        {.loss = 0.0, .depth = 3, .tracedLength = 0.0, .point = {2.0, 0.0, 0.0}, .reached = false},
    };

    const auto best =
        vc::fiber_tracer::testing::debugBestReachedBeamStateIndex(states);

    REQUIRE(best.has_value());
    CHECK(*best == 0);
}

TEST_CASE("compact normal principal axis uses the symmetric eigensolver")
{
    const cv::Vec3d principal =
        cv::Vec3d{1.0, 1.0, 0.0} / std::sqrt(2.0);
    const cv::Vec3d secondary =
        cv::Vec3d{1.0, -1.0, 0.0} / std::sqrt(2.0);
    const cv::Vec3d z{0.0, 0.0, 1.0};
    const cv::Matx33d tensor =
        outer(principal) * 1.01 + outer(secondary) * 1.0 + outer(z) * 0.2;

    const cv::Vec3d axis =
        vc::lasagna::principalCompactTensorAxis(tensor, {0.0, 0.0, 0.0});
    const cv::Vec3d flipped =
        vc::lasagna::principalCompactTensorAxis(tensor, -principal);

    CHECK(std::abs(axis.dot(principal)) > 0.999999);
    CHECK(flipped.dot(-principal) > 0.999999);
}

TEST_CASE("native fiber tracer default angle-step cone uses circular 81-candidate disk")
{
    CountingPrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {2.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {1.0, 0.0, 0.0};
    request.budgetSpanVoxels = 2.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, &normals);

    CHECK(result.reachedTargetPlane);
    CHECK(predictions.sampleCalls == 82);
}

TEST_CASE("native fiber tracer rejects invalid start predictions")
{
    InvalidStartPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {8.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {1.0, 0.0, 0.0};
    request.budgetSpanVoxels = 8.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    CHECK_THROWS_WITH_AS(
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr),
        doctest::Contains("start point"),
        std::invalid_argument);
}

TEST_CASE("native fiber tracer interpolates the target-plane crossing point")
{
    StraightPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {6.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {1.0, 0.0, 0.0};
    request.budgetSpanVoxels = 6.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    REQUIRE(result.reachedTargetPlane);
    REQUIRE(!result.points.empty());
    CHECK(result.points.back()[0] == doctest::Approx(6.0));
    CHECK(result.points.back()[1] == doctest::Approx(0.0));
    CHECK(result.points.back()[2] == doctest::Approx(0.0));
}

TEST_CASE("native fiber tracer max-step factor is not clamped before step-limit calculation")
{
    StraightPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {100.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {1.0, 0.0, 0.0};
    request.budgetSpanVoxels = 100.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 0.1;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    CHECK(!result.reachedTargetPlane);
    CHECK(result.reason == "max_step_factor");
    CHECK(result.steps == 3);
}

TEST_CASE("native fiber tracer ignores presence only for the start branch")
{
    StartAndCurrentBranchPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {8.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {1.0, 0.0, 0.0};
    request.budgetSpanVoxels = 8.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 4.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    REQUIRE(result.reachedTargetPlane);
    REQUIRE(result.points.size() >= 3);
    CHECK(result.points[1][0] == doctest::Approx(4.0));
    CHECK(std::abs(result.points[1][1]) < 1.0e-6);
    CHECK(std::abs(result.points.back()[1]) > 1.0);
}

TEST_CASE("native fiber tracer candidate loss uses the all-pairs direction product")
{
    DirectionProductPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {3.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {1.0, 0.0, 0.0};
    request.budgetSpanVoxels = 3.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 25.0;
    request.config.coneAngleStepDegrees = 25.0;
    request.config.beamWidth = 4;
    request.config.beamLookaheadSteps = 1;
    request.config.beamPruneDistanceVoxels = 0.0;
    request.config.maxStepFactor = 2.0;
    request.config.smoothnessWeight = 0.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    REQUIRE(result.reachedTargetPlane);
    REQUIRE(!result.points.empty());
    CHECK(result.points.back()[0] == doctest::Approx(3.0));
    CHECK(std::abs(result.points.back()[1]) < 1.0e-6);
    CHECK(std::abs(result.points.back()[2]) < 1.0e-6);
}

TEST_CASE("native fiber tracer spatially prunes near-duplicate beam states")
{
    PruningPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {8.0, 2.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlaneNormal = {0.0, 1.0, 0.0};
    request.budgetSpanVoxels = 8.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 25.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 2;
    request.config.beamLookaheadSteps = 1;
    request.config.beamPruneDistanceVoxels = 1.0;
    request.config.maxStepFactor = 1.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    REQUIRE(result.reachedTargetPlane);
    REQUIRE(!result.points.empty());
    const auto& endpoint = result.points.back();
    CHECK(endpoint[1] == doctest::Approx(2.0));
    CHECK(std::sqrt(endpoint[1] * endpoint[1] + endpoint[2] * endpoint[2]) > 1.0);
}

TEST_CASE("native fiber tracer fuses a straight cp-to-cp segment")
{
    StraightPrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberTraceSegmentRequest request;
    request.referenceLine = {
        {0.0, 0.0, 0.0},
        {64.0, 0.0, 0.0},
    };
    request.startIndex = 0;
    request.targetIndex = 1;
    request.targetPlaneNormal = cv::Vec3d{1.0, 0.0, 0.0};
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 2.0;
    request.config.endpointAcceptThresholdUm = 50.0;
    request.config.voxelSizeUm = 1.0;

    const auto result =
        vc::fiber_tracer::traceFiberSegment(predictions, request, &normals);

    CHECK(result.forward.reachedTargetPlane);
    CHECK(result.reverse.reachedTargetPlane);
    CHECK(result.accepted);
    REQUIRE(result.fusedLine.size() >= 3);
    CHECK(result.fusedLine.front()[0] == doctest::Approx(0.0));
    CHECK(result.fusedLine.back()[0] == doctest::Approx(64.0));
    CHECK(result.maxEndpointErrorVoxels == doctest::Approx(0.0));
}

TEST_CASE("native fiber tracer computes whole-fiber one-way restart metric")
{
    StraightPrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberInput fiber;
    fiber.path = "synthetic_fiber.json";
    fiber.linePointsXyzBase = {
        {0.0, 0.0, 0.0},
        {64.0, 0.0, 0.0},
        {128.0, 0.0, 0.0},
    };
    fiber.controlPointsXyzBase = fiber.linePointsXyzBase;
    fiber.controlPointLineIndices = {0, 1, 2};

    vc::fiber_tracer::FiberTraceWholeFiberMetricRequest request;
    request.fiber = fiber;
    request.workingToBaseScale = 1.0;
    request.errorThresholdVoxels = 10.0;
    request.voxelSizeUm = 2.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 2.0;

    const auto result =
        vc::fiber_tracer::traceWholeFiberMetric(predictions, request, &normals);

    CHECK(result.segmentCount == 2);
    CHECK(result.restartCount == 0);
    CHECK(result.restartsPerKvx == doctest::Approx(0.0));
    REQUIRE(result.referenceLengthMeters.has_value());
    CHECK(*result.referenceLengthMeters == doctest::Approx(0.000256));
    REQUIRE(result.restartsPerMeter.has_value());
    CHECK(*result.restartsPerMeter == doctest::Approx(0.0));
    REQUIRE(result.segments.size() == 2);
    CHECK(result.segments[0].success);
    CHECK(result.segments[1].success);
}
