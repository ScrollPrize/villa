#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/lasagna/ChannelSampler.hpp"

#include <cmath>
#include <limits>
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

class PositiveEdgePrediction final
    : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d& point,
        const cv::Vec3d& referenceDirection) const override
    {
        vc::fiber_tracer::FiberPredictionSample out;
        if (point[0] >= 8.0) {
            out.options.push_back({});
            return out;
        }
        out.options.push_back({
            referenceDirection[0] < 0.0
                ? cv::Vec3f{-1.0f, 0.0f, 0.0f}
                : cv::Vec3f{1.0f, 0.0f, 0.0f},
            1.0f,
            true,
        });
        return out;
    }
};

class SlantedPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d&,
        const cv::Vec3d& referenceDirection) const override
    {
        constexpr double x = 0.9950371902099892;
        constexpr double y = 0.09950371902099892;
        const double sign = referenceDirection[0] < 0.0 ? -1.0 : 1.0;
        vc::fiber_tracer::FiberPredictionSample out;
        out.options.push_back({
            {static_cast<float>(sign * x), static_cast<float>(sign * y), 0.0f},
            1.0f,
            true});
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
        out.options.push_back({referenceDirection, static_cast<float>(presence), true});
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
        out.options.push_back({ref, static_cast<float>(axialPresence), true});
        return out;
    }
};

class RecrossPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d& point,
        const cv::Vec3d&) const override
    {
        const cv::Vec3f direction = point[0] < 6.0
            ? cv::Vec3f{1.0f, 0.5f, 0.0f}
            : cv::Vec3f{1.0f, -0.5f, 0.0f};
        vc::fiber_tracer::FiberPredictionSample out;
        out.options.push_back({direction, 1.0f, true});
        return out;
    }
};

class BatchPrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    mutable int sampleCalls = 0;
    mutable int batchCalls = 0;

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override
    {
        return true;
    }

    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d&,
        const cv::Vec3d& referenceDirection) const override
    {
        ++sampleCalls;
        vc::fiber_tracer::FiberPredictionSample out;
        out.options.push_back({referenceDirection, 1.0, true});
        return out;
    }

    void sampleBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        const std::vector<cv::Vec3d>& referenceDirections,
        int,
        std::vector<vc::fiber_tracer::FiberPredictionSample>& samples) const override
    {
        ++batchCalls;
        samples.clear();
        samples.reserve(volumePoints.size());
        for (size_t index = 0; index < volumePoints.size(); ++index) {
            vc::fiber_tracer::FiberPredictionSample out;
            out.options.push_back({referenceDirections[index], 1.0, true});
            samples.push_back(std::move(out));
        }
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

void setExplicitTargetPlane(
    vc::fiber_tracer::FiberTraceOneWayRequest& request,
    const cv::Vec3d& normal)
{
    request.targetPlanes = {{"explicit", request.targetPoint, normal}};
}

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
    CHECK(config.lookaheadParentCap == 32);
    CHECK(config.lookaheadRetryParentCap == 0);
    CHECK(config.parallelThreads == 0);
    CHECK(config.smoothnessNormalWeight == doctest::Approx(0.1));
    CHECK(config.smoothnessTangentWeight == doctest::Approx(10.0));
    CHECK(config.smoothnessFreeAngleDegrees == doctest::Approx(0.0));
    CHECK(config.cumulativeSmoothnessSteps == 4);
    CHECK(config.cumulativeSmoothnessTangentWeight == doctest::Approx(2.0));
    CHECK(config.meetingAcceptMaxErrorRatio == doctest::Approx(0.1));
    CHECK(config.endpointAcceptThresholdBaseVoxels == doctest::Approx(20.0));
    CHECK(config.traceToBaseScale == doctest::Approx(1.0));
    CHECK_FALSE(config.baseVoxelSizeUm.has_value());
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

TEST_CASE("fiber trace coordinate adapter round trips base points and distances")
{
    const vc::fiber_tracer::FiberTraceCoordinateAdapter coordinates(4.0);
    const std::vector<cv::Vec3d> base{
        {8.0, 12.0, 16.0},
        {64.0, 20.0, 4.0},
    };
    const auto trace = coordinates.baseToTrace(base);
    REQUIRE(trace.size() == 2);
    CHECK(trace[0] == cv::Vec3d(2.0, 3.0, 4.0));
    CHECK(trace[1] == cv::Vec3d(16.0, 5.0, 1.0));
    const auto roundTrip = coordinates.traceToBase(trace);
    CHECK(roundTrip == base);
    const cv::Vec3d exactStart(8.0000000000001, 12.0, 16.0);
    const cv::Vec3d exactTarget(63.9999999999999, 20.0, 4.0);
    const auto segment = coordinates.traceSegmentToBase(
        trace, exactStart, exactTarget);
    CHECK(segment.front() == exactStart);
    CHECK(segment.back() == exactTarget);
    CHECK(coordinates.baseDistanceToTrace(20.0) == doctest::Approx(5.0));
    CHECK(coordinates.traceDistanceToBase(5.0) == doctest::Approx(20.0));
    CHECK_THROWS_AS(
        vc::fiber_tracer::FiberTraceCoordinateAdapter(0.0),
        std::invalid_argument);
    CHECK(std::isinf(coordinates.traceDistanceToBase(
        std::numeric_limits<double>::infinity())));
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
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
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

TEST_CASE("native fiber tracer parallel workers require concurrent samplers")
{
    CHECK(vc::fiber_tracer::testing::debugTraceWorkerCount(
              true, true, true, 4, 64) == 4);
    CHECK(vc::fiber_tracer::testing::debugTraceWorkerCount(
              true, false, true, 4, 64) == 1);
    CHECK(vc::fiber_tracer::testing::debugTraceWorkerCount(
              false, true, true, 4, 64) == 1);
    CHECK(vc::fiber_tracer::testing::debugTraceWorkerCount(
              true, true, false, 4, 64) == 4);
    CHECK(vc::fiber_tracer::testing::debugTraceWorkerCount(
              true, true, true, 1, 64) == 1);
    CHECK(vc::fiber_tracer::testing::debugTraceWorkerCount(
              true, true, true, 4, 1) == 1);
}

TEST_CASE("native fiber tracer scores persisted corners without materialized samples")
{
    vc::lasagna::LasagnaCornerBatch corners;
    constexpr size_t optionCount = 2;
    corners.values.resize(optionCount * 3 + 3);
    for (auto& volume : corners.values)
        volume.resize(1);
    corners.fractionsXYZ = {{0.25f, 0.5f, 0.75f}};
    corners.valid = {1};
    const auto constantCorners = [](uint8_t value) {
        std::array<uint8_t, 8> out{};
        out.fill(value);
        return out;
    };

    // Option zero points along X at half presence; option one points along Y.
    corners.values[0][0] = constantCorners(128);
    corners.values[1][0] = constantCorners(255);
    corners.values[2][0] = constantCorners(128);
    corners.values[3][0] = constantCorners(255);
    corners.values[4][0] = constantCorners(128);
    corners.values[5][0] = constantCorners(255);
    // A valid Z normal keeps straight X motion at zero smoothness cost.
    corners.values[6][0] = constantCorners(255);
    corners.values[7][0] = constantCorners(128);
    corners.values[8][0] = constantCorners(128);

    vc::fiber_tracer::FiberTraceConfig config;
    const auto score = vc::fiber_tracer::testing::debugCandidateLossFromCorners(
        corners,
        optionCount,
        0,
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        config);

    REQUIRE(score.valid);
    CHECK(score.loss == doctest::Approx(1.0 - 128.0 / 255.0));
    CHECK(score.selectedPresence == doctest::Approx(128.0 / 255.0));
    CHECK(score.selectedDirection[0] == doctest::Approx(1.0));
    CHECK(score.selectedDirection[1] == doctest::Approx(0.0));
    CHECK(score.selectedDirection[2] == doctest::Approx(0.0));
}

TEST_CASE("native fiber tracer parallel path samples predictions in a batch")
{
    BatchPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {3.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
    request.budgetSpanVoxels = 3.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 5.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 2;
    request.config.parallelThreads = 2;
    request.config.smoothnessWeight = 0.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    CHECK(result.reachedTargetPlane);
    CHECK(predictions.sampleCalls == 1);
    CHECK(predictions.batchCalls == 1);
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
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
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
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
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
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
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

TEST_CASE("native fiber tracer requires every configured target plane")
{
    SlantedPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {12.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.1, 0.0};
    request.targetPlanes = {
        {"line_next", request.targetPoint, {1.0, 1.0, 0.0}},
        {"line_prev", request.targetPoint, {1.0, 0.0, 0.0}},
        {"inferred_direction", request.targetPoint, {1.0, -1.0, 0.0}},
    };
    request.budgetSpanVoxels = 12.0;
    request.config.stepVoxels = 1.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 3.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    REQUIRE(result.reachedTargetPlane);
    CHECK(result.targetPlaneCrossings.size() == 3);
    CHECK(result.steps >= 14);
    CHECK(result.selectedTargetPlaneName == "line_prev");
    REQUIRE(result.selectedTargetPlaneCrossing.has_value());
    CHECK(result.selectedTargetPlaneErrorVoxels == doctest::Approx(1.2).epsilon(0.1));
}

TEST_CASE("native fiber tracer reports missing target planes")
{
    StraightPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {8.0, 1.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    request.targetPlanes = {
        {"crossed", {8.0, 0.0, 0.0}, {1.0, 0.0, 0.0}},
        {"missing", request.targetPoint, {0.0, 1.0, 0.0}},
    };
    request.budgetSpanVoxels = 8.0;
    request.config.stepVoxels = 2.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 2.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    CHECK_FALSE(result.reachedTargetPlane);
    CHECK(result.reason == "max_step_factor:missing_target_planes=missing");
    REQUIRE(result.targetPlaneCrossings.size() == 1);
    CHECK(result.targetPlaneCrossings.front().name == "crossed");
}

TEST_CASE("native fiber tracer replaces a crossing with lower target error")
{
    RecrossPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, -2.0, 0.0};
    request.targetPoint = {12.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.5, 0.0};
    request.targetPlanes = {
        {"recross", request.targetPoint, {0.0, 1.0, 0.0}},
    };
    request.targetPlaneAcceptThresholdVoxels = 3.0;
    request.snapTraceToSelectedCrossing = false;
    request.budgetSpanVoxels = 12.0;
    request.config.stepVoxels = 2.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 3.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    REQUIRE(result.reachedTargetPlane);
    REQUIRE(result.selectedTargetPlaneCrossing.has_value());
    CHECK((*result.selectedTargetPlaneCrossing)[0] > 9.0);
    CHECK(result.selectedTargetPlaneErrorVoxels < 3.0);
    REQUIRE(!result.points.empty());
    CHECK(cv::norm(result.points.back() - *result.selectedTargetPlaneCrossing) > 0.1);
}

TEST_CASE("native fiber tracer max-step factor is not clamped before step-limit calculation")
{
    StraightPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {100.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
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
    CHECK(result.reason == "max_step_factor:missing_target_planes=explicit");
    CHECK(result.steps == 3);
}

TEST_CASE("native fiber tracer ignores presence only for the start branch")
{
    StartAndCurrentBranchPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {8.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
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
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
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
    setExplicitTargetPlane(request, {0.0, 1.0, 0.0});
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

TEST_CASE("native fiber tracer exact lookahead bound retains equal-loss parents")
{
    const std::vector<double> parentBounds{0.1, 0.5, 0.5, 0.8};
    CHECK(vc::fiber_tracer::testing::debugExactLookaheadRequiredParentCount(
              parentBounds, 0.5, true) == 3);
    CHECK(vc::fiber_tracer::testing::debugExactLookaheadRequiredParentCount(
              parentBounds, 0.49, true) == 1);
}

TEST_CASE("native fiber tracer exact lookahead bound is conservative without a full result")
{
    const std::vector<double> parentBounds{0.1, 0.5, 0.8};
    CHECK(vc::fiber_tracer::testing::debugExactLookaheadRequiredParentCount(
              parentBounds, std::nullopt, true) == parentBounds.size());
    CHECK(vc::fiber_tracer::testing::debugExactLookaheadRequiredParentCount(
              parentBounds, 0.2, false) == parentBounds.size());
}

TEST_CASE("native fiber tracer capped parent order matches a full deterministic sort")
{
    const std::vector<double> losses{3.0, 1.0, 2.0, 1.0, 0.5, 2.0};
    CHECK(vc::fiber_tracer::testing::debugOrderedIndexPrefix(losses, 3) ==
          std::vector<size_t>{4, 1, 3});
    CHECK(vc::fiber_tracer::testing::debugOrderedIndexPrefix(losses, losses.size()) ==
          std::vector<size_t>{4, 1, 3, 2, 5, 0});
    CHECK(vc::fiber_tracer::testing::debugOrderedIndexPrefix(losses, 0).empty());
}

TEST_CASE("native fiber tracer retries only failed lower capped lazy searches")
{
    using vc::fiber_tracer::testing::debugShouldRetryLookahead;
    CHECK(debugShouldRetryLookahead(true, 28, 32, false));
    CHECK_FALSE(debugShouldRetryLookahead(true, 28, 32, true));
    CHECK_FALSE(debugShouldRetryLookahead(false, 28, 32, false));
    CHECK_FALSE(debugShouldRetryLookahead(true, 0, 32, false));
    CHECK_FALSE(debugShouldRetryLookahead(true, 28, 28, false));
    CHECK_FALSE(debugShouldRetryLookahead(true, 28, 24, false));
    CHECK_FALSE(debugShouldRetryLookahead(true, 28, 0, false));
}

TEST_CASE("native fiber tracer exact lazy lookahead matches exhaustive search")
{
    StraightPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {32.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
    request.budgetSpanVoxels = 32.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 25.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 4;
    request.config.beamLookaheadSteps = 2;
    request.config.beamPruneDistanceVoxels = 1.0;
    request.config.maxStepFactor = 2.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    vc::fiber_tracer::FiberTraceProfile exhaustiveProfile;
    request.config.lazyLookahead = false;
    request.config.profile = &exhaustiveProfile;
    const auto exhaustive =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    vc::fiber_tracer::FiberTraceProfile lazyProfile;
    request.config.lazyLookahead = true;
    request.config.lookaheadParentCap = 0;
    request.config.profile = &lazyProfile;
    const auto lazy =
        vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    CHECK(lazy.reachedTargetPlane == exhaustive.reachedTargetPlane);
    CHECK(lazy.reason == exhaustive.reason);
    CHECK(lazy.steps == exhaustive.steps);
    REQUIRE(lazy.points.size() == exhaustive.points.size());
    for (size_t index = 0; index < lazy.points.size(); ++index) {
        CHECK(cv::norm(lazy.points[index] - exhaustive.points[index]) < 1.0e-6);
    }
    CHECK(lazyProfile.lookaheadEvaluatedParents <=
          lazyProfile.lookaheadTotalParents);
    CHECK(lazyProfile.candidateTasks <= exhaustiveProfile.candidateTasks);
}

TEST_CASE("native fiber tracer lookahead parent cap bounds final expansion")
{
    StraightPrediction predictions;
    vc::fiber_tracer::FiberTraceOneWayRequest request;
    request.startPoint = {0.0, 0.0, 0.0};
    request.targetPoint = {32.0, 0.0, 0.0};
    request.initialDirection = {1.0, 0.0, 0.0};
    setExplicitTargetPlane(request, {1.0, 0.0, 0.0});
    request.budgetSpanVoxels = 32.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 25.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 4;
    request.config.beamLookaheadSteps = 2;
    request.config.lookaheadParentCap = 3;
    request.config.maxStepFactor = 2.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;
    vc::fiber_tracer::FiberTraceProfile profile;
    request.config.profile = &profile;

    (void)vc::fiber_tracer::traceFiberOneWay(predictions, request, nullptr);

    CHECK(profile.lookaheadFinalFrontiers > 0);
    CHECK(profile.lookaheadEvaluatedParents <=
          profile.lookaheadFinalFrontiers * request.config.lookaheadParentCap);
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
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.coneAngleStepDegrees = 5.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 2.0;
    request.config.endpointAcceptThresholdBaseVoxels = 20.0;
    request.config.traceToBaseScale = 4.0;
    request.config.baseVoxelSizeUm = 2.0;

    const auto result =
        vc::fiber_tracer::traceFiberSegment(predictions, request, &normals);

    CHECK(result.forward.reachedTargetPlane);
    CHECK(result.reverse.reachedTargetPlane);
    CHECK(result.accepted);
    REQUIRE(result.forward.targetPlaneCrossings.size() == 2);
    CHECK(result.forward.targetPlaneCrossings[0].name == "line_prev");
    CHECK(result.forward.targetPlaneCrossings[1].name == "inferred_direction");
    REQUIRE(result.reverse.targetPlaneCrossings.size() == 2);
    CHECK(result.reverse.targetPlaneCrossings[0].name == "line_next");
    CHECK(result.reverse.targetPlaneCrossings[1].name == "inferred_direction");
    REQUIRE(result.fusedLine.size() >= 3);
    CHECK(result.fusedLine.front()[0] == doctest::Approx(0.0));
    CHECK(result.fusedLine.back()[0] == doctest::Approx(64.0));
    CHECK(result.maxEndpointErrorTraceVoxels == doctest::Approx(0.0));
    CHECK(result.maxEndpointErrorBaseVoxels == doctest::Approx(0.0));
    CHECK(result.meetingErrorTraceVoxels == doctest::Approx(0.0));
    CHECK(result.meetingErrorBaseVoxels == doctest::Approx(0.0));
    CHECK(result.meetingErrorRatio == doctest::Approx(0.0));
    CHECK_FALSE(result.meetingSource.empty());
    REQUIRE(result.maxEndpointErrorUm.has_value());
    CHECK(*result.maxEndpointErrorUm == doctest::Approx(0.0));

    request.config.baseVoxelSizeUm.reset();
    const auto resultWithoutPhysicalSize =
        vc::fiber_tracer::traceFiberSegment(predictions, request, &normals);
    CHECK(resultWithoutPhysicalSize.accepted);
    CHECK_FALSE(resultWithoutPhysicalSize.maxEndpointErrorUm.has_value());
}

TEST_CASE("native fiber segment uses moving-plane fusion after endpoint misses")
{
    SlantedPrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberTraceSegmentRequest request;
    request.referenceLine = {
        {0.0, 0.0, 0.0},
        {64.0, 0.0, 0.0},
    };
    request.startIndex = 0;
    request.targetIndex = 1;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 2.0;
    request.config.smoothnessWeight = 0.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;
    request.config.traceToBaseScale = 4.0;
    request.config.endpointAcceptThresholdBaseVoxels = 20.0;

    const auto recovered =
        vc::fiber_tracer::traceFiberSegment(predictions, request, &normals);

    CHECK_FALSE(recovered.forward.reachedTargetPlane);
    CHECK_FALSE(recovered.reverse.reachedTargetPlane);
    CHECK(recovered.forward.steps > 16);
    CHECK(recovered.reverse.steps > 16);
    CHECK(recovered.maxEndpointErrorBaseVoxels > 20.0);
    CHECK(recovered.accepted);
    CHECK(recovered.reason == "ok");
    CHECK(recovered.meetingErrorTraceVoxels > 0.0);
    CHECK(recovered.meetingErrorRatio == doctest::Approx(0.1));
    CHECK(recovered.meetingSource.find("moving_plane") != std::string::npos);
}

TEST_CASE("native fiber moving-plane fusion uses traced-length ratio and base scale")
{
    vc::fiber_tracer::FiberTraceConfig config;
    config.stepVoxels = 2.0;
    config.traceToBaseScale = 4.0;
    config.meetingAcceptMaxErrorRatio = 0.1;
    const std::vector<cv::Vec3d> forward{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
    };
    const std::vector<cv::Vec3d> reverse{
        {10.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };

    const auto accepted = vc::fiber_tracer::testing::debugFuseTraceSegment(
        forward, reverse, config);

    REQUIRE(accepted.accepted);
    CHECK(accepted.meetingErrorTraceVoxels == doctest::Approx(1.0));
    CHECK(accepted.meetingErrorBaseVoxels == doctest::Approx(4.0));
    CHECK(accepted.meetingTraceLengthTraceVoxels == doctest::Approx(10.0));
    CHECK(accepted.meetingErrorRatio == doctest::Approx(0.1));
    REQUIRE(accepted.fusedLine.size() >= 3);
    CHECK(accepted.fusedLine.front() == forward.front());
    CHECK(accepted.fusedLine.back() == reverse.front());
    for (size_t index = 1; index < accepted.fusedLine.size(); ++index) {
        CHECK(cv::norm(accepted.fusedLine[index] - accepted.fusedLine[index - 1]) <=
              config.stepVoxels + 1.0e-8);
    }

    config.meetingAcceptMaxErrorRatio = 0.09;
    const auto rejected = vc::fiber_tracer::testing::debugFuseTraceSegment(
        forward, reverse, config);
    CHECK_FALSE(rejected.accepted);
    CHECK(rejected.reason == "meeting_error_ratio");
    CHECK(rejected.meetingErrorBaseVoxels == doctest::Approx(4.0));
    CHECK(rejected.meetingErrorRatio == doctest::Approx(0.1));
}

TEST_CASE("native fiber moving-plane fusion reports no intersection")
{
    vc::fiber_tracer::FiberTraceConfig config;
    config.stepVoxels = 1.0;
    const auto result = vc::fiber_tracer::testing::debugFuseTraceSegment(
        {{0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}},
        {{10.0, 1.0, 0.0}, {8.0, 1.0, 0.0}},
        config);

    CHECK_FALSE(result.accepted);
    CHECK(result.reason == "no_trace_plane_intersection");
    CHECK(result.fusedLine.empty());
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
    request.errorThresholdBaseVoxels = 20.0;
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

TEST_CASE("native whole-fiber tracing keeps the live endpoint past a crossing")
{
    StraightPrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberInput fiber;
    fiber.path = "synthetic_unsnapped_fiber.json";
    fiber.linePointsXyzBase = {
        {0.0, 0.0, 0.0},
        {6.0, 0.0, 0.0},
        {12.0, 0.0, 0.0},
    };
    fiber.controlPointsXyzBase = fiber.linePointsXyzBase;
    fiber.controlPointLineIndices = {0, 1, 2};

    vc::fiber_tracer::FiberTraceWholeFiberMetricRequest request;
    request.fiber = fiber;
    request.errorThresholdBaseVoxels = 20.0;
    request.config.stepVoxels = 4.0;
    request.config.coneAngleDegrees = 0.0;
    request.config.beamWidth = 1;
    request.config.maxStepFactor = 3.0;
    request.config.smoothnessNormalWeight = 0.0;
    request.config.smoothnessTangentWeight = 0.0;
    request.config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result =
        vc::fiber_tracer::traceWholeFiberMetric(predictions, request, &normals);

    REQUIRE(result.segments.size() == 2);
    REQUIRE(result.segments[0].success);
    REQUIRE(result.segments[0].trace.selectedTargetPlaneCrossing.has_value());
    CHECK((*result.segments[0].trace.selectedTargetPlaneCrossing)[0] ==
          doctest::Approx(6.0));
    REQUIRE(!result.segments[0].trace.points.empty());
    CHECK(result.segments[0].trace.points.back()[0] == doctest::Approx(8.0));
}

TEST_CASE("native fiber extrapolation stops on the requested distance plane")
{
    StraightPrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberTraceConfig config;
    config.stepVoxels = 4.0;
    config.coneAngleDegrees = 0.0;
    config.beamWidth = 1;
    config.maxStepFactor = 2.0;
    config.smoothnessNormalWeight = 0.0;
    config.smoothnessTangentWeight = 0.0;
    config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result = vc::fiber_tracer::traceFiberExtrapolation(
        predictions,
        {3.0, 2.0, 1.0},
        {2.0, 0.0, 0.0},
        10.0,
        config,
        &normals);

    REQUIRE(result.reachedTargetPlane);
    REQUIRE(result.points.size() >= 2);
    CHECK(result.points.front()[0] == doctest::Approx(3.0));
    CHECK(result.points.back()[0] == doctest::Approx(13.0));
    CHECK(result.points.back()[1] == doctest::Approx(2.0));
    CHECK(result.selectedTargetPlaneName == "extrapolation_distance");
    CHECK_THROWS_AS(
        vc::fiber_tracer::traceFiberExtrapolation(
            predictions, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.0, config),
        std::invalid_argument);
}

TEST_CASE("native fiber extrapolation retains its path at invalid directions")
{
    PositiveEdgePrediction predictions;
    ConstantNormalSampler normals;
    vc::fiber_tracer::FiberTraceConfig config;
    config.stepVoxels = 4.0;
    config.coneAngleDegrees = 0.0;
    config.beamWidth = 1;
    config.maxStepFactor = 2.0;
    config.smoothnessNormalWeight = 0.0;
    config.smoothnessTangentWeight = 0.0;
    config.cumulativeSmoothnessTangentWeight = 0.0;

    const auto result = vc::fiber_tracer::traceFiberExtrapolation(
        predictions,
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        12.0,
        config,
        &normals);

    CHECK_FALSE(result.reachedTargetPlane);
    CHECK(result.reason.starts_with("no_valid_candidates"));
    REQUIRE(result.points.size() == 2);
    CHECK(result.points.front()[0] == doctest::Approx(0.0));
    CHECK(result.points.back()[0] == doctest::Approx(4.0));
}
