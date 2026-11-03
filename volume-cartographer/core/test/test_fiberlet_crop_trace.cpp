#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberletCropTrace.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <thread>

namespace {
using namespace vc::fiber_tracer;

FiberletStorageKey key(std::int64_t x, std::uint8_t variant = 0)
{
    return {{0, 0, x}, variant};
}

class ZNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override
    {
        return true;
    }
    [[nodiscard]] vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override { return {{0, 0, 1}, true, {}}; }
};

class SerialZNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    [[nodiscard]] vc::lasagna::NormalSample sampleNormal(
        const cv::Vec3d&) const override
    {
        return {{0, 0, 1}, true, {}};
    }
};

class TestGraph final : public FiberletReplayGraphSource
{
public:
    void addAnchor(FiberletStorageKey id, cv::Vec3d point) { positions[id] = point; }

    void connect(FiberletStorageKey first, FiberletStorageKey second)
    {
        FiberletStorageId physical{std::min(first, second), std::max(first, second)};
        const bool reverse = first != physical.first;
        DirectedFiberletStorageId directed{physical, reverse};
        adjacency[first].push_back(directed);
        directed.reverse = !directed.reverse;
        adjacency[second].push_back(directed);
    }

    void setCostDensity(
        FiberletStorageKey first,
        FiberletStorageKey second,
        float density)
    {
        edgeCostDensity[{std::min(first, second), std::max(first, second)}] =
            density;
    }

    [[nodiscard]] bool supportsConcurrentQueries() const noexcept override { return concurrentQueries; }
    [[nodiscard]] float predictionToBaseScale() const noexcept override { return 1.0F; }
    [[nodiscard]] int anchorCellSizePredictionVoxels() const noexcept override { return 4; }
    [[nodiscard]] float maximumJoinAngleDegrees() const noexcept override { return 45.0F; }
    [[nodiscard]] std::vector<FiberletReplaySourceAnchor> anchorsNearReference(const PolylineArcGeometry&, double, double, double) const override
    {
        return {};
    }
    [[nodiscard]] std::vector<DirectedFiberletStorageId> outgoing(const FiberletStorageKey& anchor) const override
    {
        struct ActiveQuery {
            explicit ActiveQuery(const TestGraph& owner) : owner_(owner)
            {
                if (!owner_.observeConcurrency)
                    return;
                const int active = owner_.activeQueries.fetch_add(1) + 1;
                int maximum = owner_.maximumConcurrentQueries.load();
                while (active > maximum && !owner_.maximumConcurrentQueries.compare_exchange_weak(maximum, active)) {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            ~ActiveQuery()
            {
                if (owner_.observeConcurrency)
                    owner_.activeQueries.fetch_sub(1);
            }
            const TestGraph& owner_;
        } activeQuery(*this);
        if (const auto delay = queryDelay.find(anchor); delay != queryDelay.end())
            std::this_thread::sleep_for(delay->second);
        if (const auto failure = queryFailure.find(anchor); failure != queryFailure.end())
            throw std::runtime_error(failure->second);
        auto found = adjacency.find(anchor);
        if (found == adjacency.end())
            return {};
        auto result = found->second;
        std::sort(result.begin(), result.end());
        return result;
    }
    [[nodiscard]] FiberletReplaySourceArc arc(const DirectedFiberletStorageId& id) const override
    {
        const auto source = id.reverse ? id.fiberlet.second : id.fiberlet.first;
        const auto target = id.reverse ? id.fiberlet.first : id.fiberlet.second;
        const cv::Vec3d delta = positions.at(target) - positions.at(source);
        const float edgeLength = static_cast<float>(cv::norm(delta));
        const auto found = edgeCostDensity.find(id.fiberlet);
        const float density = found == edgeCostDensity.end()
            ? 1.0F
            : found->second;
        return {id, source, target, positions.at(source), positions.at(target), cv::Vec3f(delta), cv::Vec3f(delta), edgeLength, {0, edgeLength * density, 0, 0, 0}, std::nullopt, std::nullopt};
    }
    [[nodiscard]] FiberletReplaySourceCostProfile costProfile(const DirectedFiberletStorageId& id) const override
    {
        const auto edge = arc(id);
        return {{edge.pathLengthPredictionVoxels},
                {edge.cost.total() / edge.pathLengthPredictionVoxels}};
    }
    [[nodiscard]] std::vector<cv::Vec3d> routePoints(const DirectedFiberletStorageId& id) const override
    {
        const auto edge = arc(id);
        return {edge.sourcePositionBaseXYZ, edge.targetPositionBaseXYZ};
    }
    [[nodiscard]] std::optional<FiberletReplaySourceTransition> transition(const FiberletReplaySourceArc& incoming, const FiberletReplaySourceArc& outgoing) const override
    {
        if (incoming.target != outgoing.source || incoming.id.fiberlet == outgoing.id.fiberlet) {
            return std::nullopt;
        }
        const cv::Vec3d left = incoming.endStepBaseXYZ / cv::norm(incoming.endStepBaseXYZ);
        const cv::Vec3d right = outgoing.startStepBaseXYZ / cv::norm(outgoing.startStepBaseXYZ);
        if (!(left.dot(right) > std::cos(45.0 * 3.14159265358979323846 / 180.0)))
            return std::nullopt;
        return FiberletReplaySourceTransition{incoming.id, outgoing.id, {0.0F, joinCost, 0.0F, 0.0F, 0.0F}, std::nullopt};
    }

    std::map<FiberletStorageKey, cv::Vec3d> positions;
    std::map<FiberletStorageKey, std::vector<DirectedFiberletStorageId>> adjacency;
    std::map<FiberletStorageKey, std::chrono::milliseconds> queryDelay;
    std::map<FiberletStorageKey, std::string> queryFailure;
    std::map<FiberletStorageId, float> edgeCostDensity;
    bool concurrentQueries = false;
    bool observeConcurrency = false;
    mutable std::atomic<int> activeQueries{0};
    mutable std::atomic<int> maximumConcurrentQueries{0};
    float joinCost = 0.0F;
};

FiberletStoredAnchor anchor(FiberletStorageKey id, cv::Vec3f point, cv::Vec3f axis, float presence)
{
    FiberletStoredAnchor result;
    result.key = id;
    result.positionPredictionXYZ = point;
    result.fittedAxisXYZ = axis;
    result.predictionPresence = presence;
    result.predictionValid = true;
    result.predictionPresenceValid = true;
    return result;
}

void addAmbiguityTestGraph(
    TestGraph& graph,
    double alternativeOffsetZ,
    float alternativeDensity,
    bool shortAlternative = false)
{
    const auto leftEnd = key(-100);
    const auto leftMiddle = key(-50);
    const auto seed = key(0);
    const auto bestMiddle = key(50);
    const auto bestEnd = key(100);
    const auto alternativeMiddle = key(50, 1);
    const auto alternativeEnd = key(100, 1);
    graph.addAnchor(leftEnd, {-100, 0, 0});
    graph.addAnchor(leftMiddle, {-50, 0, 0});
    graph.addAnchor(seed, {0, 0, 0});
    graph.addAnchor(bestMiddle, {50, 0, 0});
    graph.addAnchor(bestEnd, {100, 0, 0});
    graph.addAnchor(alternativeMiddle, {50, 0, alternativeOffsetZ});
    graph.addAnchor(alternativeEnd, {100, 0, alternativeOffsetZ});
    graph.connect(leftEnd, leftMiddle);
    graph.connect(leftMiddle, seed);
    graph.connect(seed, bestMiddle);
    graph.connect(bestMiddle, bestEnd);
    graph.connect(seed, alternativeMiddle);
    if (!shortAlternative)
        graph.connect(alternativeMiddle, alternativeEnd);
    graph.setCostDensity(seed, alternativeMiddle, alternativeDensity);
    if (!shortAlternative) {
        graph.setCostDensity(
            alternativeMiddle, alternativeEnd, alternativeDensity);
    }
}

} // namespace

TEST_CASE("Dense beam crop shares deterministic coverage")
{
    class StraightField final : public FiberPredictionSource {
    public:
        bool supportsConcurrentSampling() const noexcept override { return true; }
        FiberPredictionSample sample(const cv::Vec3d&, const cv::Vec3d&) const override {
            FiberPredictionSample result;
            result.options.push_back({{1, 0, 0}, 1.0F, true});
            return result;
        }
    } field;
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, 0, 0};
    config.maximumBaseXYZ = {64, 64, 64};
    config.parallelThreads = 1;
    FiberTraceConfig dense;
    dense.coneGridSize = 5;
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(key(1), {32, 32, 32}, {1, 0, 0}, 1.0F),
        anchor(key(2), {40, 32, 32}, {1, 0, 0}, 0.9F)};
    const auto serial = traceDenseBeamCrop(field, anchors, 1.0, normals, dense, config);
    REQUIRE(serial.lines.size() == 1);
    CHECK(serial.coveredAnchors == 1);
    CHECK(serial.lines[0].negativeTermination == "crop_exit");
    CHECK(serial.lines[0].positiveTermination == "crop_exit");
    CHECK(serial.lines[0].pointsBaseXYZ.front()[0] == doctest::Approx(0));
    CHECK(serial.lines[0].pointsBaseXYZ.back()[0] == doctest::Approx(64));
    CHECK(serial.lines[0].pathLengthPredictionVoxels == doctest::Approx(64));
    CHECK(std::isfinite(serial.lines[0].totalMetricCost));
    config.parallelThreads = 2;
    const auto concurrent = traceDenseBeamCrop(field, anchors, 1.0, normals, dense, config);
    REQUIRE(concurrent.lines.size() == 1);
    CHECK(concurrent.lines[0].pointsBaseXYZ == serial.lines[0].pointsBaseXYZ);
    CHECK(concurrent.lines[0].totalMetricCost == serial.lines[0].totalMetricCost);

}

TEST_CASE("Dense indexed coverage stops converging traces with ordered parallel revalidation")
{
    class ConvergingField final : public FiberPredictionSource {
    public:
        bool supportsConcurrentSampling() const noexcept override { return true; }
        FiberPredictionSample sample(const cv::Vec3d&, const cv::Vec3d& hint) const override {
            FiberPredictionSample s;
            cv::Vec3d d{1.,std::abs(hint[1])>.1 ? -.25 : 0.,0.}; d/=cv::norm(d);
            s.options.push_back({d,1.F,true}); return s;
        }
    } field;
    ZNormalSampler normals;
    FiberletCropTraceConfig c;
    c.minimumBaseXYZ={0,0,0}; c.maximumBaseXYZ={128,64,64};
    c.coverageNormalRadiusBaseVoxels=2.; c.coverageTangentialRadiusBaseVoxels=3.;
    c.coverageDirectionDegrees=45.;
    FiberTraceConfig dense; dense.coneGridSize=5; dense.stepVoxels=2.;
    std::vector<FiberletStoredAnchor> anchors{
        anchor(key(1),{64,32,32},{1,0,0},1.F),
        anchor(key(2),{8,50,32},{1,-.25,0},.9F)};
    c.parallelThreads=1;
    const auto unrestricted=traceDenseBeamCrop(field,anchors,1.,normals,dense,c);
    c.stopAtCoveredTraces=true;
    c.captureStepCosts=true;
    for(bool reverse:{false,true}) {
        c.reverseConsistency=reverse; c.reversePrefixIntervalBaseVoxels=reverse ? 16. : 0.;
        c.parallelThreads=1;
        const auto serial=traceDenseBeamCrop(field,anchors,1.,normals,dense,c);
        c.parallelThreads=4;
        const auto parallel=traceDenseBeamCrop(field,anchors,1.,normals,dense,c);
        CHECK(serial.coveredTraceStops>0);
        CHECK(parallel.coverageCommitRechecks>0);
        REQUIRE(serial.lines.size()==parallel.lines.size());
        for(std::size_t i=0;i<serial.lines.size();++i) {
            CHECK(serial.lines[i].pointsBaseXYZ==parallel.lines[i].pointsBaseXYZ);
            CHECK(serial.lines[i].totalMetricCost==parallel.lines[i].totalMetricCost);
            CHECK(serial.lines[i].pathLengthPredictionVoxels==parallel.lines[i].pathLengthPredictionVoxels);
            CHECK(serial.lines[i].segmentMetricCosts==parallel.lines[i].segmentMetricCosts);
            REQUIRE(serial.lines[i].segmentMetricCosts.size()+1==serial.lines[i].pointsBaseXYZ.size());
        }
        if(!reverse) {
            REQUIRE(serial.lines.size()==unrestricted.lines.size());
            CHECK(serial.lines.back().pathLengthPredictionVoxels<unrestricted.lines.back().pathLengthPredictionVoxels);
        }
    }
}

TEST_CASE("Dense indexed coverage preserves separated parallel and orthogonal traces")
{
    class AxisField final : public FiberPredictionSource {
    public:
        bool supportsConcurrentSampling() const noexcept override { return true; }
        FiberPredictionSample sample(const cv::Vec3d&,const cv::Vec3d& hint) const override {
            FiberPredictionSample s;
            s.options.push_back({std::abs(hint[0])>std::abs(hint[1]) ? cv::Vec3d{1,0,0} : cv::Vec3d{0,1,0},1.F,true});return s;
        }
    } field;
    ZNormalSampler normals; FiberTraceConfig dense; dense.coneGridSize=5;
    FiberletCropTraceConfig c; c.minimumBaseXYZ={0,0,0};c.maximumBaseXYZ={64,64,64};
    c.coverageNormalRadiusBaseVoxels=2;c.coverageTangentialRadiusBaseVoxels=3;c.coverageDirectionDegrees=45;
    c.parallelThreads=1;
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(key(1),{32,32,32},{1,0,0},1.F),
        anchor(key(2),{32,40,32},{-1,0,0},.9F),
        anchor(key(3),{32,8,32},{0,1,0},.8F)};
    const auto baseline=traceDenseBeamCrop(field,anchors,1.,normals,dense,c);
    c.stopAtCoveredTraces=true;c.parallelThreads=4;
    const auto stopped=traceDenseBeamCrop(field,anchors,1.,normals,dense,c);
    REQUIRE(baseline.lines.size()==3);REQUIRE(stopped.lines.size()==3);
    for(std::size_t i=0;i<3;++i)CHECK(baseline.lines[i].pointsBaseXYZ==stopped.lines[i].pointsBaseXYZ);
    CHECK(stopped.coveredTraceStops==0);
}

TEST_CASE("Dense beam crop keeps trace prediction and base coordinate units distinct")
{
    class StraightField final : public FiberPredictionSource {
    public:
        FiberPredictionSample sample(const cv::Vec3d&, const cv::Vec3d&) const override {
            FiberPredictionSample result;
            result.options.push_back({{1, 0, 0}, 1.0F, true});
            return result;
        }
    } field;
    class RecordingNormals final : public vc::lasagna::NormalSampler {
    public:
        mutable std::vector<cv::Vec3d> points;
        vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& point) const override {
            points.push_back(point);
            return {{0, 0, 1}, true, {}};
        }
    } normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, 0, 0};
    config.maximumBaseXYZ = {64, 64, 64};
    config.parallelThreads = 1;
    FiberTraceConfig dense;
    dense.traceToBaseScale = 2.0;
    dense.coneGridSize = 5;
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(key(1), {4, 4, 4}, {1, 0, 0}, 1.0F),
        anchor(key(2), {5, 4, 4}, {1, 0, 0}, 0.9F)};
    const auto result = traceDenseBeamCrop(field, anchors, 8.0, normals, dense, config);
    REQUIRE(result.lines.size() == 1);
    CHECK(result.coveredAnchors == 1);
    const auto& line = result.lines.front();
    CHECK(line.seedBaseXYZ == cv::Vec3d(32, 32, 32));
    CHECK(line.pointsBaseXYZ.front() == cv::Vec3d(0, 32, 32));
    CHECK(line.pointsBaseXYZ.back() == cv::Vec3d(64, 32, 32));
    CHECK(line.pathLengthPredictionVoxels == doctest::Approx(8));
    CHECK(line.negativeTermination == "crop_exit");
    CHECK(line.positiveTermination == "crop_exit");
    REQUIRE_FALSE(normals.points.empty());
    // Both tracing and coverage must query the normal sampler in trace units,
    // around z=16, not in prediction (z=4) or base (z=32) coordinates.
    for (const auto& point : normals.points) {
        CHECK(point[2] > 8.0);
        CHECK(point[2] < 24.0);
    }
}

TEST_CASE("Dense reverse consistency preserves a reversible route and its cost")
{
    class StraightField final : public FiberPredictionSource {
    public:
        bool supportsConcurrentSampling() const noexcept override { return true; }
        FiberPredictionSample sample(const cv::Vec3d&, const cv::Vec3d&) const override {
            FiberPredictionSample result;
            result.options.push_back({{1, 0, 0}, 0.9F, true});
            return result;
        }
    } field;
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, 0, 0};
    config.maximumBaseXYZ = {256, 256, 256};
    config.parallelThreads = 1;
    FiberTraceConfig dense;
    dense.traceToBaseScale = 2;
    dense.coneGridSize = 5;
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(key(1), {16, 16, 16}, {1, 0, 0}, 1)};
    const auto original = traceDenseBeamCrop(field, anchors, 8, normals, dense, config);
    REQUIRE(original.lines.size() == 1);
    config.captureStepCosts = true;
    const auto diagnostic = traceDenseBeamCrop(field, anchors, 8, normals, dense, config);
    CHECK(diagnostic.lines.front().pointsBaseXYZ == original.lines.front().pointsBaseXYZ);
    CHECK(diagnostic.lines.front().totalMetricCost == original.lines.front().totalMetricCost);
    const auto checkCosts = [](const FiberletCropTraceLine& line) {
        REQUIRE(line.segmentMetricCosts.size() + 1 == line.pointsBaseXYZ.size());
        REQUIRE(line.segmentOriginalStepCosts.size() == line.segmentMetricCosts.size());
        double sum = 0;
        for (std::size_t i = 0; i < line.segmentMetricCosts.size(); ++i) {
            sum += line.segmentMetricCosts[i];
            CHECK(line.segmentMetricCosts[i] >= 0);
            CHECK(line.segmentOriginalStepCosts[i] + 1e-9 >= line.segmentMetricCosts[i]);
        }
        CHECK(sum == doctest::Approx(line.totalMetricCost));
    };
    checkCosts(diagnostic.lines.front());
    config.reverseConsistency = true;
    const auto checked = traceDenseBeamCrop(field, anchors, 8, normals, dense, config);
    REQUIRE(checked.lines.size() == 1);
    checkCosts(checked.lines.front());
    CHECK(checked.reverseAttempts == 2);
    CHECK(checked.reverseFailures == 0);
    CHECK(checked.reverseSkippedLengthBaseVoxels == 0);
    CHECK(checked.lines.front().pointsBaseXYZ == original.lines.front().pointsBaseXYZ);
    CHECK(checked.lines.front().totalMetricCost == doctest::Approx(original.lines.front().totalMetricCost));
    CHECK(checked.lines.front().pathLengthPredictionVoxels == original.lines.front().pathLengthPredictionVoxels);
    CHECK_FALSE(checked.lines.front().originalSeedBaseXYZ);
    config.parallelThreads = 4;
    CHECK(traceDenseBeamCrop(field, anchors, 8, normals, dense, config).lines.front().pointsBaseXYZ == checked.lines.front().pointsBaseXYZ);
    config.reversePrefixIntervalBaseVoxels = 32;
    const auto periodic = traceDenseBeamCrop(field, anchors, 8, normals, dense, config);
    REQUIRE(periodic.lines.size() == 1);
    CHECK(periodic.reverseAttempts > checked.reverseAttempts);
    CHECK(periodic.reverseFailures == 0);
    CHECK(periodic.reverseSkippedLengthBaseVoxels == 0);
    CHECK(periodic.lines.front().pointsBaseXYZ == original.lines.front().pointsBaseXYZ);
    checkCosts(periodic.lines.front());
    config.reversePrefixIntervalBaseVoxels = 0;
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        auto bad = config;
        bad.reverseBackoffBaseVoxels = invalid;
        CHECK_THROWS(traceDenseBeamCrop(field, anchors, 8, normals, dense, bad));
        bad = config;
        bad.reverseNormalRadiusBaseVoxels = invalid;
        CHECK_THROWS(traceDenseBeamCrop(field, anchors, 8, normals, dense, bad));
        bad = config;
        bad.reverseTangentialRadiusBaseVoxels = invalid;
        CHECK_THROWS(traceDenseBeamCrop(field, anchors, 8, normals, dense, bad));
    }
    const std::vector<std::vector<cv::Vec3d>> imported{{{0, 128, 128}, {256, 128, 128}}};
    CHECK_THROWS(importFiberCropLines(imported, config));
}

TEST_CASE("Dense reverse consistency preserves gaps and split provenance")
{
    class OneWayGap final : public FiberPredictionSource {
    public:
        FiberPredictionSample sample(const cv::Vec3d& point, const cv::Vec3d& direction) const override {
            FiberPredictionSample result;
            if (direction[0] < 0 && point[0] > 160 && point[0] < 192)
                return result;
            result.options.push_back({{1, 0, 0}, 0.9F, true});
            return result;
        }
    } field;
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, 0, 0};
    config.maximumBaseXYZ = {256, 256, 256};
    config.parallelThreads = 1;
    config.reverseConsistency = true;
    config.captureStepCosts = true;
    FiberTraceConfig dense;
    dense.coneGridSize = 5;
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(key(1), {128, 128, 128}, {1, 0, 0}, 1)};
    const auto checked = traceDenseBeamCrop(field, anchors, 1, normals, dense, config);
    REQUIRE(checked.lines.size() >= 2);
    CHECK(checked.reverseFailures >= 1);
    CHECK(checked.reverseSkippedLengthBaseVoxels >= 40);
    double retained = 0;
    std::size_t relocatedSeeds = 0;
    for (const auto& line : checked.lines) {
        retained += line.pathLengthPredictionVoxels;
        REQUIRE(line.segmentMetricCosts.size() + 1 == line.pointsBaseXYZ.size());
        REQUIRE(line.segmentOriginalStepCosts.size() == line.segmentMetricCosts.size());
        double segmentSum = 0.0;
        for (std::size_t i = 0; i < line.segmentMetricCosts.size(); ++i) {
            segmentSum += line.segmentMetricCosts[i];
            CHECK(line.segmentMetricCosts[i] >= -1e-9);
            CHECK(line.segmentMetricCosts[i] <= line.segmentOriginalStepCosts[i] + 1e-9);
        }
        CHECK(segmentSum == doctest::Approx(line.totalMetricCost));
        CHECK(line.totalMetricCost >= 0);
        CHECK(std::find(line.pointsBaseXYZ.begin(), line.pointsBaseXYZ.end(), line.seedBaseXYZ) != line.pointsBaseXYZ.end());
        if (line.originalSeedBaseXYZ) {
            ++relocatedSeeds;
            CHECK(*line.originalSeedBaseXYZ == cv::Vec3d(128, 128, 128));
        }
        CHECK_FALSE(line.pointsBaseXYZ.front()[0] < 160 && line.pointsBaseXYZ.back()[0] > 192);
    }
    CHECK(relocatedSeeds >= 1);
    CHECK(retained + checked.reverseSkippedLengthBaseVoxels == doctest::Approx(256));
}

TEST_CASE("Dense reverse consistency detects lateral disagreement and bounds immediate failure")
{
    class ReverseField final : public FiberPredictionSource {
    public:
        bool missing = false;
        bool supportsConcurrentSampling() const noexcept override { return true; }
        FiberPredictionSample sample(const cv::Vec3d& point, const cv::Vec3d& direction) const override {
            FiberPredictionSample result;
            const bool reversePositiveHalf = direction[0] < 0 && point[0] > 128;
            if (reversePositiveHalf && missing)
                return result;
            result.options.push_back({reversePositiveHalf ? cv::Vec3f(1, -0.5F, 0) : cv::Vec3f(1, 0, 0), 1, true});
            return result;
        }
    } field;
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, 0, 0};
    config.maximumBaseXYZ = {256, 1024, 256};
    config.parallelThreads = 1;
    config.reverseConsistency = true;
    config.reverseTangentialRadiusBaseVoxels = 10;
    FiberTraceConfig dense;
    dense.coneGridSize = 5;
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(key(1), {128, 128, 128}, {1, 0, 0}, 1),
        anchor(key(2), {128, 512, 128}, {1, 0, 0}, 0.9F)};
    const auto tight = traceDenseBeamCrop(field, anchors, 1, normals, dense, config);
    CHECK(tight.reverseFailures > 0);
    CHECK(tight.reverseSkippedLengthBaseVoxels > 0);
    auto parallel = config;
    parallel.parallelThreads = 4;
    const auto concurrent = traceDenseBeamCrop(field, anchors, 1, normals, dense, parallel);
    REQUIRE(concurrent.lines.size() == tight.lines.size());
    CHECK(concurrent.reverseAttempts == tight.reverseAttempts);
    for (std::size_t i = 0; i < tight.lines.size(); ++i)
        CHECK(concurrent.lines[i].pointsBaseXYZ == tight.lines[i].pointsBaseXYZ);
    auto loose = config;
    loose.reverseTangentialRadiusBaseVoxels = 200;
    const auto wide = traceDenseBeamCrop(field, anchors, 1, normals, dense, loose);
    CHECK(wide.reverseSkippedLengthBaseVoxels < tight.reverseSkippedLengthBaseVoxels);
    auto prefixConfig = config;
    prefixConfig.reversePrefixIntervalBaseVoxels = 64;
    const auto prefix = traceDenseBeamCrop(field, anchors, 1, normals, dense, prefixConfig);
    CHECK(prefix.reverseFailures > 0);
    CHECK(prefix.reverseSkippedLengthBaseVoxels > 0);
    double retained = 0;
    for (const auto& line : prefix.lines)
        retained += line.pathLengthPredictionVoxels;
    CHECK(retained + prefix.reverseSkippedLengthBaseVoxels == doctest::Approx(512));
    prefixConfig.parallelThreads = 4;
    const auto prefixParallel = traceDenseBeamCrop(field, anchors, 1, normals, dense, prefixConfig);
    REQUIRE(prefixParallel.lines.size() == prefix.lines.size());
    for (std::size_t i = 0; i < prefix.lines.size(); ++i)
        CHECK(prefixParallel.lines[i].pointsBaseXYZ == prefix.lines[i].pointsBaseXYZ);
    // A terminal check is mandatory even before the first interval.
    prefixConfig.reversePrefixIntervalBaseVoxels = 512;
    const auto terminal = traceDenseBeamCrop(field, anchors, 1, normals, dense, prefixConfig);
    CHECK(terminal.reverseAttempts == 4);
    CHECK(terminal.reverseFailures > 0);
    field.missing = true;
    const auto stopped = traceDenseBeamCrop(field, anchors, 1, normals, dense, config);
    CHECK(stopped.reverseFailures > 0);
    CHECK(stopped.reverseAttempts < 20);
    CHECK(stopped.reverseSkippedLengthBaseVoxels == doctest::Approx(256));
    REQUIRE(stopped.lines.size() == 2);
    for (const auto& line : stopped.lines)
        CHECK(line.pathLengthPredictionVoxels == doctest::Approx(128));
}

TEST_CASE("imported crop preserves reentry runs and rejects quality scoring")
{
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0,0,0};
    config.maximumBaseXYZ = {10,10,10};
    const std::vector<std::vector<cv::Vec3d>> sources{
        {{-5,2,2},{15,2,2},{15,8,2},{-5,8,2}}};
    const auto lines = importFiberCropLines(sources, config);
    REQUIRE(lines.size() == 2);
    CHECK(lines[0].pointsBaseXYZ.front() == cv::Vec3d(0,2,2));
    CHECK(lines[0].pointsBaseXYZ.back() == cv::Vec3d(10,2,2));
    CHECK(lines[1].pointsBaseXYZ.front() == cv::Vec3d(10,8,2));
    CHECK(lines[1].pointsBaseXYZ.back() == cv::Vec3d(0,8,2));
    CHECK(lines[0].pathLengthPredictionVoxels == doctest::Approx(10));
    CHECK(lines[1].pathLengthPredictionVoxels == doctest::Approx(10));
    CHECK(lines[1].seed.coordinateZYX[2] == 1);
    CHECK(lines[1].seedBaseXYZ == cv::Vec3d(5,8,2));
    config.maximumAcceptedCostDensity = 0.21;
    CHECK_THROWS_AS(importFiberCropLines(sources, config), std::invalid_argument);
}

TEST_CASE("Fiberlet crop search box expands every face by exact lookahead")
{
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {10.0, 20.0, 30.0};
    config.maximumBaseXYZ = {110.0, 220.0, 330.0};
    config.lookaheadDistanceBaseVoxels = 48.0;
    const auto search = fiberletCropTraceSearchBox(config);
    CHECK(search.minimumBaseXYZ == cv::Vec3d{-38.0, -28.0, -18.0});
    CHECK(search.maximumBaseXYZ == cv::Vec3d{158.0, 268.0, 378.0});

    config.lookaheadDistanceBaseVoxels = 0.0;
    CHECK_THROWS_AS(
        fiberletCropTraceSearchBox(config), std::invalid_argument);
    config.lookaheadDistanceBaseVoxels = 48.0;
    config.maximumBaseXYZ[1] = config.minimumBaseXYZ[1];
    CHECK_THROWS_AS(
        fiberletCropTraceSearchBox(config), std::invalid_argument);
}

TEST_CASE("Fiberlet crop tracing stops before a close distinct lookahead route")
{
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-200, -200, -200};
    config.maximumBaseXYZ = {200, 200, 200};
    config.lookaheadDistanceBaseVoxels = 80.0;
    config.maximumAttempts = 1;
    config.parallelThreads = 1;
    config.ambiguityRelativeCostMargin = 0.1;
    config.ambiguityNormalRadiusBaseVoxels = 20.0;

    TestGraph graph;
    addAmbiguityTestGraph(graph, 25.0, 1.05F);
    const auto result = traceFiberletCrop(
        graph, {anchor(key(0), {0, 0, 0}, {1, 0, 0}, 1.0F)},
        normals, 1.0, config);

    REQUIRE(result.lines.size() == 1);
    const auto& line = result.lines.front();
    CHECK(line.negativeTermination == "graph_exhausted");
    CHECK(line.positiveTermination == "ambiguous_route");
    CHECK(line.negativeFiberlets == 2);
    CHECK(line.positiveFiberlets == 0);
    CHECK(line.pointsBaseXYZ.front() == cv::Vec3d{-100, 0, 0});
    CHECK(line.pointsBaseXYZ.back() == cv::Vec3d{0, 0, 0});
    CHECK(result.ambiguityDecisions == 1);
    CHECK(result.acceptedAmbiguityStops == 1);
    CHECK(result.ambiguityRouteComparisons >= 1);
    REQUIRE(result.minimumAmbiguityRelativeCostGap.has_value());
    CHECK(*result.minimumAmbiguityRelativeCostGap ==
          doctest::Approx(0.05).epsilon(1.0e-5));
    CHECK(result.maximumAmbiguityThresholdRatio > 1.0);

    graph.concurrentQueries = true;
    config.parallelThreads = 4;
    const auto parallel = traceFiberletCrop(
        graph, {anchor(key(0), {0, 0, 0}, {1, 0, 0}, 1.0F)},
        normals, 1.0, config);
    REQUIRE(parallel.lines.size() == 1);
    CHECK(parallel.lines.front().pointsBaseXYZ == line.pointsBaseXYZ);
    CHECK(parallel.lines.front().negativeTermination ==
          line.negativeTermination);
    CHECK(parallel.lines.front().positiveTermination ==
          line.positiveTermination);
    CHECK(parallel.ambiguityDecisions == result.ambiguityDecisions);
    CHECK(parallel.acceptedAmbiguityStops == result.acceptedAmbiguityStops);
}

TEST_CASE("Fiberlet crop ambiguity ignores ineligible alternatives")
{
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-200, -200, -200};
    config.maximumBaseXYZ = {200, 200, 200};
    config.lookaheadDistanceBaseVoxels = 80.0;
    config.maximumAttempts = 1;
    config.parallelThreads = 1;
    config.ambiguityRelativeCostMargin = 0.1;
    config.ambiguityNormalRadiusBaseVoxels = 20.0;

    SUBCASE("an alternative inside the anisotropic tube is equivalent")
    {
        TestGraph graph;
        addAmbiguityTestGraph(graph, 10.0, 1.0F);
        const auto result = traceFiberletCrop(
            graph, {anchor(key(0), {0, 0, 0}, {1, 0, 0}, 1.0F)},
            normals, 1.0, config);
        REQUIRE(result.lines.size() == 1);
        CHECK(result.lines.front().positiveTermination == "graph_exhausted");
        CHECK(result.ambiguityDecisions == 0);
        CHECK(result.acceptedAmbiguityStops == 0);
    }

    SUBCASE("an expensive distinct route is outside the cost margin")
    {
        TestGraph graph;
        addAmbiguityTestGraph(graph, 25.0, 1.2F);
        const auto result = traceFiberletCrop(
            graph, {anchor(key(0), {0, 0, 0}, {1, 0, 0}, 1.0F)},
            normals, 1.0, config);
        REQUIRE(result.lines.size() == 1);
        CHECK(result.lines.front().positiveTermination == "graph_exhausted");
        CHECK(result.ambiguityDecisions == 0);
    }

    SUBCASE("a within-margin short dead end does not compete with a horizon route")
    {
        TestGraph graph;
        addAmbiguityTestGraph(graph, 25.0, 1.05F, true);
        const auto result = traceFiberletCrop(
            graph, {anchor(key(0), {0, 0, 0}, {1, 0, 0}, 1.0F)},
            normals, 1.0, config);
        REQUIRE(result.lines.size() == 1);
        CHECK(result.lines.front().positiveTermination == "graph_exhausted");
        CHECK(result.ambiguityDecisions == 0);
    }

    config.ambiguityRelativeCostMargin = -0.1;
    TestGraph invalidGraph;
    addAmbiguityTestGraph(invalidGraph, 25.0, 1.05F);
    CHECK_THROWS_AS(
        traceFiberletCrop(
            invalidGraph,
            {anchor(key(0), {0, 0, 0}, {1, 0, 0}, 1.0F)},
            normals, 1.0, config),
        std::invalid_argument);
}

TEST_CASE("Fiberlet crop tracing is bidirectional and uses anisotropic directional coverage")
{
    TestGraph graph;
    graph.joinCost = 2.0F;
    const auto outsideLeft = key(-10);
    const auto left = key(20);
    const auto seed = key(50);
    const auto right = key(80);
    const auto outsideRight = key(110);
    graph.addAnchor(outsideLeft, {-10, 0, 0});
    graph.addAnchor(left, {20, 0, 0});
    graph.addAnchor(seed, {50, 0, 0});
    graph.addAnchor(right, {80, 0, 0});
    graph.addAnchor(outsideRight, {110, 0, 0});
    graph.connect(outsideLeft, left);
    graph.connect(left, seed);
    graph.connect(seed, right);
    graph.connect(right, outsideRight);

    const auto parallel = key(51);
    const auto crossing = key(52);
    const auto normalFar = key(53);
    std::vector<FiberletStoredAnchor> anchors{
        anchor(seed, {50, 0, 0}, {1, 0, 0}, 1.0F),
        anchor(parallel, {50, 50, 0}, {1, 0, 0}, 0.8F),
        anchor(crossing, {50, 0, 0}, {0, 1, 0}, 0.7F),
        anchor(normalFar, {50, 0, 21}, {1, 0, 0}, 0.6F),
    };
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, -100, -100};
    config.maximumBaseXYZ = {100, 100, 100};
    config.lookaheadDistanceBaseVoxels = 48;
    const auto result = traceFiberletCrop(graph, anchors, normals, 1.0, config);

    REQUIRE(result.lines.size() == 1);
    CHECK(result.bidirectionalLines == 1);
    CHECK(result.oneSidedLines == 0);
    CHECK(result.coveredAnchors == 1);
    CHECK(result.attemptedAnchors == 3);
    CHECK(result.noEdgeAnchors == 2);
    CHECK(result.lines.front().negativeTermination == "crop_boundary");
    CHECK(result.lines.front().positiveTermination == "crop_boundary");
    CHECK(result.lines.front().seedBaseXYZ == cv::Vec3d{50, 0, 0});
    REQUIRE(result.lines.front().pointsBaseXYZ.size() >= 5);
    CHECK(result.lines.front().pointsBaseXYZ.front()[0] == doctest::Approx(0));
    CHECK(result.lines.front().pointsBaseXYZ.back()[0] == doctest::Approx(100));
    CHECK(result.lines.front().pathLengthPredictionVoxels == doctest::Approx(100.0));
    CHECK(result.lines.front().totalMetricCost == doctest::Approx(106.0));

    auto limitedConfig = config;
    limitedConfig.maximumAttempts = 2;
    const auto limited = traceFiberletCrop(graph, anchors, normals, 1.0, limitedConfig);
    CHECK(limited.attemptedAnchors == 2);
    CHECK(limited.coveredAnchors == 1);
    CHECK(limited.lines.size() == 1);
    CHECK(limited.noEdgeAnchors == 1);

    auto wide = config;
    wide.coverageNormalRadiusBaseVoxels = 40;
    wide.coverageTangentialRadiusBaseVoxels = 80;
    CHECK(fiberletCropCoverageTangentialRadius(wide) == 80);
    const auto widened = traceFiberletCrop(graph, anchors, normals, 1.0, wide);
    CHECK(widened.coveredAnchors == 2); // Normal offset 21 now covered.
    CHECK(widened.attemptedAnchors == 2); // Crossing direction still excluded.
    auto far = anchors;
    far[1].positionPredictionXYZ = {50, 90, 0};
    const auto fixedTangent = traceFiberletCrop(graph, far, normals, 1.0, wide);
    CHECK(fixedTangent.coveredAnchors == 1); // 90 in-sheet is NOT covered.
    wide.coverageTangentialRadiusBaseVoxels.reset();
    CHECK(fiberletCropCoverageTangentialRadius(wide) == 160);
    CHECK(traceFiberletCrop(graph, far, normals, 1.0, wide).coveredAnchors == 2);
    auto explicitDefault = config;
    explicitDefault.coverageTangentialRadiusBaseVoxels = 80;
    CHECK(traceFiberletCrop(graph, anchors, normals, 1.0, explicitDefault).coveredAnchors == result.coveredAnchors);
    for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        wide.coverageTangentialRadiusBaseVoxels = invalid;
        CHECK_THROWS(fiberletCropCoverageTangentialRadius(wide));
    }
}

TEST_CASE("Fiberlet crop tracing accepts an immediate minimum-face exit")
{
    TestGraph graph;
    const auto outsideLeft = key(-10);
    const auto seed = key(0);
    const auto right = key(40);
    const auto outsideRight = key(110);
    graph.addAnchor(outsideLeft, {-10, 0, 0});
    graph.addAnchor(seed, {0, 0, 0});
    graph.addAnchor(right, {40, 0, 0});
    graph.addAnchor(outsideRight, {110, 0, 0});
    graph.connect(outsideLeft, seed);
    graph.connect(seed, right);
    graph.connect(right, outsideRight);

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, -100, -100};
    config.maximumBaseXYZ = {100, 100, 100};
    config.lookaheadDistanceBaseVoxels = 48;
    config.maximumAttempts = 1;
    ZNormalSampler normals;
    const auto result = traceFiberletCrop(
        graph,
        {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)},
        normals,
        1.0,
        config);

    REQUIRE(result.lines.size() == 1);
    const auto& line = result.lines.front();
    CHECK(line.negativeTermination == "crop_boundary");
    CHECK(line.positiveTermination == "crop_boundary");
    CHECK(line.negativeFiberlets == 1);
    CHECK(line.positiveFiberlets == 2);
    CHECK(line.pathLengthPredictionVoxels == doctest::Approx(100.0));
    REQUIRE(line.pointsBaseXYZ.size() == 3);
    CHECK(line.pointsBaseXYZ.front() == cv::Vec3d{0, 0, 0});
    CHECK(line.pointsBaseXYZ.back() == cv::Vec3d{100, 0, 0});
}

TEST_CASE("Fiberlet crop tracing rejects a one-sided zero-length crop exit")
{
    TestGraph graph;
    const auto outside = key(-10);
    const auto seed = key(0);
    graph.addAnchor(outside, {-10, 0, 0});
    graph.addAnchor(seed, {0, 0, 0});
    graph.connect(outside, seed);

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, -100, -100};
    config.maximumBaseXYZ = {100, 100, 100};
    config.lookaheadDistanceBaseVoxels = 48;
    config.maximumAttempts = 1;
    ZNormalSampler normals;

    for (const bool stopAtCovered : {false, true}) {
        config.stopAtCoveredAnchors = stopAtCovered;
        const auto result = traceFiberletCrop(
            graph,
            {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)},
            normals,
            1.0,
            config);
        CHECK(result.attemptedAnchors == 1);
        CHECK(result.noEdgeAnchors == 1);
        CHECK(result.lines.empty());
        CHECK(result.coveredAnchorStops == 0);
    }
}

TEST_CASE("Fiberlet crop lookahead ranks beyond output boundary but clips output")
{
    TestGraph graph;
    const auto seed = key(50);
    const auto branch = key(90);
    const auto cheapFirst = key(110);
    const auto goodFirst = key(110, 1);
    const auto expensiveFuture = key(160);
    const auto goodFuture = key(160, 1);
    graph.addAnchor(seed, {50, 0, 0});
    graph.addAnchor(branch, {90, 0, 0});
    graph.addAnchor(cheapFirst, {110, 0, 0});
    graph.addAnchor(goodFirst, {110, 10, 0});
    graph.addAnchor(expensiveFuture, {160, 0, 0});
    graph.addAnchor(goodFuture, {160, 10, 0});
    graph.connect(seed, branch);
    graph.connect(branch, cheapFirst);
    graph.connect(branch, goodFirst);
    graph.connect(cheapFirst, expensiveFuture);
    graph.connect(goodFirst, goodFuture);
    graph.setCostDensity(branch, cheapFirst, 0.1F);
    graph.setCostDensity(cheapFirst, expensiveFuture, 10.0F);
    graph.setCostDensity(branch, goodFirst, 0.2F);
    graph.setCostDensity(goodFirst, goodFuture, 0.01F);

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {0, -100, -100};
    config.maximumBaseXYZ = {100, 100, 100};
    config.lookaheadDistanceBaseVoxels = 40.0;
    config.maximumAttempts = 1;
    ZNormalSampler normals;
    const auto result = traceFiberletCrop(
        graph,
        {anchor(seed, {50, 0, 0}, {1, 0, 0}, 1.0F)},
        normals,
        1.0,
        config);

    REQUIRE(result.lines.size() == 1);
    const auto& line = result.lines.front();
    CHECK(line.positiveTermination == "crop_boundary");
    REQUIRE(line.pointsBaseXYZ.size() == 3);
    CHECK(line.pointsBaseXYZ[0] == cv::Vec3d{50, 0, 0});
    CHECK(line.pointsBaseXYZ[1] == cv::Vec3d{90, 0, 0});
    CHECK(line.pointsBaseXYZ[2][0] == doctest::Approx(100.0));
    CHECK(line.pointsBaseXYZ[2][1] == doctest::Approx(5.0));
}

TEST_CASE("Fiberlet crop lookahead preserves lexicographic pruning and state-cap ordering")
{
    TestGraph graph;
    const auto seed = key(0);
    const auto branch = key(10);
    graph.addAnchor(seed, {0, 0, 0});
    graph.addAnchor(branch, {10, 0, 0});
    graph.connect(seed, branch);
    for (std::int64_t index = 0; index < 65; ++index) {
        const auto target = key(100 + index);
        graph.addAnchor(target, {20, static_cast<double>(index) * 0.01, 0});
        graph.connect(branch, target);
    }

    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-100, -100, -100};
    config.maximumBaseXYZ = {1'000, 100, 100};
    config.beamWidth = 1;
    config.lookaheadDistanceBaseVoxels = 100;
    config.maximumAttempts = 1;

    const auto pruned = traceFiberletCrop(graph, {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)}, normals, 1.0, config);
    REQUIRE(pruned.lines.size() == 1);
    REQUIRE(pruned.lines.front().pointsBaseXYZ.size() == 3);
    CHECK(pruned.lines.front().pointsBaseXYZ.back() == cv::Vec3d{20, 0, 0});
    CHECK(pruned.lines.front().positiveTermination == "graph_exhausted");

    config.maximumGeneratedStatesPerStep = 1;
    const auto capped = traceFiberletCrop(graph, {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)}, normals, 1.0, config);
    REQUIRE(capped.lines.size() == 1);
    CHECK(capped.lines.front().pointsBaseXYZ == pruned.lines.front().pointsBaseXYZ);
    CHECK(capped.lines.front().positiveTermination == "graph_exhausted");
}

TEST_CASE("Fiberlet crop tracing computes concurrently and integrates canonically")
{
    TestGraph graph;
    const auto outsideLeft = key(-10);
    const auto left = key(20);
    const auto seed = key(50);
    const auto right = key(80);
    const auto outsideRight = key(110);
    graph.addAnchor(outsideLeft, {-10, 0, 0});
    graph.addAnchor(left, {20, 0, 0});
    graph.addAnchor(seed, {50, 0, 0});
    graph.addAnchor(right, {80, 0, 0});
    graph.addAnchor(outsideRight, {110, 0, 0});
    graph.connect(outsideLeft, left);
    graph.connect(left, seed);
    graph.connect(seed, right);
    graph.connect(right, outsideRight);

    const auto covered = key(51);
    const auto crossing = key(52);
    const auto normalFar = key(53);
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(seed, {50, 0, 0}, {1, 0, 0}, 1.0F),
        anchor(covered, {50, 50, 0}, {1, 0, 0}, 0.8F),
        anchor(crossing, {50, 0, 0}, {0, 1, 0}, 0.7F),
        anchor(normalFar, {50, 0, 21}, {1, 0, 0}, 0.6F),
    };
    ZNormalSampler normals;
    FiberletCropTraceConfig serialConfig;
    serialConfig.minimumBaseXYZ = {0, -100, -100};
    serialConfig.maximumBaseXYZ = {100, 100, 100};
    serialConfig.lookaheadDistanceBaseVoxels = 48;
    serialConfig.parallelThreads = 1;
    const auto serial = traceFiberletCrop(graph, anchors, normals, 1.0, serialConfig);

    graph.concurrentQueries = true;
    graph.observeConcurrency = true;
    graph.queryDelay[seed] = std::chrono::milliseconds(40);
    FiberletCropTraceConfig parallelConfig = serialConfig;
    parallelConfig.parallelThreads = 4;
    const auto parallel = traceFiberletCrop(graph, anchors, normals, 1.0, parallelConfig);
    CHECK(graph.maximumConcurrentQueries.load() >= 2);
    CHECK(parallel.candidateAnchors == serial.candidateAnchors);
    CHECK(parallel.attemptedAnchors == serial.attemptedAnchors);
    CHECK(parallel.coveredAnchors == serial.coveredAnchors);
    CHECK(parallel.noEdgeAnchors == serial.noEdgeAnchors);
    CHECK(parallel.oneSidedLines == serial.oneSidedLines);
    CHECK(parallel.bidirectionalLines == serial.bidirectionalLines);
    REQUIRE(parallel.lines.size() == serial.lines.size());
    for (std::size_t line = 0; line < serial.lines.size(); ++line) {
        CHECK(parallel.lines[line].seed == serial.lines[line].seed);
        CHECK(parallel.lines[line].seedBaseXYZ == serial.lines[line].seedBaseXYZ);
        CHECK(parallel.lines[line].seedPresence == serial.lines[line].seedPresence);
        CHECK(parallel.lines[line].negativeTermination == serial.lines[line].negativeTermination);
        CHECK(parallel.lines[line].positiveTermination == serial.lines[line].positiveTermination);
        CHECK(parallel.lines[line].negativeFiberlets == serial.lines[line].negativeFiberlets);
        CHECK(parallel.lines[line].positiveFiberlets == serial.lines[line].positiveFiberlets);
        CHECK(parallel.lines[line].pointsBaseXYZ == serial.lines[line].pointsBaseXYZ);
    }

    graph.maximumConcurrentQueries = 0;
    parallelConfig.parallelThreads = 0;
    parallelConfig.maximumAttempts = 2;
    const auto defaultWorkers = traceFiberletCrop(graph, anchors, normals, 1.0, parallelConfig);
    CHECK(defaultWorkers.attemptedAnchors == 2);
    CHECK(defaultWorkers.lines.size() == 1);

    graph.concurrentQueries = false;
    graph.maximumConcurrentQueries = 0;
    parallelConfig.parallelThreads = 4;
    const auto unsupportedSource = traceFiberletCrop(graph, anchors, normals, 1.0, parallelConfig);
    CHECK(unsupportedSource.attemptedAnchors == 2);
    CHECK(graph.maximumConcurrentQueries.load() == 1);

    graph.concurrentQueries = true;
    graph.maximumConcurrentQueries = 0;
    parallelConfig.ambiguityRelativeCostMargin = 0.1;
    SerialZNormalSampler serialNormals;
    const auto unsupportedNormals = traceFiberletCrop(
        graph, anchors, serialNormals, 1.0, parallelConfig);
    CHECK(unsupportedNormals.attemptedAnchors == 2);
    CHECK(graph.maximumConcurrentQueries.load() == 1);
}

TEST_CASE("Fiberlet crop tracing optionally stops at prior covered anchor endpoints")
{
    TestGraph graph;
    const auto firstSeed = key(0);
    const auto firstEnd = key(100);
    const auto secondSeed = key(200);
    const auto coveredEndpoint = key(50);
    const auto continuation = key(51);
    graph.addAnchor(firstSeed, {0, 0, 0});
    graph.addAnchor(firstEnd, {100, 0, 0});
    graph.addAnchor(secondSeed, {50, 100, 0});
    graph.addAnchor(coveredEndpoint, {50, 0, 0});
    graph.addAnchor(continuation, {50, -100, 0});
    graph.connect(firstSeed, firstEnd);
    graph.connect(secondSeed, coveredEndpoint);
    graph.connect(coveredEndpoint, continuation);

    const std::vector<FiberletStoredAnchor> anchors{
        anchor(firstSeed, {0, 0, 0}, {1, 0, 0}, 1.0F),
        anchor(secondSeed, {50, 100, 0}, {0, -1, 0}, 0.9F),
        anchor(coveredEndpoint, {50, 0, 0}, {1, 0, 0}, 0.8F),
        anchor(continuation, {50, -100, 0}, {0, 1, 0}, 0.7F),
    };
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-200, -200, -100};
    config.maximumBaseXYZ = {200, 200, 100};
    config.lookaheadDistanceBaseVoxels = 48;
    config.parallelThreads = 1;

    const auto baseline = traceFiberletCrop(
        graph, anchors, normals, 1.0, config);
    REQUIRE(baseline.lines.size() == 2);
    CHECK(baseline.attemptedAnchors == 2);
    CHECK(baseline.coveredAnchorStops == 0);
    const auto baselineSecond = std::find_if(
        baseline.lines.begin(), baseline.lines.end(), [&](const auto& line) {
            return line.seed == secondSeed;
        });
    REQUIRE(baselineSecond != baseline.lines.end());
    CHECK(baselineSecond->positiveTermination == "graph_exhausted");
    CHECK(baselineSecond->positiveFiberlets == 2);
    CHECK(baselineSecond->pathLengthPredictionVoxels == doctest::Approx(200.0));
    CHECK(baselineSecond->totalMetricCost == doctest::Approx(200.0));
    CHECK(baselineSecond->pointsBaseXYZ.back() == cv::Vec3d{50, -100, 0});

    config.stopAtCoveredAnchors = true;
    const auto serial = traceFiberletCrop(
        graph, anchors, normals, 1.0, config);
    REQUIRE(serial.lines.size() == 3);
    CHECK(serial.attemptedAnchors == 3);
    CHECK(serial.coveredAnchorStops == 2);
    const auto stopped = std::find_if(
        serial.lines.begin(), serial.lines.end(), [&](const auto& line) {
            return line.seed == secondSeed;
        });
    REQUIRE(stopped != serial.lines.end());
    CHECK(stopped->positiveTermination == "covered_anchor");
    CHECK(stopped->positiveFiberlets == 1);
    CHECK(stopped->pathLengthPredictionVoxels == doctest::Approx(100.0));
    CHECK(stopped->totalMetricCost == doctest::Approx(100.0));
    REQUIRE(stopped->pointsBaseXYZ.size() == 2);
    CHECK(stopped->pointsBaseXYZ.back() == cv::Vec3d{50, 0, 0});
    CHECK(std::any_of(
        serial.lines.begin(), serial.lines.end(), [&](const auto& line) {
            return line.seed == continuation;
        }));

    graph.concurrentQueries = true;
    graph.queryDelay[firstSeed] = std::chrono::milliseconds(30);
    config.parallelThreads = 4;
    const auto parallel = traceFiberletCrop(
        graph, anchors, normals, 1.0, config);
    CHECK(parallel.attemptedAnchors == serial.attemptedAnchors);
    CHECK(parallel.coveredAnchors == serial.coveredAnchors);
    CHECK(parallel.noEdgeAnchors == serial.noEdgeAnchors);
    CHECK(parallel.oneSidedLines == serial.oneSidedLines);
    CHECK(parallel.bidirectionalLines == serial.bidirectionalLines);
    CHECK(parallel.coveredAnchorStops == serial.coveredAnchorStops);
    REQUIRE(parallel.lines.size() == serial.lines.size());
    for (std::size_t index = 0; index < serial.lines.size(); ++index) {
        CHECK(parallel.lines[index].seed == serial.lines[index].seed);
        CHECK(parallel.lines[index].negativeTermination ==
              serial.lines[index].negativeTermination);
        CHECK(parallel.lines[index].positiveTermination ==
              serial.lines[index].positiveTermination);
        CHECK(parallel.lines[index].negativeFiberlets ==
              serial.lines[index].negativeFiberlets);
        CHECK(parallel.lines[index].positiveFiberlets ==
              serial.lines[index].positiveFiberlets);
        CHECK(parallel.lines[index].totalMetricCost ==
              serial.lines[index].totalMetricCost);
        CHECK(parallel.lines[index].pathLengthPredictionVoxels ==
              serial.lines[index].pathLengthPredictionVoxels);
        CHECK(parallel.lines[index].pointsBaseXYZ ==
              serial.lines[index].pointsBaseXYZ);
    }
}

TEST_CASE("Attempted anchors are not covered-anchor stop barriers")
{
    TestGraph graph;
    const auto attempted = key(50);
    const auto seed = key(200);
    const auto finish = key(51);
    graph.addAnchor(attempted, {50, 0, 0});
    graph.addAnchor(seed, {50, 100, 0});
    graph.addAnchor(finish, {50, -100, 0});
    graph.connect(seed, attempted);
    graph.connect(attempted, finish);

    const std::vector<FiberletStoredAnchor> anchors{
        anchor(attempted, {50, 0, 0}, {1, 0, 0}, 1.0F),
        anchor(seed, {50, 100, 0}, {0, -1, 0}, 0.9F),
    };
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-200, -200, -100};
    config.maximumBaseXYZ = {200, 200, 100};
    config.lookaheadDistanceBaseVoxels = 48;
    config.parallelThreads = 1;
    config.stopAtCoveredAnchors = true;

    const auto result = traceFiberletCrop(
        graph, anchors, normals, 1.0, config);
    REQUIRE(result.lines.size() == 1);
    CHECK(result.attemptedAnchors == 2);
    CHECK(result.noEdgeAnchors == 1);
    CHECK(result.lines.front().seed == seed);
    CHECK(result.lines.front().positiveTermination == "graph_exhausted");
    CHECK(result.lines.front().positiveFiberlets == 2);
    CHECK(result.lines.front().pathLengthPredictionVoxels ==
          doctest::Approx(200.0));
    CHECK(result.lines.front().pointsBaseXYZ.back() == cv::Vec3d{50, -100, 0});
}

TEST_CASE("Fiberlet crop tracing applies an inclusive quality threshold before coverage")
{
    TestGraph graph;
    const auto seed = key(0);
    const auto finish = key(100);
    graph.addAnchor(seed, {0, 0, 0});
    graph.addAnchor(finish, {100, 0, 0});
    graph.connect(seed, finish);
    graph.setCostDensity(seed, finish, 2.0F);

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-200, -200, -100};
    config.maximumBaseXYZ = {200, 200, 100};
    config.lookaheadDistanceBaseVoxels = 48;
    config.maximumAttempts = 1;
    config.maximumAcceptedCostDensity = 1.999;
    ZNormalSampler normals;

    const auto rejected = traceFiberletCrop(
        graph,
        {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)},
        normals,
        1.0,
        config);
    CHECK(rejected.attemptedAnchors == 1);
    CHECK(rejected.qualityRejectedAnchors == 1);
    CHECK(rejected.coveredAnchors == 0);
    CHECK(rejected.lines.empty());

    config.maximumAcceptedCostDensity = 2.0;
    const auto accepted = traceFiberletCrop(
        graph,
        {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)},
        normals,
        1.0,
        config);
    CHECK(accepted.qualityRejectedAnchors == 0);
    REQUIRE(accepted.lines.size() == 1);
    CHECK(accepted.lines.front().totalMetricCost /
              accepted.lines.front().pathLengthPredictionVoxels ==
          doctest::Approx(2.0));

    config.maximumAcceptedCostDensity =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        traceFiberletCrop(
            graph,
            {anchor(seed, {0, 0, 0}, {1, 0, 0}, 1.0F)},
            normals,
            1.0,
            config),
        std::invalid_argument);
}

TEST_CASE("Quality-rejected crop traces leave nearby seeds eligible")
{
    TestGraph graph;
    const auto strongSeed = key(0);
    const auto strongFinish = key(100);
    const auto nearbySeed = key(200);
    const auto nearbyFinish = key(300);
    graph.addAnchor(strongSeed, {0, 0, 0});
    graph.addAnchor(strongFinish, {100, 0, 0});
    graph.addAnchor(nearbySeed, {0, 10, 0});
    graph.addAnchor(nearbyFinish, {100, 10, 0});
    graph.connect(strongSeed, strongFinish);
    graph.connect(nearbySeed, nearbyFinish);
    graph.setCostDensity(strongSeed, strongFinish, 2.0F);
    graph.setCostDensity(nearbySeed, nearbyFinish, 1.0F);

    const std::vector<FiberletStoredAnchor> anchors{
        anchor(strongSeed, {0, 0, 0}, {1, 0, 0}, 1.0F),
        anchor(nearbySeed, {0, 10, 0}, {1, 0, 0}, 0.9F),
    };
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-200, -200, -100};
    config.maximumBaseXYZ = {200, 200, 100};
    config.lookaheadDistanceBaseVoxels = 48;
    config.maximumAcceptedCostDensity = 1.5;
    config.stopAtCoveredAnchors = true;
    config.parallelThreads = 1;
    ZNormalSampler normals;

    const auto rejected = traceFiberletCrop(
        graph, anchors, normals, 1.0, config);
    CHECK(rejected.attemptedAnchors == 2);
    CHECK(rejected.qualityRejectedAnchors == 1);
    CHECK(rejected.coveredAnchorStops == 0);
    REQUIRE(rejected.lines.size() == 1);
    CHECK(rejected.lines.front().seed == nearbySeed);

    config.maximumAcceptedCostDensity = 2.0;
    const auto accepted = traceFiberletCrop(
        graph, anchors, normals, 1.0, config);
    CHECK(accepted.attemptedAnchors == 1);
    CHECK(accepted.qualityRejectedAnchors == 0);
    REQUIRE(accepted.lines.size() == 1);
    CHECK(accepted.lines.front().seed == strongSeed);
}

TEST_CASE("Fiberlet crop tracing reports speculative failures in canonical order")
{
    TestGraph graph;
    const auto first = key(10);
    const auto second = key(20);
    graph.addAnchor(first, {0, 0, 0});
    graph.addAnchor(second, {200, 0, 0});
    graph.concurrentQueries = true;
    graph.queryDelay[first] = std::chrono::milliseconds(30);
    graph.queryFailure[first] = "first canonical failure";
    graph.queryFailure[second] = "later speculative failure";

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-100, -100, -100};
    config.maximumBaseXYZ = {1'000, 100, 100};
    config.parallelThreads = 2;
    ZNormalSampler normals;
    CHECK_THROWS_WITH_AS(traceFiberletCrop(graph, {anchor(first, {0, 0, 0}, {1, 0, 0}, 1.0F), anchor(second, {200, 0, 0}, {1, 0, 0}, 0.9F)}, normals, 1.0, config), doctest::Contains("first canonical failure"), std::runtime_error);
}

TEST_CASE("Fiberlet crop tracing ignores failures beyond a serial fiber limit")
{
    TestGraph graph;
    const auto first = key(10);
    const auto firstTarget = key(11);
    const auto later = key(20);
    graph.addAnchor(first, {0, 0, 0});
    graph.addAnchor(firstTarget, {10, 0, 0});
    graph.addAnchor(later, {200, 0, 0});
    graph.connect(first, firstTarget);
    graph.concurrentQueries = true;
    graph.queryDelay[first] = std::chrono::milliseconds(30);
    graph.queryFailure[later] = "uncommitted speculative failure";

    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-100, -100, -100};
    config.maximumBaseXYZ = {1'000, 100, 100};
    config.parallelThreads = 2;
    config.maximumFibers = 1;
    ZNormalSampler normals;
    const auto result =
        traceFiberletCrop(graph, {anchor(first, {0, 0, 0}, {1, 0, 0}, 1.0F), anchor(later, {200, 0, 0}, {1, 0, 0}, 0.9F)}, normals, 1.0, config);
    CHECK(result.lines.size() == 1);
    CHECK(result.lines.front().seed == first);
    CHECK(result.attemptedAnchors == 1);
    CHECK(result.computedCandidates == 2);
}

TEST_CASE("Owned Fiberlet replay views retain data after the source buffer dies")
{
    const auto points = FiberletReplayRoutePointView::owned({{1, 2, 3}, {4, 5, 6}}, true);
    REQUIRE(points.leased());
    REQUIRE(points.size() == 2);
    CHECK(points.front() == cv::Vec3d{4, 5, 6});
    CHECK(points.back() == cv::Vec3d{1, 2, 3});

    FiberletReplaySourceCostProfile profile;
    profile.segmentLengthsPredictionVoxels = {2.0F, 3.0F};
    profile.segmentCostDensities = {0.25F, 0.5F};
    const auto costs = FiberletReplayCostProfileView::owned(std::move(profile), true);
    REQUIRE(costs.leased());
    CHECK(costs.segmentLengthsPredictionVoxels[0] == 3.0F);
    CHECK(costs.segmentCostDensities[1] == 0.25F);
}

TEST_CASE("Fiberlet crop attempts follow strongest-first deterministic ordering")
{
    TestGraph graph;
    const auto first = key(10);
    const auto second = key(20);
    const auto third = key(30);
    graph.addAnchor(first, {0, 0, 0});
    graph.addAnchor(key(11), {10, 0, 0});
    graph.addAnchor(second, {200, 0, 0});
    graph.addAnchor(key(21), {210, 0, 0});
    graph.addAnchor(third, {400, 0, 0});
    graph.addAnchor(key(31), {410, 0, 0});
    graph.connect(first, key(11));
    graph.connect(second, key(21));
    graph.connect(third, key(31));
    const std::vector<FiberletStoredAnchor> anchors{
        anchor(third, {400, 0, 0}, {1, 0, 0}, 0.5F),
        anchor(second, {200, 0, 0}, {1, 0, 0}, 0.9F),
        anchor(first, {0, 0, 0}, {1, 0, 0}, 0.9F),
    };
    ZNormalSampler normals;
    FiberletCropTraceConfig config;
    config.minimumBaseXYZ = {-100, -100, -100};
    config.maximumBaseXYZ = {1'000, 100, 100};
    config.maximumAttempts = 2;

    const auto result = traceFiberletCrop(graph, anchors, normals, 1.0, config);
    REQUIRE(result.lines.size() == 2);
    CHECK(result.attemptedAnchors == 2);
    CHECK(result.lines[0].seed == first);
    CHECK(result.lines[1].seed == second);

    config.maximumAttempts = 10;
    config.maximumFibers = 1;
    const auto acceptedLimited = traceFiberletCrop(graph, anchors, normals, 1.0, config);
    CHECK(acceptedLimited.attemptedAnchors == 1);
    CHECK(acceptedLimited.lines.size() == 1);

    TestGraph failureGraph;
    failureGraph.addAnchor(first, {0, 0, 0});
    failureGraph.addAnchor(second, {200, 0, 0});
    failureGraph.addAnchor(key(21), {210, 0, 0});
    failureGraph.connect(second, key(21));
    config.maximumAttempts = 1;
    config.maximumFibers = 0;
    const auto failureLimited =
        traceFiberletCrop(failureGraph, {anchor(second, {200, 0, 0}, {1, 0, 0}, 0.5F), anchor(first, {0, 0, 0}, {1, 0, 0}, 0.9F)}, normals, 1.0, config);
    CHECK(failureLimited.attemptedAnchors == 1);
    CHECK(failureLimited.noEdgeAnchors == 1);
    CHECK(failureLimited.lines.empty());
}
