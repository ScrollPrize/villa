#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberSkeleton.hpp"
#include "vc/fiber_tracer/FiberSkeletonChoice.hpp"
#include "vc/fiber_tracer/FiberSkeletonRefine.hpp"
#include "vc/fiber_tracer/FiberSkeletonRegions.hpp"
#include "vc/fiber_tracer/FiberSkeletonReplay.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>

namespace
{

using namespace vc::fiber_tracer;

std::array<std::uint8_t, 27> neighborhoodOf(std::initializer_list<int> set)
{
    std::array<std::uint8_t, 27> neighborhood{};
    for (const int slot : set)
        neighborhood[static_cast<std::size_t>(slot)] = 1;
    return neighborhood;
}

constexpr std::size_t slotOf(int dz, int dy, int dx)
{
    return static_cast<std::size_t>((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1));
}

FiberSkeletonCrop makeCrop(
    std::size_t sizeZ, std::size_t sizeY, std::size_t sizeX)
{
    FiberSkeletonCrop crop;
    crop.shapeZYX = {sizeZ, sizeY, sizeX};
    crop.predictionToBaseScale = 1.0;
    crop.presence.assign(crop.voxelCount(), 0);
    return crop;
}

// Straight fiber in the z = centerZ plane, given as a function of x.
struct StraightFiber {
    cv::Vec3d start;
    cv::Vec3d end;

    [[nodiscard]] cv::Vec3d direction() const
    {
        const cv::Vec3d delta = end - start;
        return delta / cv::norm(delta);
    }

    [[nodiscard]] double distance(const cv::Vec3d& point) const
    {
        const cv::Vec3d delta = end - start;
        const double t = std::clamp(
            (point - start).dot(delta) / delta.dot(delta), 0.0, 1.0);
        return cv::norm(point - (start + delta * t));
    }
};

// Analytic presence/direction/normal field around a set of straight fibers.
class SyntheticFields final : public FiberSkeletonFieldProvider
{
public:
    SyntheticFields(FiberSkeletonCrop crop, std::vector<StraightFiber> fibers,
                    double radius)
        : crop_(std::move(crop)), fibers_(std::move(fibers)), radius_(radius)
    {
        for (std::size_t flat = 0; flat < crop_.voxelCount(); ++flat) {
            const auto sample = evaluate(crop_.baseXYZ(flat));
            crop_.presence[flat] = static_cast<std::uint8_t>(
                std::lround(std::clamp(sample.presence, 0.0, 1.0) * 255.0));
        }
    }

    [[nodiscard]] FiberSkeletonFieldSample evaluate(
        const cv::Vec3d& point) const
    {
        FiberSkeletonFieldSample sample;
        sample.normal = {0.0, 0.0, 1.0};
        sample.normalValid = true;
        double best = std::numeric_limits<double>::infinity();
        for (const auto& fiber : fibers_) {
            const double distance = fiber.distance(point);
            if (distance < best) {
                best = distance;
                sample.direction = fiber.direction();
            }
        }
        if (best <= radius_) {
            sample.presence = std::exp(-(best * best) / (2.0 * radius_ * radius_));
            sample.valid = true;
        } else {
            sample.direction = {0.0, 0.0, 0.0};
        }
        return sample;
    }

    void loadCorners(
        const std::vector<cv::Vec3d>& pointsBaseXYZ,
        const std::vector<cv::Vec3d>& referenceDirections,
        std::vector<FiberSkeletonCornerBlock>& blocks) override
    {
        REQUIRE(pointsBaseXYZ.size() == referenceDirections.size());
        blocks.assign(pointsBaseXYZ.size(), {});
        for (std::size_t index = 0; index < pointsBaseXYZ.size(); ++index) {
            auto& block = blocks[index];
            block.spacingBase = crop_.predictionToBaseScale;
            block.referenceDirection = referenceDirections[index];
            cv::Vec3d origin;
            for (int axis = 0; axis < 3; ++axis)
                origin[axis] = std::floor(pointsBaseXYZ[index][axis]);
            block.cellOriginBaseXYZ = origin;
            block.inside = true;
            for (std::size_t corner = 0; corner < 8; ++corner) {
                const cv::Vec3d position{
                    origin[0] + ((corner & 1U) != 0 ? 1.0 : 0.0),
                    origin[1] + ((corner & 2U) != 0 ? 1.0 : 0.0),
                    origin[2] + ((corner & 4U) != 0 ? 1.0 : 0.0)};
                const auto sample = evaluate(position);
                if (sample.valid) {
                    block.presence[corner] = sample.presence;
                    block.direction[corner] =
                        sample.direction.dot(referenceDirections[index]) < 0.0
                        ? -sample.direction
                        : sample.direction;
                    block.valid[corner] = 1;
                }
                block.normal[corner] = sample.normal;
                block.normalValid[corner] = 1;
            }
        }
    }

    void loadVoxels(
        const std::vector<std::size_t>& flatCropIndices,
        std::vector<FiberSkeletonFieldSample>& samples) override
    {
        samples.clear();
        samples.reserve(flatCropIndices.size());
        for (const auto flat : flatCropIndices) {
            samples.push_back(flat < crop_.voxelCount()
                                  ? evaluate(crop_.baseXYZ(flat))
                                  : FiberSkeletonFieldSample{});
        }
    }

    [[nodiscard]] const FiberSkeletonCrop& crop() const noexcept override
    {
        return crop_;
    }
    [[nodiscard]] double predictionToBaseScale() const noexcept override
    {
        return crop_.predictionToBaseScale;
    }

private:
    FiberSkeletonCrop crop_;
    std::vector<StraightFiber> fibers_;
    double radius_ = 2.0;
};

class ConstantNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override
    {
        return true;
    }
    [[nodiscard]] vc::lasagna::NormalSample sampleNormal(
        const cv::Vec3d&) const override
    {
        return {{0.0, 0.0, 1.0}, true, {}};
    }
};

void paintLine(
    std::vector<std::uint8_t>& mask, const FiberSkeletonCrop& crop,
    std::size_t z, std::size_t y, std::size_t fromX, std::size_t toX)
{
    for (std::size_t x = fromX; x <= toX; ++x)
        mask[crop.flatIndex(z, y, x)] = 1;
}

std::vector<StraightFiber> crossingFibers()
{
    return {StraightFiber{{30.0, 10.0, 20.0}, {90.0, 50.0, 20.0}},
            StraightFiber{{30.0, 50.0, 20.0}, {90.0, 10.0, 20.0}}};
}

}  // namespace

TEST_CASE("simple point test recognises the standard cases")
{
    // Two opposite neighbors are two separate local components, so the voxel
    // between them is not simple: removing it would break the line.
    CHECK_FALSE(fiberSkeletonSimplePoint(
        neighborhoodOf({static_cast<int>(slotOf(0, 0, -1)),
                        static_cast<int>(slotOf(0, 0, 1))})));
    // Two mutually adjacent neighbors form one component, so the voxel is
    // simple and a thick corner can be thinned away.
    CHECK(fiberSkeletonSimplePoint(
        neighborhoodOf({static_cast<int>(slotOf(0, 0, -1)),
                        static_cast<int>(slotOf(0, -1, -1))})));
    // An isolated voxel is never simple: deleting it removes a component.
    CHECK_FALSE(fiberSkeletonSimplePoint(neighborhoodOf({})));
    // A single neighbor is simple, which is why endpoints need their own rule.
    CHECK(fiberSkeletonSimplePoint(
        neighborhoodOf({static_cast<int>(slotOf(0, 0, -1))})));
}

TEST_CASE("thinning reduces a solid bar to one chain")
{
    auto crop = makeCrop(11, 11, 40);
    std::vector<std::uint8_t> mask(crop.voxelCount(), 0);
    for (std::size_t z = 4; z <= 6; ++z) {
        for (std::size_t y = 4; y <= 6; ++y)
            paintLine(mask, crop, z, y, 5, 34);
    }
    const auto solid = std::count(mask.begin(), mask.end(),
                                  static_cast<std::uint8_t>(1));
    thinFiberSkeletonMask(crop.shapeZYX, mask);
    const auto thinned = std::count(mask.begin(), mask.end(),
                                    static_cast<std::uint8_t>(1));
    CHECK(thinned < solid);
    CHECK(thinned >= 28);
    CHECK(thinned <= 40);

    std::size_t components = 0;
    const auto labels =
        labelFiberSkeletonComponents(crop.shapeZYX, mask, components);
    CHECK(labels.size() == crop.voxelCount());
    CHECK(components == 1);

    const auto graph = buildFiberSkeletonGraph(crop, mask);
    CHECK(graph.statistics.terminalNodes == 2);
    CHECK(graph.statistics.junctionNodes == 0);
    CHECK(graph.statistics.cycleRank == 0);
}

TEST_CASE("thinning preserves the cycle rank of a ring")
{
    auto crop = makeCrop(5, 30, 30);
    std::vector<std::uint8_t> mask(crop.voxelCount(), 0);
    for (std::size_t y = 8; y <= 21; ++y) {
        for (std::size_t x = 8; x <= 21; ++x) {
            const double dy = static_cast<double>(y) - 14.5;
            const double dx = static_cast<double>(x) - 14.5;
            const double radius = std::sqrt(dy * dy + dx * dx);
            if (radius >= 4.0 && radius <= 6.5)
                mask[crop.flatIndex(2, y, x)] = 1;
        }
    }
    thinFiberSkeletonMask(crop.shapeZYX, mask);
    std::size_t components = 0;
    const auto labels =
        labelFiberSkeletonComponents(crop.shapeZYX, mask, components);
    CHECK(labels.size() == crop.voxelCount());
    CHECK(components == 1);
    const auto graph = buildFiberSkeletonGraph(crop, mask);
    CHECK(graph.statistics.cycleRank == 1);
}

TEST_CASE("thinning keeps the junction of a crossing")
{
    auto crop = makeCrop(5, 41, 41);
    std::vector<std::uint8_t> mask(crop.voxelCount(), 0);
    for (std::size_t offset = 0; offset < 31; ++offset) {
        mask[crop.flatIndex(2, 20, 5 + offset)] = 1;
        mask[crop.flatIndex(2, 5 + offset, 20)] = 1;
    }
    thinFiberSkeletonMask(crop.shapeZYX, mask);
    auto graph = buildFiberSkeletonGraph(crop, mask);
    CHECK(graph.statistics.terminalNodes == 4);
    CHECK(graph.statistics.junctionNodes >= 1);
    // Junction voxels are no longer contracted, so the crossing's mutually
    // adjacent critical voxels are several nodes joined by short edges. That
    // blob carries cycles of its own; what must hold is that grouping collects
    // it into exactly one ambiguous region with the four arms leaving it.
    reduceFiberSkeletonGraph(graph, 16.0, 2.0);
    CHECK(graph.statistics.terminalNodes == 4);
    std::set<std::size_t> groups;
    for (const auto& node : graph.nodes) {
        if (node.kind == FiberSkeletonNodeKind::Junction) {
            CHECK(node.regionGroup != kFiberSkeletonNoGroup);
            groups.insert(node.regionGroup);
        }
    }
    CHECK(groups.size() == 1);
    CHECK(graph.statistics.regionGroups == 1);
}

TEST_CASE("graph reduction prunes spurs and suppresses degree two nodes")
{
    auto crop = makeCrop(5, 21, 61);
    std::vector<std::uint8_t> mask(crop.voxelCount(), 0);
    paintLine(mask, crop, 2, 10, 5, 55);
    // A short spur that must disappear, and a long arm that must not.
    for (std::size_t offset = 1; offset <= 4; ++offset)
        mask[crop.flatIndex(2, 10 + offset, 20)] = 1;
    for (std::size_t offset = 1; offset <= 9; ++offset)
        mask[crop.flatIndex(2, 10 - offset, 40)] = 1;

    auto graph = buildFiberSkeletonGraph(crop, mask);
    const std::size_t rawTerminals = graph.statistics.terminalNodes;
    CHECK(rawTerminals == 4);
    reduceFiberSkeletonGraph(graph, 4.0, 6.0);
    CHECK(graph.statistics.prunedSpurs >= 1);
    CHECK(graph.statistics.terminalNodes == 3);
    for (const auto& node : graph.nodes)
        CHECK(node.degree != 2);
}

TEST_CASE("polyline helpers preserve endpoints and length")
{
    const std::vector<cv::Vec3d> points{
        {0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {10.0, 10.0, 0.0}};
    CHECK(fiberPolylineLength(points) == doctest::Approx(20.0));

    const auto resampled = resampleFiberPolyline(points, 4.0);
    CHECK(resampled.size() == 6);
    CHECK(cv::norm(resampled.front() - points.front()) ==
          doctest::Approx(0.0));
    CHECK(cv::norm(resampled.back() - points.back()) == doctest::Approx(0.0));
    // Uniform resampling cuts the corner, so the length may only shrink.
    CHECK(fiberPolylineLength(resampled) <= 20.0 + 1.0e-9);
    CHECK(fiberPolylineLength(resampled) > 18.0);

    const auto slice = fiberPolylineSlice(points, 5.0, 15.0);
    CHECK(fiberPolylineLength(slice) == doctest::Approx(10.0));
    CHECK(slice.front()[0] == doctest::Approx(5.0));
    CHECK(slice.back()[1] == doctest::Approx(5.0));

    CHECK(fiberPolylinePointAtArc(points, -1.0)[0] == doctest::Approx(0.0));
    CHECK(fiberPolylinePointAtArc(points, 100.0)[1] == doctest::Approx(10.0));
}

TEST_CASE("matching enumeration produces every partial matching")
{
    CHECK(enumerateFiberSkeletonMatchings(0).size() == 1);
    CHECK(enumerateFiberSkeletonMatchings(1).size() == 1);
    CHECK(enumerateFiberSkeletonMatchings(2).size() == 2);
    CHECK(enumerateFiberSkeletonMatchings(3).size() == 4);
    CHECK(enumerateFiberSkeletonMatchings(4).size() == 10);
    // Involution numbers: 26, 76, 232, 764 for five to eight ports.
    CHECK(enumerateFiberSkeletonMatchings(5).size() == 26);
    CHECK(enumerateFiberSkeletonMatchings(6).size() == 76);
    CHECK(enumerateFiberSkeletonMatchings(7).size() == 232);
    CHECK(enumerateFiberSkeletonMatchings(8).size() == 764);

    for (const auto& partner : enumerateFiberSkeletonMatchings(5)) {
        for (std::size_t port = 0; port < partner.size(); ++port) {
            if (partner[port] == kFiberSkeletonNoIndex)
                continue;
            CHECK(partner[partner[port]] == port);
            CHECK(partner[port] != port);
        }
    }
}

TEST_CASE("refinement pulls a displaced initialization back onto the fiber")
{
    auto crop = makeCrop(41, 41, 81);
    SyntheticFields fields(
        crop, {StraightFiber{{5.0, 20.0, 20.0}, {75.0, 20.0, 20.0}}}, 3.0);

    std::vector<cv::Vec3d> initial;
    for (std::size_t index = 0; index <= 14; ++index) {
        const double x = 10.0 + static_cast<double>(index) * 4.0;
        // Endpoints sit on the fiber; the interior is displaced laterally.
        const double offset = (index == 0 || index == 14) ? 0.0 : 2.5;
        initial.push_back({x, 20.0 + offset, 20.0});
    }

    FiberSkeletonRefineConfig config;
    config.startTangent = cv::Vec3d{1.0, 0.0, 0.0};
    config.endTangent = cv::Vec3d{1.0, 0.0, 0.0};
    const auto result = refineFixedEndpointPolyline(initial, fields, config);

    REQUIRE(result.pointsBaseXYZ.size() == initial.size());
    CHECK(cv::norm(result.pointsBaseXYZ.front() - initial.front()) ==
          doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(cv::norm(result.pointsBaseXYZ.back() - initial.back()) ==
          doctest::Approx(0.0).epsilon(1.0e-9));
    CHECK(result.finalCost.total() <= result.initialCost.total());

    double worst = 0.0;
    for (const auto& point : result.pointsBaseXYZ)
        worst = std::max(worst, std::abs(point[1] - 20.0));
    CHECK(worst < 1.5);

    // Spacing stays uniform and the length does not run away.
    CHECK(result.lengthBase ==
          doctest::Approx(result.initialLengthBase).epsilon(0.25));
    double minimumStep = std::numeric_limits<double>::infinity();
    double maximumStep = 0.0;
    for (std::size_t index = 1; index < result.pointsBaseXYZ.size(); ++index) {
        const double step = cv::norm(
            result.pointsBaseXYZ[index] - result.pointsBaseXYZ[index - 1]);
        minimumStep = std::min(minimumStep, step);
        maximumStep = std::max(maximumStep, step);
    }
    CHECK(maximumStep / minimumStep < 2.0);
}

TEST_CASE("skeleton model routes a crossing straight through")
{
    auto crop = makeCrop(41, 61, 121);
    SyntheticFields fields(crop, crossingFibers(), 2.5);

    FiberSkeletonTraceConfig config;
    config.thresholdU8 = 128;
    config.minComponentVoxels = 4;
    config.junctionMergeMaxBase = 12.0;
    config.spurPruneMaxBase = 6.0;
    config.entryOffsetBase = 12.0;
    config.stepBase = 4.0;
    config.nominalStepBase = 12.0;

    const auto model = buildFiberSkeletonModel(fields, config);
    CHECK(model.graph.statistics.terminalNodes == 4);
    REQUIRE(model.topology.regions.size() == 1);
    CHECK(model.topology.regions.front().portIds.size() == 4);
    REQUIRE(model.choices.size() == 1);

    const auto& choice = model.choices.front();
    CHECK_FALSE(choice.portLimitExceeded);
    REQUIRE(choice.entryConnections.size() == choice.items.size());
    // Every sub-port has a continuation, and it is the straight one: the
    // connection a fibre takes when it arrives joins ports whose tangents
    // oppose each other, not a turn.
    const auto& region = model.topology.regions.front();
    for (std::size_t item = 0; item < choice.items.size(); ++item) {
        const auto entry = choice.entryConnections[item];
        REQUIRE(entry != kFiberSkeletonNoIndex);
        const auto& connection = choice.connections[entry];
        const auto& first = model.topology.ports[region.portIds[connection.portIds[0]]];
        const auto& second = model.topology.ports[region.portIds[connection.portIds[1]]];
        CHECK(first.tangentXYZ.dot(second.tangentXYZ) < -0.5);
    }
}

TEST_CASE("skeleton replay traces a reference and is deterministic")
{
    auto crop = makeCrop(41, 61, 121);
    SyntheticFields fields(crop, crossingFibers(), 2.5);

    FiberSkeletonTraceConfig config;
    config.thresholdU8 = 128;
    config.minComponentVoxels = 4;
    config.junctionMergeMaxBase = 12.0;
    config.spurPruneMaxBase = 6.0;
    config.entryOffsetBase = 12.0;
    config.stepBase = 4.0;
    config.nominalStepBase = 12.0;

    const auto model = buildFiberSkeletonModel(fields, config);
    const ConstantNormalSampler normals;

    FiberSkeletonReplayRequest request;
    const StraightFiber reference = crossingFibers().front();
    // Stop just inside the painted fiber: thinning does not reach the very tip,
    // and a ground-truth piece does not end exactly on a skeleton terminal.
    for (std::size_t index = 0; index <= 50; ++index) {
        const double t = 0.05 + 0.90 * static_cast<double>(index) / 50.0;
        request.referencePointsBase.push_back(
            reference.start + (reference.end - reference.start) * t);
    }
    request.errorThresholdBaseVoxels = 8.0;
    request.tangentialThresholdFactor = 2.0;
    request.excursionGraceBaseVoxels = 16.0;

    const auto result =
        traceFiberSkeletonReplay(model, request, normals, 1.0);
    CHECK(result.replay.completedReferenceArcBase ==
          doctest::Approx(result.replay.referenceEndArcBase));
    CHECK(result.unseededSegments == 0);
    CHECK(result.replay.failures.empty());

    // Geometry, not just completion: every traced point must sit on the fiber
    // it is following, which is what recentring onto the prediction buys.
    double worst = 0.0;
    for (const auto& segment : result.replay.segments) {
        for (const auto& point : segment.tracePointsBase)
            worst = std::max(worst, reference.distance(point));
    }
    CHECK(worst < 3.0);
    REQUIRE_FALSE(result.replay.segments.empty());
    CHECK(result.replay.segments.front().tracePointsBase.size() > 5);
    CHECK(result.replay.segments.front().cumulativeLosses.size() ==
          result.replay.segments.front().tracePointsBase.size());

    const auto repeat = traceFiberSkeletonReplay(model, request, normals, 1.0);
    REQUIRE(repeat.replay.segments.size() == result.replay.segments.size());
    for (std::size_t segment = 0; segment < repeat.replay.segments.size();
         ++segment) {
        const auto& left = result.replay.segments[segment].tracePointsBase;
        const auto& right = repeat.replay.segments[segment].tracePointsBase;
        REQUIRE(left.size() == right.size());
        for (std::size_t index = 0; index < left.size(); ++index)
            CHECK(cv::norm(left[index] - right[index]) == doctest::Approx(0.0));
    }

    const auto first = fiberSkeletonModelJson(model, fields.crop(), &result);
    const auto second = fiberSkeletonModelJson(model, fields.crop(), &repeat);
    CHECK(first["choices"].dump() == second["choices"].dump());
    CHECK(first["ports"].dump() == second["ports"].dump());
}

TEST_CASE("skeleton replay reports an unusable seed instead of crediting it")
{
    auto crop = makeCrop(41, 61, 121);
    SyntheticFields fields(crop, crossingFibers(), 2.5);

    FiberSkeletonTraceConfig config;
    config.thresholdU8 = 128;
    config.minComponentVoxels = 4;
    config.entryOffsetBase = 12.0;
    config.stepBase = 4.0;
    config.nominalStepBase = 12.0;

    const auto model = buildFiberSkeletonModel(fields, config);
    const ConstantNormalSampler normals;

    // A reference far away from every fiber cannot be seeded.
    FiberSkeletonReplayRequest request;
    for (std::size_t index = 0; index <= 20; ++index) {
        request.referencePointsBase.push_back(
            {10.0 + static_cast<double>(index) * 5.0, 58.0, 38.0});
    }
    request.errorThresholdBaseVoxels = 2.0;
    request.tangentialThresholdFactor = 2.0;

    const auto result = traceFiberSkeletonReplay(model, request, normals, 1.0);
    CHECK(result.unseededSegments > 0);
    CHECK_FALSE(result.replay.failures.empty());
    CHECK(result.replay.failures.front().reason ==
          std::string(kFiberSkeletonNoSeed));
    for (const auto& segment : result.segments)
        CHECK_FALSE(segment.seeded);
}

TEST_CASE("skeleton artifacts are written and reloadable")
{
    auto crop = makeCrop(41, 61, 121);
    SyntheticFields fields(crop, crossingFibers(), 2.5);

    FiberSkeletonTraceConfig config;
    config.thresholdU8 = 128;
    config.minComponentVoxels = 4;
    config.entryOffsetBase = 12.0;
    config.stepBase = 4.0;
    config.nominalStepBase = 12.0;

    const auto model = buildFiberSkeletonModel(fields, config);
    const auto directory = std::filesystem::temp_directory_path() /
        "vc_skeleton_artifacts_test";
    std::filesystem::remove_all(directory);
    writeFiberSkeletonArtifacts(directory, model, fields.crop(), nullptr);

    REQUIRE(std::filesystem::exists(directory / "skeleton.json"));
    REQUIRE(std::filesystem::exists(directory / "skeleton_graph.obj"));
    std::ifstream stream(directory / "skeleton.json");
    nlohmann::json document;
    stream >> document;
    CHECK(document.at("format") == "vc_skeleton_trace");
    CHECK(document.at("version") == 1);
    CHECK(document.at("statistics").at("regions") ==
          model.topology.regions.size());
    CHECK(document.at("config").at("threshold_u8") == config.thresholdU8);
    std::filesystem::remove_all(directory);
}

TEST_CASE("port candidates cover both fibres of a merged section")
{
    // Two fibres side by side in one sheet, touching: the section is one
    // ribbon whose extremes are the two fibres and whose centre is the seam.
    auto crop = makeCrop(15, 40, 60);
    for (std::size_t z = 5; z <= 7; ++z) {
        for (std::size_t x = 0; x < crop.shapeZYX[2]; ++x) {
            for (std::size_t y = 10; y <= 25; ++y)
                crop.presence[crop.flatIndex(z, y, x)] = 255;
        }
    }
    const cv::Vec3d point{30.0, 17.5, 6.0};
    const cv::Vec3d tangent{1.0, 0.0, 0.0};
    const auto section = fiberSkeletonSectionAt(crop, 128, point, tangent, 24.0);
    REQUIRE(section.valid);
    CHECK(section.longMinBase < -6.0);
    CHECK(section.longMaxBase > 6.0);
    CHECK(std::abs(section.shortAxisXYZ[2]) > 0.9);

    // The extremes sit on the mask's edge; the local centroid pulls each one
    // inward, onto its own fibre, and never across the seam.
    const cv::Vec3d low = section.centreBaseXYZ + section.longAxisXYZ * section.longMinBase;
    const cv::Vec3d high = section.centreBaseXYZ + section.longAxisXYZ * section.longMaxBase;
    const auto lowRefined = refineFiberSkeletonPointInPlane(crop, 128, low, tangent, 4.0, 8);
    const auto highRefined = refineFiberSkeletonPointInPlane(crop, 128, high, tangent, 4.0, 8);
    CHECK(std::abs(lowRefined[1] - low[1]) > 0.5);
    CHECK(std::abs(highRefined[1] - high[1]) > 0.5);
    CHECK((lowRefined[1] < 17.5) == (low[1] < 17.5));
    CHECK((highRefined[1] < 17.5) == (high[1] < 17.5));
    CHECK(std::abs(lowRefined[1] - highRefined[1]) > 6.0);
}

TEST_CASE("path search follows presence and direction round a bend")
{
    // A bent fibre: the straight chord between its ends leaves the mask, so
    // the initial line must come from the search, not from the chord.
    auto crop = makeCrop(15, 40, 60);
    std::vector<StraightFiber> fibers{
        {{0.0, 10.0, 6.0}, {30.0, 20.0, 6.0}},
        {{30.0, 20.0, 6.0}, {59.0, 10.0, 6.0}}};
    SyntheticFields fields(std::move(crop), fibers, 3.0);
    std::vector<cv::Vec3d> corridor;
    for (double x = 5.0; x <= 55.0; x += 5.0) {
        const double y = x < 30.0 ? 10.0 + x / 3.0 : 20.0 - (x - 30.0) / 3.0;
        corridor.emplace_back(x, y, 6.0);
    }
    FiberSkeletonPathSearchConfig config;
    config.corridorRadiusBase = 8.0;
    config.thresholdU8 = 128;
    const cv::Vec3d start{5.0, 11.7, 6.0};
    const cv::Vec3d end{55.0, 11.7, 6.0};
    std::vector<cv::Vec3d> path;
    REQUIRE(fiberSkeletonSearchPath(fields, config, corridor, start, end, path));
    REQUIRE(path.size() >= 3);
    CHECK(path.front() == start);
    CHECK(path.back() == end);
    const auto& mask = fields.crop();
    double topY = 0.0;
    for (std::size_t index = 1; index + 1 < path.size(); ++index) {
        const auto& point = path[index];
        const auto z = static_cast<std::size_t>(std::lround(point[2]));
        const auto y = static_cast<std::size_t>(std::lround(point[1]));
        const auto x = static_cast<std::size_t>(std::lround(point[0]));
        CHECK(mask.presence[mask.flatIndex(z, y, x)] >= 128);
        topY = std::max(topY, point[1]);
    }
    // It went round the bend rather than through the gap.
    CHECK(topY > 15.0);

    // No way through: the mask is cut.
    auto cut = makeCrop(15, 40, 60);
    SyntheticFields cutFields(std::move(cut), fibers, 3.0);
    auto& cutMask = const_cast<FiberSkeletonCrop&>(cutFields.crop());
    for (std::size_t z = 0; z < cutMask.shapeZYX[0]; ++z)
        for (std::size_t y = 0; y < cutMask.shapeZYX[1]; ++y)
            cutMask.presence[cutMask.flatIndex(z, y, 30)] = 0;
    CHECK_FALSE(fiberSkeletonSearchPath(cutFields, config, corridor, start, end, path));
}
