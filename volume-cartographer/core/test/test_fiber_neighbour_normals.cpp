#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberNeighbourNormals.hpp"

#include <algorithm>
#include <filesystem>
#include <vector>

using namespace vc::fiber_tracer;

namespace
{

// A straight fiber along x at (y, z), sampled every 16 voxels, with one normal.
std::vector<FiberStripSample> straight(const double y, const double z, const cv::Vec3d& normal, const double length = 640.)
{
    std::vector<FiberStripSample> samples;
    for (double x = 0.; x <= length; x += 16.) samples.push_back({x, {x, y, z}, normal, {0., 0., 0.}});
    return samples;
}

const FiberNeighbourVariant kIsotropic{GapFalloff::Gaussian, FiberNeighbourFrame::Isotropic, {64., 64., 64.}, {64., 64., 64.}, 192.};

}  // namespace

TEST_CASE("neighbours state their signed normal at a foreign fiber")
{
    // Two fibers beside the query; the second is stored facing down and signed -1.
    const FiberNeighbourField field({straight(-40., 0., {0., 0., 1.}), straight(40., 0., {0., 0., -1.})}, {1, -1}, 192.);
    CHECK(field.signedSampleCount() == field.sampleCount());
    const auto query = straight(0., 0., {0., 0., 0.});
    FiberNeighbourCounts counts;
    const auto estimates = field.infer(query, std::span<const FiberNeighbourVariant>(&kIsotropic, 1), -1, nullptr, &counts);
    REQUIRE(estimates.size() == query.size());
    for (const auto& e : estimates) {
        CHECK(e.count > 0.);
        CHECK(e.axis[2] == doctest::Approx(1.));
        CHECK(e.margin == doctest::Approx(1.));
        CHECK(e.effectiveFibers == doctest::Approx(2.));
    }
}

TEST_CASE("an unsigned fiber, the query itself and excluded fibers are no source")
{
    const FiberNeighbourField field({straight(0., 0., {0., 0., 1.}), straight(40., 0., {0., 1., 0.}), straight(-40., 0., {0., 0., 1.})},
                                    {1, 1, 0}, 192.);
    CHECK(field.signedSampleCount() == field.fiber(0).size() + field.fiber(1).size());
    const std::span<const FiberNeighbourVariant> variants(&kIsotropic, 1);
    // Fiber 0 asked as itself: only fiber 1 answers (fiber 2 is unsigned).
    const auto own = field.infer(field.fiber(0), variants, 0);
    CHECK(own.front().effectiveFibers == doctest::Approx(1.));
    CHECK(std::abs(own.front().axis[1]) == doctest::Approx(1.));
    const std::set<std::uint32_t> excluded{1};
    const auto none = field.infer(field.fiber(0), variants, 0, &excluded);
    CHECK(none.front().count == 0.);
    CHECK(cv::norm(none.front().axis) == 0.);
}

TEST_CASE("the fiber frame measures a crossing fiber along both fibers")
{
    // A fiber along y crossing the query 30 voxels above it at x = 320.
    std::vector<FiberStripSample> crossing;
    for (double y = -320.; y <= 320.; y += 16.) crossing.push_back({y + 320., {320., y, 30.}, {0., 0., 1.}, {0., 0., 0.}});
    const FiberNeighbourField field({crossing}, {1}, 384.);
    const auto query = straight(0., 0., {0., 0., 0.});
    const FiberNeighbourVariant variant{.estimatedParallelFrame = true};
    FiberNeighbourCounts counts;
    const auto estimates = field.infer(query, std::span<const FiberNeighbourVariant>(&variant, 1), -1, nullptr, &counts);
    CHECK(counts.crossingPairs > 0);
    CHECK(counts.parallelPairs == 0);
    // Strongest at the crossing, fading with the own arc to it.
    const auto& at = estimates[20];  // x = 320
    const auto& away = estimates[4];  // x = 64
    CHECK(at.axis[2] == doctest::Approx(1.));
    CHECK(at.weight > away.weight);
}

TEST_CASE("the field finds the crossings of a fiber and leaves parallel fibers out")
{
    // One fiber along y crossing the query 30 voxels above it at x = 320, one running beside it.
    std::vector<FiberStripSample> crossing;
    for (double y = -320.; y <= 320.; y += 16.) crossing.push_back({y + 320., {320., y, 30.}, {0., 0., 1.}, {0., 0., 0.}});
    const FiberNeighbourField field({crossing, straight(40., 0., {0., 0., 1.})}, {1, 1}, 384.);
    const auto query = straight(0., 0., {0., 0., 0.});
    const auto crossings = field.crossings(query);
    REQUIRE(crossings.size() == 1);
    CHECK(crossings.front().arc == doctest::Approx(320.));
    CHECK(crossings.front().gap == doctest::Approx(30.));
    CHECK(crossings.front().offset[2] == doctest::Approx(30.));
    // Own tangent x partner tangent: x cross y.
    CHECK(std::abs(crossings.front().axis[2]) == doctest::Approx(1.));
    CHECK(crossings.front().ownTangent[0] == doctest::Approx(1.));

    // The crossing axis reaches the chain nodes around it, sign-free.
    std::vector<cv::Vec3d> line;
    for (const auto& sample : query) line.push_back(sample.point);
    const auto chain = fiberStripChain(line);
    std::vector<FiberStripAxisTarget> targets(chain.points.size());
    FiberCrossingTargetCounts counts;
    setFiberCrossingTargets(chain, crossings, FiberCrossingTargetOptions{}, targets, &counts);
    CHECK(counts.used == 1);
    CHECK(counts.crossedNodes > 0);
    const auto middle = std::size_t(std::lower_bound(chain.arc.begin(), chain.arc.end(), 320.) - chain.arc.begin());
    CHECK(std::abs(targets[middle].crossing[2]) == doctest::Approx(1.));
    CHECK(targets[middle].crossingWeight > 0.);
    CHECK(targets.front().crossingWeight == 0.);
}

TEST_CASE("inferred normals become signed chain targets")
{
    std::vector<cv::Vec3d> line;
    for (double x = 0.; x <= 640.; x += 8.) line.push_back({x, 0., 0.});
    const auto chain = fiberStripChain(line);
    std::vector<FiberStripAxisTarget> targets(chain.points.size());
    // Inferred normals over the first half only.
    std::vector<FiberStripSample> inferred;
    for (double x = 0.; x <= 320.; x += 16.) inferred.push_back({x, {x, 0., 0.}, {0., 0., -1.}, {0., 0., 0.}});
    const auto nodes = setFiberNeighbourTargets(chain, inferred, 300., 32., targets);
    CHECK(nodes > 0);
    CHECK(nodes < chain.points.size());
    CHECK(targets.front().signedAxis[2] == doctest::Approx(-1.));
    CHECK(targets.front().signedWeight == doctest::Approx(300.));
    CHECK(targets.back().signedWeight == 0.);
}

TEST_CASE("a strip samples file keeps its fibers and leaves out the empty ones")
{
    const auto path = std::filesystem::temp_directory_path() / "vc_test_strip_samples.bin";
    const std::vector<FiberStripUuid> ids{FiberStripUuid{1}, FiberStripUuid{2}, FiberStripUuid{3}};
    const std::vector<std::vector<FiberStripSample>> samples{straight(0., 0., {0., 0., 1.}, 32.), {}, straight(8., 0., {0., 1., 0.}, 16.)};
    writeFiberStripSampleFile(path, ids, samples);
    const auto file = readFiberStripSampleFile(path);
    std::filesystem::remove(path);
    REQUIRE(file.ids.size() == 2);
    CHECK(file.ids[1] == ids[2]);
    CHECK(file.samples[0].size() == 3);
    CHECK(file.samples[1][1].point[0] == 16.);
    CHECK(file.samples[1][0].normal[1] == 1.);
}
