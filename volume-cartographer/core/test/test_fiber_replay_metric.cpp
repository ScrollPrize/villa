#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

using namespace vc::fiber_tracer;

namespace {
class Normals final : public vc::lasagna::NormalSampler {
public:
    cv::Vec3d normal{0, 0, 1};
    bool valid = true;
    mutable cv::Vec3d lastPoint{};
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& point) const override {
        lastPoint = point;
        return {normal, valid, {}};
    }
};
}

TEST_CASE("normal and tangential radii have strict inclusive boundaries") {
    Normals normals;
    for (double factor : {2., 4.}) {
        for (auto delta : {cv::Vec3d{0,0,40}, cv::Vec3d{40*factor,0,0}}) {
            const auto boundary = measureFiberReplayThreshold(delta, {}, normals, 1, 40, factor);
            CHECK(boundary.thresholdErrorRatio == doctest::Approx(1));
            CHECK_FALSE(fiberReplayThresholdExceeded(boundary, 40));
            CHECK(fiberReplayThresholdExceeded(
                measureFiberReplayThreshold(delta*1.01, {}, normals, 1, 40, factor), 40));
        }
        CHECK(fiberReplayThresholdDescriptorJson(40, factor).at("tangential_radius_base_voxels") == 40*factor);
    }
}

TEST_CASE("normal axes are sign invariant and sampled in working coordinates") {
    Normals normals;
    const cv::Vec3d reference{100,200,300}, point = reference + cv::Vec3d{24,0,8};
    const auto first = measureFiberReplayThreshold(point, reference, normals, 4, 10);
    CHECK(normals.lastPoint == reference/4);
    normals.normal = {0,0,-5};
    const auto second = measureFiberReplayThreshold(point, reference, normals, 4, 10);
    CHECK(first.thresholdErrorBaseVoxels == second.thresholdErrorBaseVoxels);
    CHECK(first.thresholdErrorBaseVoxels == doctest::Approx(10));
    CHECK(fiberReplayThresholdMeasurementJson(first).at("local_normal_valid") == true);
}

TEST_CASE("missing normals use isotropic Euclidean error and zero radius stays finite") {
    Normals normals;
    normals.valid = false;
    const auto result = measureFiberReplayThreshold({3,4,0}, {}, normals, 1, 0);
    CHECK(result.thresholdErrorBaseVoxels == 5);
    CHECK(result.thresholdErrorRatio == std::numeric_limits<double>::max());
    CHECK_FALSE(result.normalErrorBaseVoxels.has_value());
    CHECK(fiberReplayThresholdMeasurementJson(result).at("normal_error_base_voxels").is_null());
    CHECK(measureFiberReplayThreshold({}, {}, normals, 1, 0).thresholdErrorRatio == 0);
    CHECK(fiberReplayOptionalThresholdMeasurementJson(std::nullopt, 10).at("threshold_error_ratio").is_null());
}

TEST_CASE("invalid inputs and inconsistent measurements are rejected") {
    Normals normals;
    CHECK_THROWS(measureFiberReplayThreshold({}, {}, normals, 0, 10));
    CHECK_THROWS(measureFiberReplayThreshold({}, {}, normals, 1, -1));
    CHECK_THROWS(measureFiberReplayThreshold({}, {}, normals, 1, 10, 0));
    CHECK_THROWS(fiberReplayThresholdDescriptorJson(10, -1));
    auto result = measureFiberReplayThreshold({0,0,5}, {}, normals, 1, 10);
    result.thresholdErrorRatio = 100;
    CHECK_THROWS(validateFiberReplayThresholdMeasurement(result, 10));
}

TEST_CASE("excursion grace counts travel after first outside point and resets on reentry") {
    FiberReplayExcursion state{160, std::nullopt};
    CHECK_FALSE(state.observe(true, 20));
    CHECK_FALSE(state.observe(true, 159));
    CHECK_FALSE(state.observe(false, 1));
    CHECK_FALSE(state.outsideTravel.has_value());
    CHECK_FALSE(state.observe(true, 20));
    CHECK(state.observe(true, 160));
    FiberReplayExcursion immediate;
    CHECK(immediate.observe(true, 0));
    CHECK_THROWS(immediate.observe(true, -1));
}

TEST_CASE("real anchor positions and normal volume use the same metric formula") {
    const char* manifest = std::getenv("VC_TEST_METRIC_NORMAL_MANIFEST");
    const char* anchors = std::getenv("VC_TEST_METRIC_ANCHORS");
    if (!manifest || !anchors) return;
    const char* scaleText = std::getenv("VC_TEST_METRIC_NORMAL_SCALE");
    REQUIRE(scaleText != nullptr);
    const double scale = std::stod(scaleText);
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.workingToBaseScale = scale;
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifest, options);
    vc::lasagna::LasagnaNormalSampler normals(dataset);
    std::ifstream input(anchors);
    REQUIRE(input.good());
    const auto data = nlohmann::json::parse(input);
    size_t count = 0;
    for (const auto& cell : data.at("cells")) {
        for (const auto& component : cell.at("components")) {
            if (!component.at("retained").get<bool>()) continue;
            const auto& p = component.at("position_base_xyz");
            const cv::Vec3d point{p[0].get<double>(), p[1].get<double>(), p[2].get<double>()};
            const auto sampled = normals.sampleNormal(point/scale);
            REQUIRE(sampled.valid);
            const auto normal = sampled.normal/cv::norm(sampled.normal);
            const cv::Vec3d delta{12,7,3};
            const double perpendicular = std::abs(delta.dot(normal));
            const double lateral = std::sqrt(std::max(0., delta.dot(delta)-perpendicular*perpendicular));
            const auto actual = measureFiberReplayThreshold(point+delta, point, normals, scale, 20);
            CHECK(actual.thresholdErrorBaseVoxels == doctest::Approx(std::hypot(perpendicular,lateral/4)));
            ++count;
        }
    }
    CHECK(count > 0);
    std::cout << "Validated metric at " << count << " real anchor positions\n";
}
