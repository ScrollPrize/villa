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

