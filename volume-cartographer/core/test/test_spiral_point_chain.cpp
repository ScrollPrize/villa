#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "SpiralPointChain.hpp"

#include <cmath>

namespace {

std::optional<cv::Vec3f> planarSample(const QPointF& point)
{
    return cv::Vec3f(static_cast<float>(point.x()),
                     static_cast<float>(point.y()), 0.0f);
}

vc3d::spiral::PointChainAnchor anchor(float x, float y)
{
    return {QPointF(x, y), cv::Vec3f(x, y, 0.0f)};
}

} // namespace

TEST_CASE("Spiral point chain retains anchors and inserts ordered 30 voxel samples")
{
    const auto result = vc3d::spiral::buildPointChain(
        {anchor(0.0f, 0.0f), anchor(75.0f, 0.0f)}, planarSample);
    REQUIRE(result.error == vc3d::spiral::PointChainBuildError::None);
    REQUIRE(result.samples.size() == 4);
    CHECK(result.samples[0].isAnchor);
    CHECK(result.samples[0].volume[0] == doctest::Approx(0.0f));
    CHECK_FALSE(result.samples[1].isAnchor);
    CHECK(result.samples[1].volume[0] == doctest::Approx(30.0f));
    CHECK_FALSE(result.samples[2].isAnchor);
    CHECK(result.samples[2].volume[0] == doctest::Approx(60.0f));
    CHECK(result.samples[3].isAnchor);
    CHECK(result.samples[3].anchorIndex == 1);
    CHECK(result.samples[3].volume[0] == doctest::Approx(75.0f));
}

TEST_CASE("Spiral point chain adds no derived sample inside a short anchor span")
{
    const auto result = vc3d::spiral::buildPointChain(
        {anchor(2.0f, 3.0f), anchor(22.0f, 3.0f)}, planarSample);
    REQUIRE(result.error == vc3d::spiral::PointChainBuildError::None);
    REQUIRE(result.samples.size() == 2);
    CHECK(result.samples[0].isAnchor);
    CHECK(result.samples[1].isAnchor);
    CHECK(result.samples[0].surface == QPointF(2.0, 3.0));
    CHECK(result.samples[1].surface == QPointF(22.0, 3.0));
}

TEST_CASE("Spiral point chain preserves click order across curved spans")
{
    const auto result = vc3d::spiral::buildPointChain(
        {anchor(0.0f, 0.0f), anchor(40.0f, 0.0f), anchor(55.0f, 25.0f)},
        planarSample);
    REQUIRE(result.error == vc3d::spiral::PointChainBuildError::None);
    std::vector<std::size_t> anchorOrder;
    for (const auto& sample : result.samples) {
        if (sample.isAnchor) anchorOrder.push_back(sample.anchorIndex);
    }
    CHECK(anchorOrder == std::vector<std::size_t>{0, 1, 2});
    CHECK(result.samples.front().surface == QPointF(0.0, 0.0));
    CHECK(result.samples.back().surface == QPointF(55.0, 25.0));
}

TEST_CASE("Spiral point chain rejects a self-intersecting candidate")
{
    const auto result = vc3d::spiral::buildPointChain(
        {anchor(0.0f, 0.0f), anchor(20.0f, 20.0f),
         anchor(0.0f, 20.0f), anchor(20.0f, 0.0f)},
        planarSample);
    CHECK(result.error == vc3d::spiral::PointChainBuildError::SelfIntersection);
    CHECK(result.samples.empty());
}

TEST_CASE("Spiral anchor eraser trims only contiguous ends")
{
    using vc3d::spiral::AnchorEraseAction;
    auto decision = vc3d::spiral::classifyAnchorErase(
        {true, true, false, false, true});
    CHECK(decision.action == AnchorEraseAction::Trim);
    CHECK(decision.removePrefix == 2);
    CHECK(decision.removeSuffix == 1);

    decision = vc3d::spiral::classifyAnchorErase(
        {false, true, false, false});
    CHECK(decision.action == AnchorEraseAction::DeleteChain);

    decision = vc3d::spiral::classifyAnchorErase(
        {true, false});
    CHECK(decision.action == AnchorEraseAction::DeleteChain);

    decision = vc3d::spiral::classifyAnchorErase(
        {false, false, false});
    CHECK(decision.action == AnchorEraseAction::None);
}
