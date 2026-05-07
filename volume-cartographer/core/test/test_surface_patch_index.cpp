#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <unordered_set>

namespace {

cv::Mat_<cv::Vec3f> makeSurfaceWithInvalidTileCorners()
{
    cv::Mat_<cv::Vec3f> points(9, 9);
    points.setTo(cv::Vec3f(-1.0f, -1.0f, -1.0f));

    for (int row = 2; row <= 6; ++row) {
        for (int col = 2; col <= 6; ++col) {
            points(row, col) = cv::Vec3f(static_cast<float>(col),
                                         static_cast<float>(row),
                                         static_cast<float>(row));
        }
    }

    return points;
}

} // namespace

TEST_CASE("SurfacePatchIndex keeps tiles with valid interior quads and invalid tile corners")
{
    auto surface = std::make_shared<QuadSurface>(makeSurfaceWithInvalidTileCorners(),
                                                 cv::Vec2f(1.0f, 1.0f));
    surface->id = "interior-island";

    SurfacePatchIndex index;
    index.setSamplingStride(1);
    index.rebuild({surface});

    PlaneSurface plane(cv::Vec3f(0.0f, 0.0f, 3.5f), cv::Vec3f(0.0f, 0.0f, 1.0f));
    std::unordered_set<SurfacePatchIndex::SurfacePtr> targets{surface};
    const auto intersections = index.computePlaneIntersections(
        plane, cv::Rect(0, 0, 9, 9), targets);

    auto it = intersections.find(surface);
    REQUIRE(it != intersections.end());
    CHECK_FALSE(it->second.empty());
}
