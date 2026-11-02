#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/SurfaceQuerySet.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/geometry/VertexIndex.hpp>

TEST_CASE("resident surface queries retain identity, grid coordinates and immutable edits") {
    cv::Mat_<cv::Vec3f> points(3,3);
    for(int y=0;y<3;++y) for(int x=0;x<3;++x) points(y,x)={float(x),float(y),5.f};
    auto first=std::make_shared<QuadSurface>(points.clone(),cv::Vec2f(1,1));
    auto second=std::make_shared<QuadSurface>(points.clone(),cv::Vec2f(1,1));
    SurfaceQuerySet index;
    index.rebuild({first,second});
    SurfaceQuerySet::PointQuery query{{0.25f,0.75f,5.5f},0.5f,{}};
    auto all=index.locateAll(query);
    REQUIRE(all.size()==2);
    CHECK(all[0].surface==first);
    CHECK(first->ptrToGrid(all[0].ptr)==cv::Vec2f(0.25f,0.75f));
    query.surfaces.only=second;
    REQUIRE(index.locateAll(query).size()==1);
    CHECK(index.locate(query)->surface==second);
    for (auto& p : *second->rawPointsPtr()) p[2]=50;
    CHECK(index.locate(query));
    index.rebuild({first,second});
    CHECK_FALSE(index.locate(query));
}
TEST_CASE("nearest vertices retain original indices and explicit negative validity") {
    cv::Mat_<cv::Vec3f> points(1,4);
    points(0,0)={-1,-1,-1}; points(0,1)={-5,2,3};
    points(0,2)={10,2,3}; points(0,3)={-5,2,3};
    std::vector<uint8_t> valid{0,1,1,1};
    vc::geometry::VertexIndex index;
    index.rebuild(points,valid);
    auto hit=index.nearest({-5,2,3},0);
    REQUIRE(hit);
    CHECK(hit->index==1);
    CHECK_FALSE(index.nearest({0,2,3},4.9f));
}
TEST_CASE("warped cells use triangle coordinates, distinct from reader bilinear patches") {
    cv::Mat_<cv::Vec3f> points(2,2);
    points(0,0)={0,0,0};points(0,1)={1,0,0};points(1,0)={0,1,0};points(1,1)={1,1,2};
    std::vector<uint8_t> valid(4,1);
    auto grid=std::make_shared<vc::object_links::SurfaceGrid>(points,cv::Vec2f(1,1),valid);
    vc::object_links::SurfaceIndex index;index.rebuild({grid});
    const auto hit=index.locate({{0.5f,0.5f,0.f},0.f,{}});
    REQUIRE(hit);CHECK(hit->grid==cv::Vec2f(0.5f,0.5f));
    vcr::grid::Surface reader({reinterpret_cast<const float*>(points.data),2,2,6});
    const float xy[]{0.5f,0.5f}; float xyz[3];reader.pointsAt(xy,1,xyz,nullptr);
    CHECK(xyz[2]==doctest::Approx(0.5f));
}
