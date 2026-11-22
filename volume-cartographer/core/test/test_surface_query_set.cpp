#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/SurfaceQuerySet.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <vc_object_store/geometry/VertexIndex.hpp>
#include <vc_object_store/objects/ResidentSurface.hpp>
#include <vc_object_store/objects/ObjectStoreSession.hpp>
#include <vc_object_store/Uuid.hpp>

TEST_CASE("displayed surface cuts retain geometry and share draft intersection semantics") {
    vc::geometry::Surface grid{2,2,{1,1},{{0,0,5},{1,0,5},{0,1,5},{1,1,5}}};
    const auto uuid = vc::objects::util::newUuid();
    const vc::objects::ResidentSurface selected(grid);
    vc::objects::RegionSlice slab;
    slab.origin={0.5,0,0}; slab.u={0,1,0}; slab.v={0,0,1}; slab.normal={1,0,0};
    slab.localBounds={-1,-1,-0.1,2,6,0.1};
    const auto shown = selected.region(uuid,slab);
    REQUIRE(shown.complete());
    REQUIRE_FALSE(shown.runs.empty());
    vc::objects::ObjectStoreSession session;
    session.setDraft(uuid,grid,nlohmann::json::object());
    vc::objects::RegionOptions options; options.kind="surface";
    const auto drafted = session.geometryRegion({},slab,options);
    REQUIRE(drafted.size()==1);
    REQUIRE(drafted.front().result.runs.size()==shown.runs.size());
    for(size_t i=0;i<shown.runs.size();++i) {
        CHECK(drafted.front().result.runs[i].points==shown.runs[i].points);
        CHECK(drafted.front().result.runs[i].grid==shown.runs[i].grid);
    }
    for(auto& p:grid.points) p[0]+=10;
    const vc::objects::ResidentSurface successor(grid);
    CHECK(successor.region(uuid,slab).runs.empty());
    CHECK(selected.region(uuid,slab).runs.front().points==shown.runs.front().points);
    CHECK(selected.region(uuid,slab,[](const auto&) {return true;}).stop==vc::objects::RegionStop::Cancelled);
}

TEST_CASE("resident surface queries retain identity, grid coordinates and immutable edits") {
    cv::Mat_<cv::Vec3f> points(3,3);
    for(int y=0;y<3;++y) for(int x=0;x<3;++x) points(y,x)={float(x),float(y),5.f};
    auto first=std::make_shared<QuadSurface>(points.clone(),cv::Vec2f(1,1));
    auto second=std::make_shared<QuadSurface>(points.clone(),cv::Vec2f(1,1));
    // Two retained versions of the same object are separate displayed grids.
    first->id = second->id = "selected-object";
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
    const auto selectedPtr = index.locate(query)->ptr;
    second->shiftSurfaceOrigin({3,4});
    CHECK(index.locate(query)->ptr == selectedPtr);
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
