#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "spatial/PatchIndex.hpp"
using namespace vc::object_links;

TEST_CASE("patch queries preserve grid coordinates, holes and per-surface hits") {
    cv::Mat_<cv::Vec3f> points(18,18);
    for(int r=0;r<points.rows;++r)for(int c=0;c<points.cols;++c)points(r,c)={float(c),float(r),20.f};
    auto a=std::make_shared<PatchGrid>(points,cv::Vec2f(.5f,2.f));
    auto shifted=points.clone();for(auto& p:shifted)p[2]+=1;
    auto b=std::make_shared<PatchGrid>(shifted,cv::Vec2f(1.f,1.f));
    PatchIndex index;index.rebuild({a,b});
    auto hits=index.locateAll({{8.25f,8.5f,20.2f},2.f,{}});
    REQUIRE(hits.size()==2);
    CHECK(hits[0].surface==a);CHECK(hits[1].surface==b);
    CHECK(hits[0].grid[0]==doctest::Approx(8.25));CHECK(hits[0].grid[1]==doctest::Approx(8.5));
    CHECK(hits[0].distance==doctest::Approx(.2).epsilon(1e-4));
    auto closest=index.locate({{8.f,8.f,20.9f},2.f,{}});REQUIRE(closest);CHECK(closest->surface==b);
    std::unordered_set<PatchGrid*> selected{b.get()};
    auto filtered=index.locateAll({{8,8,20},2.f,{&selected}});REQUIRE(filtered.size()==1);CHECK(filtered[0].surface==b);
    std::array<cv::Vec3f,2> queries{{{0,0,20},{17,17,20}}};size_t count=0;
    index.locateAllBatch(queries,2.f,{},[&](size_t,const auto& result){CHECK(result.size()==2);++count;});CHECK(count==2);
    points(8,8)={-1,-1,-1};index.rebuild({std::make_shared<PatchGrid>(points,cv::Vec2f(1,1))});
    CHECK(index.locateAll({{8,8,20},.1f,{}}).empty());
    CHECK(index.locateAll({{8,8,20},0.f,{}}).empty());
    CHECK(index.locateAll({{NAN,8,20},1.f,{}}).empty());
}

TEST_CASE("degenerate and empty surfaces do not create nonfinite hits") {
    PatchIndex index;index.rebuild({});CHECK(index.locateAll({{0,0,0},1,{}}).empty());
    cv::Mat_<cv::Vec3f> points(2,2,cv::Vec3f(1,1,1));
    index.rebuild({std::make_shared<PatchGrid>(points,cv::Vec2f(1,1))});
    auto result=index.locate({{1,1,1},1,{}});REQUIRE(result);CHECK(result->distance==0.f);
}
