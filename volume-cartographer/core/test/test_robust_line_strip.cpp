#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/lasagna/RobustLineStrip.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <cmath>

namespace {
struct Field : vc::lasagna::NormalSampler {
    bool cylinder=false,flip=false,missing=false;
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& p) const override {
        if(missing)return {{},false,"test missing field"};
        cv::Vec3d n=cylinder?cv::Vec3d{0,p[1],p[2]}:cv::Vec3d{0,0,1};n/=cv::norm(n);
        if(flip&&int(std::floor(p[0]/7)+std::floor(p[1]/7))%2)n=-n;
        return {n,true,{}};
    }
};
vc::lasagna::LineModel line() {
    vc::lasagna::LineModel line;
    for(int x=0;x<=80;x+=20)line.points.push_back({{double(x),0,200},{{0,0,1},true,{}},true});
    return line;
}
vc::lasagna::LineViewConfig config(const Field& f) {
    vc::lasagna::LineViewConfig c;c.crossSpacingBaseVoxels=20;c.crossSampleCount=7;
    c.targetSpacingBaseVoxels=20;c.controlPointLinePositions={0,4};c.normalSampler=&f;c.buildLineZSlices=false;return c;
}
}
TEST_CASE("robust fitting preserves the requested output grid and centerline") {
    Field f;const auto views=vc::lasagna::buildLineViewSurfaces(line(),config(f));const auto& g=*views.lineSurface->rawPointsPtr();
    CHECK(g.rows==7);CHECK(g.cols==5);CHECK(views.stripPositionMap.stripGridColumnCount==5);
    for(int r=0;r<g.rows;++r)for(int c=0;c<g.cols;++c){CHECK(g(r,c)[2]==200);CHECK(std::abs(g(r,c)[1])<=60.001);}
    for(int c=0;c<g.cols;++c){CHECK(g(3,c)[0]==20*c);CHECK(g(3,c)[1]==0);}
}
TEST_CASE("robust strip follows cylindrical sheet across its full width") {
    Field f;f.cylinder=true;const auto views=vc::lasagna::buildLineViewSurfaces(line(),config(f));const auto& g=*views.lineSurface->rawPointsPtr();
    for(int r=0;r<g.rows;++r)for(int c=0;c<g.cols;++c)CHECK(std::abs(std::hypot(g(r,c)[1],g(r,c)[2])-200)<.05);
    CHECK(std::abs(g(0,2)[2]-200)>8);
}
TEST_CASE("robust strip retains its valid fitting mesh when the output grid is too coarse") {
    Field f;auto shortLine=line();shortLine.points.resize(2);shortLine.points[1].position[0]=5;
    auto c=config(f);c.controlPointLinePositions={0,1};
    const auto views=vc::lasagna::buildLineViewSurfaces(shortLine,c);
    const auto& g=*views.lineSurface->rawPointsPtr();
    CHECK(g.cols>2);CHECK(g.rows==61);
    CHECK(views.stripPositionMap.stripGridColumnCount==size_t(g.cols));
    CHECK(views.stripPositionMap.stripGridSpacingBaseVoxels==2);
    CHECK(views.lineSideSlice->rawPointsPtr()->size()==g.size());
    CHECK(g(g.rows/2,0)==cv::Vec3f(0,0,200));
    CHECK(g(g.rows/2,g.cols-1)==cv::Vec3f(5,0,200));
}
TEST_CASE("robust normal fit reduces sheet offset and is deterministic") {
    Field f;cv::Mat_<cv::Vec3f> seed(61,41);
    for(int r=0;r<seed.rows;++r)for(int c=0;c<seed.cols;++c)seed(r,c)={float(2*c),float(2*(r-30)),float(200+2*std::pow((r-30)/30.,2))};
    auto a=vc::lasagna::fitRobustLineStrip(seed,f,false),b=vc::lasagna::fitRobustLineStrip(seed,f,false);
    double before=0,after=0;
    for(int r=0;r<seed.rows;++r)for(int c=0;c<seed.cols;++c){before+=std::abs(seed(r,c)[2]-200);after+=std::abs(a(r,c)[2]-200);CHECK(a(r,c)==b(r,c));}
    CHECK(after<before);for(int c=0;c<seed.cols;++c)CHECK(a(30,c)==seed(30,c));
}
TEST_CASE("field-axis sign changes do not change robust geometry") {
    Field f;f.cylinder=true;const auto a=vc::lasagna::buildLineViewSurfaces(line(),config(f));f.flip=true;
    const auto b=vc::lasagna::buildLineViewSurfaces(line(),config(f));
    CHECK(cv::norm(*a.lineSurface->rawPointsPtr(),*b.lineSurface->rawPointsPtr(),cv::NORM_INF)<.001);
}
TEST_CASE("missing normals and cancellation do not publish partial strips") {
    Field f;f.missing=true;CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(line(),config(f)),std::runtime_error);
    f.missing=false;auto c=config(f);c.checkCancelled=[]{throw std::runtime_error("cancelled");};
    CHECK_THROWS_AS(vc::lasagna::buildLineViewSurfaces(line(),c),std::runtime_error);
}
