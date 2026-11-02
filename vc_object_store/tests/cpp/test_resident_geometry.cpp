#include <vc_object_store/geometry/SurfaceIndex.hpp>
#include <vc_object_store/geometry/VertexIndex.hpp>
#include <random>
#include <stdexcept>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while(false)
int main() {
    std::mt19937 rng(17);
    cv::Mat_<cv::Vec3f> points(17,19);
    std::vector<uint8_t> valid(points.total(),1);
    for (std::size_t i=0;i<points.total();++i) {
        points(int(i))={float(int(rng()%30)-15),float(int(rng()%30)-15),float(int(rng()%30)-15)};
        valid[i]=rng()%4!=0;
    }
    vc::geometry::VertexIndex index;index.rebuild(points,valid);
    for(int trial=0;trial<200;++trial) {
        const cv::Vec3f query{float(int(rng()%40)-20),float(int(rng()%40)-20),float(int(rng()%40)-20)};
        float best=INFINITY;std::size_t chosen=0;
        for(std::size_t i=0;i<points.total();++i) if(valid[i]) {
            const auto d=points(int(i))-query;const float squared=d.dot(d);
            if(squared<best) {best=squared;chosen=i;}
        }
        const auto hit=index.nearest(query);CHECK(hit && hit->index==chosen && hit->distanceSq==best);
    }
    cv::Mat_<cv::Vec3f> grid(2,2);
    grid(0,0)={0,0,0};grid(0,1)={1,0,0};grid(1,0)={0,1,0};grid(1,1)={1,1,2};
    valid.assign(4,1);
    auto a=std::make_shared<vc::object_links::SurfaceGrid>(grid,cv::Vec2f(1,1),valid);
    auto b=std::make_shared<vc::object_links::SurfaceGrid>(grid,cv::Vec2f(1,1),valid);
    vc::object_links::SurfaceIndex surfaces;surfaces.rebuild({a,b});
    vc::object_links::SurfaceIndex::PointQuery query{{0.5f,0.5f,0},0,{}};
    CHECK(surfaces.locateAll(query).size()==2);
    CHECK(surfaces.locate(query)->grid==cv::Vec2f(0.5f,0.5f));
    std::unordered_set<vc::object_links::SurfaceGrid*> excluded{a.get()};
    query.surfaces.excludeRaw=&excluded;
    CHECK(surfaces.locate(query)->surface==b);
    query.surfaces.only=a.get();CHECK(!surfaces.locate(query));
}
