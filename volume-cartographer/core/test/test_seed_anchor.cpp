#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/tracer/SeedAnchor.hpp"
#include <limits>

namespace {
struct Pull {
    template <typename T> bool operator()(const T* p, T* residual) const {
        residual[0] = p[0] - T(4225); residual[1] = p[1] - T(3186);
        residual[2] = p[2] - T(4602); return true;
    }
};
void solve(double* point, const vc::tracer::SeedAnchor& anchor) {
    ceres::Problem p;
    p.AddResidualBlock(new ceres::AutoDiffCostFunction<Pull,3,3>(new Pull), nullptr, point);
    anchor.apply(p);
    ceres::Solver::Options o; o.linear_solver_type=ceres::DENSE_QR;
    o.num_threads=1; o.max_num_iterations=20;
    ceres::Solver::Summary s; ceres::Solve(o,&p,&s);
    CHECK(s.IsSolutionUsable());
    anchor.verify();
}
}
TEST_CASE("unanchored control drifts but anchored solves preserve the seed") {
    double loose[3]={4201,3186,4602}; vc::tracer::SeedAnchor disabled;
    solve(loose,disabled); CHECK(loose[0] > 4224);
    double fixed[3]={4201,3186,4602}; vc::tracer::SeedAnchor anchor;
    anchor.configure({4201,3186,4602}); anchor.bind(fixed);
    for(int i=0;i<5;++i) solve(fixed,anchor);
    CHECK(fixed[0]==4201); CHECK(fixed[1]==3186); CHECK(fixed[2]==4602);
}
TEST_CASE("a float32-rounded anchor binds again by its coordinate and stays put") {
    vc::tracer::SeedAnchor fresh; fresh.configure({4124.4,3173.2,4615.7});
    auto xyz=fresh.xyz(); double loaded[3]={xyz[0],xyz[1],xyz[2]};
    vc::tracer::SeedAnchor resumed; resumed.configure({4124.4,3173.2,4615.7});
    resumed.bind(loaded); solve(loaded,resumed); CHECK(loaded[0]==xyz[0]);
}
TEST_CASE("moved missing duplicated or nonfinite anchors fail closed") {
    vc::tracer::SeedAnchor a; a.configure({4201,3186,4602});
    CHECK_THROWS(a.verify()); double bad[3]={4225,3186,4602}; CHECK_THROWS(a.bind(bad));
    double good[3]={4201,3186,4602}; a.bind(good); CHECK_THROWS(a.bind(good));
    good[0]+=1; CHECK_THROWS(a.verify());
    vc::tracer::SeedAnchor b; CHECK_THROWS(b.configure({NAN,0,0}));
    vc::tracer::SeedAnchor c; CHECK_THROWS(c.configure({-1,0,0}));
}
TEST_CASE("windows without anchor do not acquire a synthetic parameter block") {
    vc::tracer::SeedAnchor a; a.configure({4201,3186,4602});
    double seed[3]={4201,3186,4602}; a.bind(seed);
    ceres::Problem p; a.apply(p); CHECK(p.NumParameterBlocks()==0);
    double other[3]={0,0,0}; solve(other,a); CHECK(other[0]>4224);
}
