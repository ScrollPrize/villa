#include "LineStripInternal.hpp"
#include <ceres/gradient_problem.h>
#include <ceres/gradient_problem_solver.h>
#include <ceres/jet.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <limits>
#include <numeric>

namespace vc::lasagna::strip {
namespace {
struct RepairOptions {
    int margin=12, iterations=180;
    double normalWeight=1, qualityWeight=1000, orientationWeight=1000, normalTether=0;
    bool preserveWidth=false;
};
using Jet=ceres::Jet<double,12>;
using JVec=Eigen::Matrix<Jet,3,1>;
Jet square(const Jet& a){return a*a;}
Jet positive(const Jet& a){return a.a>0?a:Jet(0.);}
JVec constant(Vec v){return {Jet(v[0]),Jet(v[1]),Jet(v[2])};}

// The experimental repair objective expressed in Ceres' native first-order
// solver. Derivatives use small, local 12-variable jets; no dense Jacobian or
// runtime automatic-differentiation framework is needed.
class RepairObjective final : public ceres::FirstOrderFunction {
public:
    RepairObjective(Grid initial,Grid field,int first,int totalColumns,
                    RobustLineStripOptions options,RepairOptions repair)
        : initial_(std::move(initial)), normals_(std::move(field)), options_(std::move(options)), repair_(repair),
          best_(initial_), origin_(initial_(initial_.rows/2,initial_.cols/2)),scale_(options_.crossSpacing/2) {
        const Vec reference=normals_(normals_.rows/2,normals_.cols/2);
        for(auto& n:normals_.points)if(n.dot(reference)<0)n=-n;
        normals_=gaussianAlong(normals_,2.);
        for(auto& n:normals_.points)n=unit(n);
        std::vector<double> signs;
        for(int r=0;r<initial_.rows-1;++r)for(int c=0;c<initial_.cols-1;++c) {
            const Vec n=unit(normals_(r,c)+normals_(r,c+1)+normals_(r+1,c)+normals_(r+1,c+1));
            signs.push_back((initial_(r,c+1)-initial_(r,c)).cross(initial_(r+1,c)-initial_(r,c)).dot(n));
        }
        std::sort(signs.begin(),signs.end());const size_t mid=signs.size()/2;
        sign_=((signs.size()%2?signs[mid]:.5*(signs[mid-1]+signs[mid]))>=0)?1.:-1.;
        active_.resize(initial_.points.size());
        for(int r=0;r<initial_.rows;++r)for(int c=0;c<initial_.cols;++c)
            active_[r*initial_.cols+c]=r!=initial_.rows/2&&!(first>0&&c==0)&&!(first+initial_.cols<totalColumns&&c==initial_.cols-1);
        bestScore_={badCount(initial_),std::numeric_limits<double>::infinity()};
    }
    int NumParameters() const override {return int(initial_.points.size())*3;}
    Grid result() const {return best_;}
    bool Evaluate(const double* x,double* cost,double* gradient) const override {
        check(options_);*cost=0;if(gradient)std::fill_n(gradient,NumParameters(),0.);
        const int rows=initial_.rows,cols=initial_.cols,n=rows*cols;
        auto position=[&](int i){Vec p=(initial_.points[i]-origin_)/scale_;if(active_[i])for(int d=0;d<3;++d)p[d]+=x[3*i+d];return p;};
        auto term=[&](std::initializer_list<int> ids,auto function) {
            std::vector<JVec> p; p.reserve(ids.size());int k=0;
            for(int i:ids){const auto q=position(i);JVec v=constant(q);for(int d=0;d<3;++d)v[d].v[3*k+d]=active_[i]?1.:0.;p.push_back(v);++k;}
            const Jet value=function(p);*cost+=value.a;
            if(gradient){k=0;for(int i:ids){for(int d=0;d<3;++d)gradient[3*i+d]+=value.v[3*k+d];++k;}}
        };
        const double quads=double(rows-1)*(cols-1);
        for(int r=0;r<rows-1;++r)for(int c=0;c<cols-1;++c) {
            const int a=r*cols+c,b=a+1,e=a+cols,d=e+1;
            const Vec target=sign_*unit(normals_.points[a]+normals_.points[b]+normals_.points[e]+normals_.points[d]);
            term({a,b,e,d},[&](const auto& p) {
                const JVec u=(p[1]-p[0])/Jet(2.),v=(p[2]-p[0])/Jet(2.),u2=(p[3]-p[2])/Jet(2.),v2=(p[3]-p[1])/Jet(2.);
                auto quality=[&](const JVec& x,const JVec& y){const Jet aa=x.dot(x),bb=x.dot(y),cc=y.dot(y),disc=ceres::sqrt(square(aa-cc)+Jet(4.)*bb*bb+Jet(1e-12));return square(positive(Jet(.16)-(aa+cc-disc)/Jet(2.)))+square(positive((aa+cc+disc)/Jet(2.)-Jet(6.25)));};
                const JVec cross=u.cross(v),cross2=u2.cross(v2),normal=constant(target);
                return Jet(repair_.qualityWeight/(2*quads))*(quality(u,v)+quality(u2,v2))
                    +Jet(repair_.orientationWeight/(2*quads))*(square(positive(Jet(.25)-cross.dot(normal)))+square(positive(Jet(.25)-cross2.dot(normal))))
                    +Jet(repair_.qualityWeight/quads)*square(positive(Jet(.10)-cross.dot(cross2)));
            });
        }
        auto edge=[&](int a,int b,double weight){const Vec normal=unit(normals_.points[a]+normals_.points[b]);term({a,b},[&](const auto& p){return Jet(weight)*square((p[1]-p[0]).dot(constant(normal)));});};
        for(int r=0;r<rows;++r)for(int c=0;c<cols-1;++c)edge(r*cols+c,r*cols+c+1,repair_.normalWeight/(rows*double(cols-1)));
        for(int r=0;r<rows-1;++r)for(int c=0;c<cols;++c)edge(r*cols+c,(r+1)*cols+c,repair_.normalWeight/((rows-1)*double(cols)));
        auto fair=[&](int a,int b,int c,double weight) {
            const Vec rest=(initial_.points[a]-2*initial_.points[b]+initial_.points[c])/scale_;
            term({a,b,c},[&](const auto& p){return Jet(weight)*(p[0]-Jet(2.)*p[1]+p[2]-constant(rest)).squaredNorm();});
        };
        if(cols>2)for(int r=0;r<rows;++r)for(int c=0;c<cols-2;++c)fair(r*cols+c,r*cols+c+1,r*cols+c+2,.03/(3*rows*double(cols-2)));
        for(int r=0;r<rows-2;++r)for(int c=0;c<cols;++c)fair(r*cols+c,(r+1)*cols+c,(r+2)*cols+c,.03/(3*(rows-2)*double(cols)));
        for(int i=0;i<n;++i)if(active_[i]) {
            Vec delta{x[3*i],x[3*i+1],x[3*i+2]};const double dot=delta.dot(normals_.points[i]);
            *cost+=.002*delta.dot(delta)/(3*n)+repair_.normalTether*dot*dot/n;
            if(gradient)for(int d=0;d<3;++d)gradient[3*i+d]+=2*.002*delta[d]/(3*n)+2*repair_.normalTether*dot*normals_.points[i][d]/n;
        }
        if(repair_.preserveWidth) {
            const double half=rows-1;
            for(int c=0;c<cols;++c)for(int side=0;side<2;++side) {
                const int begin=side*(rows/2),end=begin+rows/2;double length=0;
                for(int r=begin;r<end;++r)length+=cv::norm(position((r+1)*cols+c)-position(r*cols+c));
                *cost+=500*std::pow((length-half)/half,2)/(2*cols);
                for(int r=begin;r<end;++r) {
                    const int a=r*cols+c,b=a+cols;const Vec edge=position(b)-position(a);const double len=cv::norm(edge);
                    *cost+=2*std::pow(len-2,2)/((rows-1)*cols);
                    const Vec grad=unit(edge)*(1000*(length-half)/(half*half*2*cols)+4*(len-2)/((rows-1)*cols));
                    if(gradient)for(int d=0;d<3;++d){if(active_[a])gradient[3*a+d]-=grad[d];if(active_[b])gradient[3*b+d]+=grad[d];}
                }
            }
        }
        if(!std::isfinite(*cost))return false;
        Grid g=initial_;for(int i=0;i<n;++i)g.points[i]=position(i)*scale_+origin_;
        for(int c=0;c<cols;++c)g(rows/2,c)=initial_(rows/2,c);
        const auto score=std::pair{badCount(g),*cost};if(score<bestScore_){bestScore_=score;best_=std::move(g);}
        return true;
    }
private:
    int badCount(const Grid& g) const {
        const auto bad=badQuads(g,options_);int count=std::count(bad.begin(),bad.end(),1);
        if(repair_.preserveWidth){const double half=(g.rows/2)*options_.crossSpacing;for(double w:widths(g))count+=w<.95*half||w>1.05*half;}
        return count;
    }
    Grid initial_,normals_;
    RobustLineStripOptions options_;RepairOptions repair_;
    mutable Grid best_;mutable std::pair<int,double> bestScore_;
    Vec origin_;double scale_,sign_;
    std::vector<unsigned char> active_;
};

Grid repairRegions(const Grid& base,const Grid& field,const RobustLineStripOptions& o,RepairOptions p={}) {
    auto cols=badColumns(base,o,2.85);
    if(p.preserveWidth){const auto w=widths(base);const double half=(base.rows/2)*o.crossSpacing;for(int c=0;c<base.cols;++c)cols[c]|=w[c]<half*58/60||w[c]>half*62/60||w[base.cols+c]<half*58/60||w[base.cols+c]>half*62/60;}
    const auto distance=distances(cols);Grid result=base;
    for(int start=0;start<base.cols;) {
        if(distance[start]>p.margin){++start;continue;}
        int end=start+1;while(end<base.cols&&distance[end]<=p.margin)++end;
        const int lo=std::max(0,start-1),hi=std::min(base.cols,end+1);check(o);
        Grid part(base.rows,hi-lo),norm(base.rows,hi-lo);
        for(int r=0;r<base.rows;++r)for(int c=lo;c<hi;++c){part(r,c-lo)=base(r,c);norm(r,c-lo)=field(r,c);}
        auto* objective=new RepairObjective(part,norm,lo,base.cols,o,p);
        ceres::GradientProblem problem(objective);std::vector<double> delta(size_t(objective->NumParameters()),0.);
        ceres::GradientProblemSolver::Options options;options.line_search_direction_type=ceres::LBFGS;options.max_lbfgs_rank=12;
        options.max_num_iterations=p.iterations;options.function_tolerance=1e-10;options.gradient_tolerance=1e-7;options.parameter_tolerance=1e-10;
        options.logging_type=ceres::SILENT;ceres::GradientProblemSolver::Summary summary;ceres::Solve(options,problem,delta.data(),&summary);
        const auto repaired=objective->result();for(int r=0;r<base.rows;++r)for(int c=lo;c<hi;++c)result(r,c)=repaired(r,c-lo);
        start=end;
    }
    return result;
}
std::pair<int,double> geometryScore(const Grid& g,const RobustLineStripOptions& o) {
    const auto bad=badQuads(g,o);double worst=0;
    for(int r=0;r<g.rows-1;++r)for(int c=0;c<g.cols-1;++c)for(int second=0;second<2;++second) {
        const Vec u=(second?g(r+1,c+1)-g(r+1,c):g(r,c+1)-g(r,c))/o.alongSpacing,
                  v=(second?g(r+1,c+1)-g(r,c+1):g(r+1,c)-g(r,c))/o.crossSpacing;
        const double a=u.dot(u),b=u.dot(v),d=v.dot(v),disc=std::sqrt((a-d)*(a-d)+4*b*b);
        worst=std::max({worst,std::sqrt(std::max(0.,(a+d+disc)/2)),1/std::sqrt(std::max(1e-24,(a+d-disc)/2))});
    }
    return {std::count(bad.begin(),bad.end(),1),worst};
}
Grid smoothRepair(const Grid& base,const RobustLineStripOptions& o,bool offsets) {
    Grid g=base;const int middle=g.rows/2;
    for(double sigma:{1.,2.,4.,8.,12.,20.}) {
        auto mask=badColumns(g,o,2.8);if(std::none_of(mask.begin(),mask.end(),[](auto b){return b;}))break;
        const auto distance=distances(mask);Grid input=g;
        if(offsets)for(int r=0;r<g.rows;++r)for(int c=0;c<g.cols;++c)input(r,c)-=g(middle,c);
        auto target=gaussianAlong(input,sigma,true);
        for(int r=0;r<g.rows;++r)for(int c=0;c<g.cols;++c) {
            double w=std::clamp(1-distance[c]/int(4*sigma),0.,1.);w=w*w*(3-2*w);
            if(offsets)target(r,c)+=g(middle,c);else w*=std::min(std::abs(r-middle)/4.,1.);
            if(r!=middle)g(r,c)+=w*(target(r,c)-g(r,c));
        }
    }
    return g;
}
Grid straighten(const Grid& base,const Grid& original,const RobustLineStripOptions& o) {
    Grid g=base,side(1,g.cols);const int middle=g.rows/2;
    const auto distance=distances(badColumns(base,o,2.85));
    for(int c=0;c<g.cols;++c)side(0,c)=original(middle+1,c)-original(middle-1,c);
    side=gaussianAlong(side,4.);for(auto& p:side.points)p=unit(p);
    for(int r=0;r<g.rows;++r)for(int c=0;c<g.cols;++c) {
        double w=std::clamp(1-distance[c]/32,0.,1.);w=w*w*(3-2*w);
        if(r!=middle)g(r,c)+=w*(original(middle,c)+(r-middle)*o.crossSpacing*side(0,c)-g(r,c));
    }
    return g;
}
} // namespace

Grid repairSeed(const Grid& base,const Grid& field,const RobustLineStripOptions& o) {
    std::vector<Grid> candidates;
    auto add=[&](Grid g){g=quantize(std::move(g));candidates.push_back(g);return g;};
    auto acceptable=[&](const Grid& g){return valid(g,o)&&widthOK(g,o);};
    Grid g=base;
    RepairOptions tangent;tangent.margin=16;tangent.iterations=540;tangent.normalTether=2;
    for(int i=0;i<3;++i){auto candidate=repairRegions(g,field,o,tangent);if(geometryScore(candidate,o)<geometryScore(g,o))g=std::move(candidate);if(valid(g,o))break;}
    g=add(g);if(acceptable(g))return g;
    g=add(repairRegions(base,field,o));if(acceptable(g))return g;
    auto smooth=smoothRepair(base,o,true),surface=smoothRepair(base,o,false);
    g=geometryScore(smooth,o)<geometryScore(surface,o)?smooth:surface;
    RepairOptions hybrid;hybrid.margin=24;hybrid.iterations=360;hybrid.normalWeight=.2;
    g=add(repairRegions(quantize(g),field,o,hybrid));if(acceptable(g))return g;
    if(!valid(g,o)) {
        RepairOptions refined;refined.margin=16;refined.iterations=720;refined.normalWeight=.1;
        for(int i=0;i<3;++i){auto candidate=repairRegions(g,field,o,refined);if(geometryScore(candidate,o)<geometryScore(g,o))g=std::move(candidate);if(valid(g,o))break;}
        g=add(g);
    }
    if(!valid(g,o)) {
        g=straighten(g,base,o);RepairOptions p;p.margin=24;p.iterations=720;p.normalWeight=.1;
        for(int i=0;i<3;++i){g=repairRegions(g,field,o,p);if(valid(g,o))break;}g=add(g);
    }
    if(!valid(g,o)){RepairOptions p;p.margin=24;p.iterations=720;p.normalWeight=.1;p.qualityWeight=10000;p.orientationWeight=1;g=add(repairRegions(g,field,o,p));}
    double best=std::numeric_limits<double>::infinity();bool found=false;
    for(const auto& candidate:candidates)if(valid(candidate,o)) {
        if(widthOK(candidate,o))return candidate;
        const double half=(candidate.rows/2)*o.crossSpacing;double violation=0;
        for(double w:widths(candidate))violation+=std::max(half*.95-w,0.)+std::max(w-half*1.05,0.);
        if(violation<best){best=violation;g=candidate;found=true;}
    }
    if(!found)throw std::runtime_error("No geometry-valid full-width robust strip repair");
    RepairOptions width;width.margin=24;width.iterations=720;width.normalTether=2;width.preserveWidth=true;width.qualityWeight=10000;width.orientationWeight=1;
    for(int i=0;i<3;++i){g=repairRegions(g,field,o,width);if(acceptable(g))break;}
    g=quantize(g);if(!acceptable(g))throw std::runtime_error("Robust strip repair cannot preserve strip width");
    return g;
}
} // namespace vc::lasagna::strip
