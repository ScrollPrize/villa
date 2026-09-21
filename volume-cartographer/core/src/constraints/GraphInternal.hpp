#pragma once
#include "vc/constraints/ConstraintGraph.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::constraints::detail {
void remapFiberLocation(Location&, const Node& oldStrip, const Node& newStrip);
void replaceFiberStrip(Graph&, const std::shared_ptr<const Node>&);
std::vector<Link> patchOverlapContacts(const Node& proposal,
    const std::map<std::string, Placement>& references, double radius,
    const std::atomic_bool* cancel, bool checkNormals = false);
FitResult restoreSurface(Graph, const cv::Mat_<cv::Vec3f>&, cv::Vec2d origin);
inline void cancelled(const std::atomic_bool* flag) {
    if (flag && flag->load()) throw std::runtime_error("Operation cancelled");
}
inline bool valid(const cv::Vec3d& p) {
    return p[0] != -1 && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
inline cv::Vec3d unit(cv::Vec3d p) {
    const double n = cv::norm(p);
    return n > 1e-12 ? p / n : cv::Vec3d{0, 0, 0};
}
inline double cross(const cv::Vec2d& a, const cv::Vec2d& b) {
    return a[0]*b[1] - a[1]*b[0];
}
inline void triangulate(Node& node) {
    for (int y=0; y+1<node.rows; ++y) for (int x=0; x+1<node.cols; ++x) {
        if (!node.quadMask.empty() && !node.quadMask[y*(node.cols-1)+x]) continue;
        int a=y*node.cols+x, b=a+1, c=a+node.cols, d=c+1;
        if (!valid(node.xyz[a]) || !valid(node.xyz[b]) || !valid(node.xyz[c]) || !valid(node.xyz[d])) continue;
        if (cv::norm((node.xyz[b]-node.xyz[a]).cross(node.xyz[c]-node.xyz[a])) < 1e-10 ||
            cv::norm((node.xyz[c]-node.xyz[d]).cross(node.xyz[b]-node.xyz[d])) < 1e-10) continue;
        node.triangles.push_back({a,b,c}); node.triangles.push_back({d,c,b});
    }
}
struct Weights { std::array<int,3> indices; cv::Vec3d values; };
inline Weights weights(const Node& n, cv::Vec2d p) {
    constexpr double epsilon=1e-5;
    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || p[0]<-epsilon || p[1]<-epsilon ||
        p[0]>n.cols-1+epsilon || p[1]>n.rows-1+epsilon)
        throw std::runtime_error("Correspondence is outside its source grid");
    p[0]=std::clamp(p[0],0.0,double(n.cols-1));
    p[1]=std::clamp(p[1],0.0,double(n.rows-1));
    // On a grid edge, either incident cell can own the point. A nearest
    // surface hit at the boundary of a hole must use the valid incident cell.
    const int baseX=int(std::floor(p[0])),baseY=int(std::floor(p[1]));
    for(int dy:{0,-1})for(int dx:{0,-1}) {
        int x=baseX+dx,y=baseY+dy;
        if(x<0||y<0||x+1>=n.cols||y+1>=n.rows)continue;
        double u=p[0]-x,v=p[1]-y;
        if(u<-epsilon||v<-epsilon||u>1+epsilon||v>1+epsilon)continue;
        u=std::clamp(u,0.0,1.0);v=std::clamp(v,0.0,1.0);
        if(!n.quadMask.empty()&&!n.quadMask[y*(n.cols-1)+x])continue;
        int a=y*n.cols+x,b=a+1,c=a+n.cols,d=c+1;
        if(!valid(n.xyz[a])||!valid(n.xyz[b])||!valid(n.xyz[c])||!valid(n.xyz[d]))continue;
        if(u+v<=1)return {{a,b,c},{1-u-v,u,v}};
        return {{d,c,b},{u+v-1,1-u,1-v}};
    }
    throw std::runtime_error("Correspondence falls in a source hole");
}

inline Weights weights(const Node& n, const Location& p) {
    return weights(n, p.lineIndex>=0 ? fiberLineLocation(n,static_cast<size_t>(p.lineIndex)).grid : p.grid);
}
inline std::shared_ptr<QuadSurface> sourceSurface(const Node& n) {
    cv::Mat_<cv::Vec3f> points(n.rows,n.cols);
    for (int i=0;i<n.rows*n.cols;++i) points(i/n.cols,i%n.cols)=cv::Vec3f(n.xyz[i]);
    auto out=std::make_shared<QuadSurface>(points,cv::Vec2f{1,1});
    out->setStrictQuadRenderValidity(true);
    return out;
}
inline cv::Vec2d interpolate(const std::vector<cv::Vec2d>& uv, const Weights& w) {
    return uv[w.indices[0]]*w.values[0]+uv[w.indices[1]]*w.values[1]+uv[w.indices[2]]*w.values[2];
}
inline cv::Vec3d interpolate(const std::vector<cv::Vec3d>& xyz, const Weights& w) {
    return xyz[w.indices[0]]*w.values[0]+xyz[w.indices[1]]*w.values[1]+xyz[w.indices[2]]*w.values[2];
}
inline std::pair<const Location*,const Location*> horizontalVerticalLink(const Graph& g,const Link& link) {
    const auto* h=&link.a;const auto* v=&link.b;
    if(g.nodes.at(h->node).source->direction=='V')std::swap(h,v);
    const auto& horizontal=*g.nodes.at(h->node).source;
    const auto& vertical=*g.nodes.at(v->node).source;
    if(!horizontal.fiber||!vertical.fiber||horizontal.direction!='H'||vertical.direction!='V')return {nullptr,nullptr};
    return {h,v};
}
inline std::pair<Weights,Weights> linkWeights(const Graph& g,const Link& link) {
    const auto& a=*g.nodes.at(link.a.node).source;
    const auto& b=*g.nodes.at(link.b.node).source;
    auto wa=weights(a,link.a),wb=weights(b,link.b);
    // A branch records two control locations, which can move apart after
    // tracing. Preserve their measured tangential separation instead of
    // collapsing different XYZ positions into one output grid point.
    // H/V controls coincide in the chart: the horizontal supplies the XYZ
    // position and FiberJunctions blends that correction into the vertical.
    const bool horizontalVertical=horizontalVerticalLink(g,link).first!=nullptr;
    if(!link.keypoint&&a.fiber&&b.fiber&&!horizontalVertical) {
        const auto delta=interpolate(b.xyz,wb)-interpolate(a.xyz,wa);
        const auto x=a.xyz[wa.indices[1]]-a.xyz[wa.indices[0]];
        const auto y=a.xyz[wa.indices[2]]-a.xyz[wa.indices[0]];
        const double det=x.dot(x)*y.dot(y)-x.dot(y)*x.dot(y);
        if(det>1e-12) {
            const double u=(delta.dot(x)*y.dot(y)-delta.dot(y)*x.dot(y))/det;
            const double v=(delta.dot(y)*x.dot(x)-delta.dot(x)*x.dot(y))/det;
            wa.values+=cv::Vec3d{-u-v,u,v};
        }
    }
    return {wa,wb};
}
inline cv::Vec2d linkError(const Graph& g,const Link& l) {
    auto [a,b]=linkWeights(g,l);
    return interpolate(g.nodes.at(l.a.node).uv,a)-interpolate(g.nodes.at(l.b.node).uv,b);
}
} // namespace vc::constraints::detail
