#pragma once
#include "GraphFiberJunctions.hpp"
#include <set>
#include <tuple>

namespace vc::constraints {
namespace detail {
struct ChartRegion {
    cv::Vec2d low{INFINITY,INFINITY},high{-INFINITY,-INFINITY};
    void include(cv::Vec2d p){for(int d=0;d<2;++d){low[d]=std::min(low[d],p[d]);high[d]=std::max(high[d],p[d]);}}
    ChartRegion padded(double margin)const{return {low-cv::Vec2d::all(margin),high+cv::Vec2d::all(margin)};}
    bool contains(cv::Vec2d p)const{return p[0]>=low[0]&&p[1]>=low[1]&&p[0]<=high[0]&&p[1]<=high[1];}
    bool intersects(const ChartRegion& other)const{return low[0]<=other.high[0]&&high[0]>=other.low[0]&&low[1]<=other.high[1]&&high[1]>=other.low[1];}
};
inline auto contactKey(const Link& l){
    return std::tie(l.a.node,l.a.lineIndex,l.a.grid[0],l.a.grid[1],l.b.node,l.b.lineIndex,l.b.grid[0],l.b.grid[1],l.keypoint,l.pending);
}
struct ContactLess {bool operator()(const Link& a,const Link& b)const{return contactKey(a)<contactKey(b);}};
struct PreparedContact {Weights a,b;};
struct ChartIndex {
    ChartRegion bounds;
    static constexpr size_t blockSize=64;
    SharedVector<ChartRegion> blocks;
    explicit ChartIndex(const Placement& p){
        for(size_t i=0;i<p.source->triangles.size();++i){
            if(i%blockSize==0)blocks.push_back(ChartRegion{});
            for(int v:p.source->triangles[i].val){bounds.include(p.uv[v]);blocks.back().include(p.uv[v]);}
        }
    }
    std::vector<size_t> query(const ChartRegion& region,const Node& source)const{
        std::vector<size_t> out;
        if(!bounds.intersects(region))return out;
        for(size_t b=0;b<blocks.size();++b)if(blocks[b].intersects(region))
            for(size_t i=b*blockSize;i<std::min((b+1)*blockSize,source.triangles.size());++i)out.push_back(i);
        return out;
    }
};
struct SurfaceTileStats {size_t quads=0;SharedVector<double> stretches;};
using TileKey=std::pair<int,int>;
inline SurfaceTileStats surfaceTileStats(const cv::Mat_<cv::Vec3f>& grid,cv::Vec2d origin,double spacing,TileKey tile){
    constexpr int side=64;
    const int ox=int(std::lround(origin[0]/spacing)),oy=int(std::lround(origin[1]/spacing));
    SurfaceTileStats stats;
    for(int y=std::max(0,tile.second*side-oy);y<std::min(grid.rows-1,(tile.second+1)*side-oy);++y)
        for(int x=std::max(0,tile.first*side-ox);x<std::min(grid.cols-1,(tile.first+1)*side-ox);++x){
            if(grid(y,x)[0]==-1||grid(y,x+1)[0]==-1||grid(y+1,x)[0]==-1||grid(y+1,x+1)[0]==-1)continue;
            ++stats.quads;
            for(const auto edge:std::array<std::pair<cv::Vec3f,cv::Vec3f>,2>{{{grid(y,x),grid(y,x+1)},{grid(y,x),grid(y+1,x)}}}){
                const double length=cv::norm(edge.second-edge.first);
                if(length>1e-8)stats.stretches.push_back(std::max(length/spacing,spacing/length));
            }
        }
    return stats;
}
inline TileKey tileAt(cv::Vec2d uv,double spacing){return {int(std::floor(uv[0]/spacing/64)),int(std::floor(uv[1]/spacing/64))};}
}

// Immutable revision data. All callers share it; editing a placement detaches
// its UV storage, while source geometry and unrelated placements remain shared.
struct PreparedGraph {
    FitResult accepted;
    std::map<std::string,detail::ChartIndex> charts;
    std::map<std::string,SharedVector<double>> chartStretches;
    std::map<detail::TileKey,detail::SurfaceTileStats> surfaceTiles;
    std::vector<detail::PreparedContact> contacts;
    std::vector<double> squaredErrors;
    std::map<std::string,std::vector<size_t>> incident;
    std::vector<size_t> junctionLinks;
    std::shared_ptr<const detail::FiberJunctions> junctions;
    std::set<std::pair<std::string,std::string>> linked;
    size_t vertices=0;
};
namespace detail {std::set<std::pair<std::string,std::string>> editedPairs(const GraphEdit&);}
}
