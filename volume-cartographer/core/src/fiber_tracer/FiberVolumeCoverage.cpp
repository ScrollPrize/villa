#include "vc/fiber_tracer/FiberVolumeCoverage.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <shared_mutex>

namespace vc::fiber_tracer {
namespace {
namespace bg=boost::geometry;
namespace bgi=boost::geometry::index;
using Point=bg::model::point<double,3,bg::cs::cartesian>;
using Box=bg::model::box<Point>;
using Key=std::array<int,3>;
Point point(const cv::Vec3d& p){return Point(p[0],p[1],p[2]);}
cv::Vec3d unit(cv::Vec3d v){return v/std::max(cv::norm(v),1e-9);}
Key key(const cv::Vec3d& p,double b){return {int(std::floor(p[0]/b)),int(std::floor(p[1]/b)),int(std::floor(p[2]/b))};}
struct Sample {cv::Vec3d p,t;int64_t id;double width;};
using Entry=std::pair<Point,size_t>;
using Tree=bgi::rtree<Entry,bgi::quadratic<16>>;
struct Block {
    std::vector<Sample> samples;
    Tree tree;
    explicit Block(std::vector<Sample> values):samples(std::move(values)) {
        std::vector<Entry> entries;entries.reserve(samples.size());
        for(size_t i=0;i<samples.size();++i)entries.emplace_back(point(samples[i].p),i);
        tree=Tree(entries.begin(),entries.end());
    }
};
using Forest=std::vector<std::shared_ptr<const Block>>;
Forest append(Forest trees,std::vector<Sample> samples) {
    while(!trees.empty() && trees.back()->samples.size()<4*samples.size()) {
        const auto& old=trees.back()->samples;
        samples.insert(samples.begin(),old.begin(),old.end());trees.pop_back();
    }
    if(!samples.empty())trees.push_back(std::make_shared<Block>(std::move(samples)));
    return trees;
}
// Preserve the historical nearest-16-before-angle/width-filter contract for
// each recent/bin forest, rather than silently expanding the collision rule.
void nearest(const Forest& forest,const cv::Vec3d& p,const cv::Vec3d& dir,double radius,
    double width,double tolerance,int64_t exclude,double& best,int64_t& result) {
    struct Hit {double distance;const Sample* sample;};
    std::vector<Hit> hits;hits.reserve(forest.size()*16);
    for(const auto& block:forest) {
        for(auto it=block->tree.qbegin(bgi::nearest(point(p),16));it!=block->tree.qend();++it) {
            const auto& sample=block->samples[it->second];
            const double d=cv::norm(p-sample.p);
            if(d<radius)hits.push_back({d,&sample});
        }
    }
    std::stable_sort(hits.begin(),hits.end(),[](const Hit&a,const Hit&b){return a.distance<b.distance;});
    if(hits.size()>16)hits.resize(16);
    for(const auto& h:hits) {
        const auto& s=*h.sample;
        if(h.distance<best && s.id!=exclude && std::abs(s.t.dot(dir))>.8 &&
           FiberVolumeCoverage::compatibleWidth(s.width,width,tolerance)) {
            best=h.distance;result=s.id;
        }
    }
}
struct Segment {cv::Vec3d a,b,dir;int64_t pair;uint64_t epoch;double width;};
using LiveEntry=std::pair<Box,uint64_t>;
using LiveTree=bgi::rtree<LiveEntry,bgi::quadratic<16>>;
Box bounds(const cv::Vec3d& a,const cv::Vec3d& b,double pad=0) {
    cv::Vec3d lo,hi;for(int i=0;i<3;++i){lo[i]=std::min(a[i],b[i])-pad;hi[i]=std::max(a[i],b[i])+pad;}
    return Box(point(lo),point(hi));
}
}
struct FiberVolumeCoverage::Impl {
    double bin;size_t merge,recentCount=0;
    std::mutex writer;
    mutable std::shared_mutex ownedMutex,liveMutex;
    Forest recent;
    std::map<Key,Forest> bins;
    LiveTree live;
    uint64_t nextSegment=0;
    std::map<uint64_t,Segment> segments;
    std::map<int64_t,std::vector<LiveEntry>> pairs;
    Impl(double b,size_t m):bin(b),merge(m){}
    void flush() { // caller holds writer, but not ownedMutex while building
        if(!recentCount)return;
        std::map<Key,std::vector<Sample>> parts;
        for(const auto& block:recent)for(const auto& s:block->samples)parts[key(s.p,bin)].push_back(s);
        std::map<Key,Forest> replacement;
        for(auto& [k,samples]:parts) {
            auto i=bins.find(k);replacement[k]=append(i==bins.end()?Forest{}:i->second,std::move(samples));
        }
        std::unique_lock lock(ownedMutex);
        for(auto& [k,forest]:replacement)bins[k]=std::move(forest);
        recent.clear();recentCount=0;
    }
};
FiberVolumeCoverage::FiberVolumeCoverage(double bin,size_t merge):impl_(std::make_unique<Impl>(bin,merge)) {
    if(!std::isfinite(bin)||bin<=0||merge==0)throw std::invalid_argument("invalid coverage bin/merge size");
}
FiberVolumeCoverage::~FiberVolumeCoverage()=default;
bool FiberVolumeCoverage::compatibleWidth(double a,double b,double tolerance) {
    return tolerance<=0 || !std::isfinite(a)||!std::isfinite(b)||a<=0||b<=0||std::abs(a-b)<=tolerance*b;
}
void FiberVolumeCoverage::addOwned(const std::vector<cv::Vec3d>& path,int64_t id,double width) {
    if(path.empty())return;
    std::vector<Sample> samples;samples.reserve(path.size());
    for(size_t i=0;i<path.size();++i) {
        const auto t=path.size()==1?cv::Vec3d{}:unit(path[std::min(i+1,path.size()-1)]-path[i?i-1:0]);
        samples.push_back({path[i],t,id,width});
    }
    std::lock_guard write(impl_->writer);
    auto replacement=append(impl_->recent,std::move(samples));
    {std::unique_lock lock(impl_->ownedMutex);impl_->recent=std::move(replacement);impl_->recentCount+=path.size();}
    if(impl_->recentCount>impl_->merge)impl_->flush();
}
void FiberVolumeCoverage::flushOwned(){std::lock_guard write(impl_->writer);impl_->flush();}
int64_t FiberVolumeCoverage::owner(const cv::Vec3d& p,const cv::Vec3d& dir,double radius,
    double width,double tolerance,int64_t exclude) const {
    std::shared_lock lock(impl_->ownedMutex);
    int64_t result=-1;double best=std::numeric_limits<double>::infinity();
    nearest(impl_->recent,p,dir,radius,width,tolerance,exclude,best,result);
    const auto lo=key(p-cv::Vec3d(radius,radius,radius),impl_->bin),hi=key(p+cv::Vec3d(radius,radius,radius),impl_->bin);
    for(int x=lo[0];x<=hi[0];++x)for(int y=lo[1];y<=hi[1];++y)for(int z=lo[2];z<=hi[2];++z) {
        auto i=impl_->bins.find({x,y,z});
        if(i!=impl_->bins.end())nearest(i->second,p,dir,radius,width,tolerance,exclude,best,result);
    }
    return result;
}
void FiberVolumeCoverage::addLive(const cv::Vec3d& a,const cv::Vec3d& b,int64_t pair,uint64_t epoch,double width) {
    if(cv::norm(b-a)<=1e-9)return;
    std::unique_lock lock(impl_->liveMutex);
    const auto id=impl_->nextSegment++;
    LiveEntry e{bounds(a,b),id};impl_->segments.emplace(id,Segment{a,b,unit(b-a),pair,epoch,width});
    impl_->pairs[pair].push_back(e);impl_->live.insert(e);
}
bool FiberVolumeCoverage::liveHit(const cv::Vec3d& p,const cv::Vec3d& dir,double radius,int64_t pair,
    uint64_t epoch,double width,double tolerance) const {
    std::shared_lock lock(impl_->liveMutex);
    for(auto it=impl_->live.qbegin(bgi::intersects(bounds(p,p,radius)));it!=impl_->live.qend();++it) {
        const auto& s=impl_->segments.at(it->second);
        if(s.pair==pair||s.epoch>=epoch||std::abs(s.dir.dot(dir))<=.8||!compatibleWidth(s.width,width,tolerance))continue;
        const auto ab=s.b-s.a;const double t=std::clamp((p-s.a).dot(ab)/ab.dot(ab),0.,1.);
        if(cv::norm(s.a+t*ab-p)<=radius)return true;
    }
    return false;
}
void FiberVolumeCoverage::dropLive(int64_t pair) {
    std::unique_lock lock(impl_->liveMutex);
    auto i=impl_->pairs.find(pair);if(i==impl_->pairs.end())return;
    for(const auto& entry:i->second){impl_->live.remove(entry);impl_->segments.erase(entry.second);}
    impl_->pairs.erase(i);
}
} // namespace vc::fiber_tracer
