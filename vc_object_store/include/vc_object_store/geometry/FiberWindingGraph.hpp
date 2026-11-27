#pragma once
#include "vc_object_store/geometry/FiberWinding.hpp"
#include <queue>

namespace vc::geometry {

template<class T>struct WindingGraphColumn {
    const T* data{};
    size_t count{};
    ptrdiff_t stride{1};
    const T& operator[](size_t i)const{return data[i*stride];}
};

// Connector points from A toward B; the two supplied normals point inward.
// Shared by the ordinary signed_order adapter and native winding preparation.
inline float fiberSignedOrder(const Point& connector,const Point& inwardA,const Point& inwardB,double cosineMinimum=.1)
{
    const double length=std::sqrt((connector[0]*connector[0]+connector[1]*connector[1])+connector[2]*connector[2]);
    const double divisor=std::max(length,1e-12);
    const double x=connector[0]/divisor,y=connector[1]/divisor,z=connector[2]/divisor;
    const double a=-((x*inwardA[0]+y*inwardA[1])+z*inwardA[2]);
    const double b=-((x*inwardB[0]+y*inwardB[1])+z*inwardB[2]);
    return a*b>0&&std::min(std::abs(a),std::abs(b))>=cosineMinimum&&length>0?(a>0?1.f:-1.f):0.f;
}

struct FiberWindingGraphInput {
    WindingGraphColumn<int64_t> a,b;
    std::span<const int64_t> graphToCatalog,family,normalSign;
    WindingGraphColumn<double> arcA,arcB,distance;
    std::span<const bool> parallel,validA,validB,validConnector;
    std::span<const double> normalA,normalB,connector; // XYZ triples, unexpanded
};
struct FiberWindingGraphOptions {
    double windingScale=1,perpendicularOffset=.5,perpendicularWeight=1;
    size_t constraintBatch=500000,readBatch=32768,workers=32;
};
struct FiberWindingL1Columns {
    std::unique_ptr<int64_t[]> a,b,target;
    std::unique_ptr<double[]> weight;
    size_t arcs{},read{},used{},expanded{};
};

namespace winding_graph_detail {
struct Arc {int64_t a,b,target;double weight;};
inline bool before(const Arc& a,const Arc& b){
    if(a.a!=b.a)return a.a<b.a;
    if(a.b!=b.b)return a.b<b.b;
    return a.target<b.target;
}
inline bool same(const Arc& a,const Arc& b){return a.a==b.a&&a.b==b.b&&a.target==b.target;}
inline void aggregate(std::vector<Arc>& arcs){
    if(arcs.empty())return;
    std::stable_sort(arcs.begin(),arcs.end(),before);
    size_t out=0;
    for(size_t i=1;i<arcs.size();++i){
        if(same(arcs[out],arcs[i]))arcs[out].weight+=arcs[i].weight;
        else arcs[++out]=arcs[i];
    }
    arcs.resize(out+1);
}
inline int64_t integerTarget(double distance,int sign,int familyDelta,bool parallel,double offset){
    const double outward=distance-sign*(parallel?0.:offset*familyDelta);
    const double minimum=!parallel&&sign*familyDelta<0?1.:0.;
    const double value=sign*std::max(std::ceil(outward-.5),minimum);
    constexpr double safe=double(std::numeric_limits<int64_t>::max()/4);
    if(!std::isfinite(value)||value>safe||value<-safe)throw std::invalid_argument("winding target outside supported integer range");
    return int64_t(value);
}
inline Point point(std::span<const double> values,size_t at,double sign=1.){
    return {values[3*at]*sign,values[3*at+1]*sign,values[3*at+2]*sign};
}
}

inline FiberWindingL1Columns prepareFiberWindingGraph(const PackedFiberGeometry& geometry,
    const FiberWindingGraphInput& in,const FiberWindingGraphOptions& options={})
{
    using namespace winding_graph_detail;
    const size_t m=in.a.count,n=in.family.size();
    if(!options.workers||!options.constraintBatch||!options.readBatch||
       !std::isfinite(options.windingScale)||options.windingScale<0||
       !std::isfinite(options.perpendicularOffset)||!std::isfinite(options.perpendicularWeight)||options.perpendicularWeight<=0)
        throw std::invalid_argument("invalid winding graph preparation settings");
    if(in.b.count!=m||in.arcA.count!=m||in.arcB.count!=m||in.distance.count!=m||
       in.parallel.size()!=m||in.validA.size()!=m||in.validB.size()!=m||in.validConnector.size()!=m||
       in.normalA.size()!=m*3||in.normalB.size()!=m*3||in.connector.size()!=m*3||in.normalSign.size()!=n||
       in.a.stride<1||in.b.stride<1||in.arcA.stride<1||in.arcB.stride<1||in.distance.stride<1)
        throw std::invalid_argument("winding graph input columns differ");
    vc::objects::parallelFor((n+4095)/4096,options.workers,[&](size_t block){
        for(size_t i=block*4096;i<std::min(n,(block+1)*4096);++i)
            if((in.family[i]!=0&&in.family[i]!=1)||(in.normalSign[i]!=-1&&in.normalSign[i]!=1))
                throw std::invalid_argument("invalid winding family or normal sign");
    });
    const auto mapped=[&](int64_t node){
        if(node<0)throw std::invalid_argument("negative winding graph node");
        const auto owner=in.graphToCatalog.empty()?node:
            uint64_t(node)<in.graphToCatalog.size()?in.graphToCatalog[node]:int64_t(-2);
        if(owner<-1||(owner>=0&&uint64_t(owner)>=n))throw std::invalid_argument("winding node outside catalog");
        return owner;
    };
    struct Job {std::vector<Arc> arcs;size_t used{},expanded{};};
    const size_t jobs=(m+options.constraintBatch-1)/options.constraintBatch;
    std::vector<Job> parts(jobs);
    vc::objects::parallelFor(jobs,options.workers,[&](size_t job){
        auto& part=parts[job];const auto first=job*options.constraintBatch,last=std::min(m,first+options.constraintBatch);
        std::vector<Arc> batch,upper;batch.reserve(std::min(options.readBatch,last-first)*2);
        upper.reserve(std::min(options.readBatch,last-first));
        for(size_t begin=first;begin<last;begin+=options.readBatch){
            batch.clear();upper.clear();const auto end=std::min(last,begin+options.readBatch);
            for(size_t k=begin;k<end;++k){
                const auto a=mapped(in.a[k]),b=mapped(in.b[k]);
                if(a<0||b<0||!in.validA[k]||!in.validB[k]||!in.validConnector[k]||
                   in.parallel[k]!=(in.family[a]==in.family[b]))continue;
                const auto connector=point(in.connector,k);
                const int sign=int(fiberSignedOrder(connector,point(in.normalA,k,in.normalSign[a]),point(in.normalB,k,in.normalSign[b])));
                const int delta=int(in.family[b]-in.family[a]);const bool parallel=in.parallel[k];
                const double distance=in.distance[k]*options.windingScale;
                if(!std::isfinite(distance)||distance<0)throw std::invalid_argument("invalid measured winding separation");
                const auto plus=integerTarget(distance,1,delta,parallel,options.perpendicularOffset);
                const auto minus=integerTarget(distance,-1,delta,parallel,options.perpendicularOffset);
                if(sign==0&&plus!=minus)continue;
                const double outward=distance-sign*(parallel?0.:options.perpendicularOffset*delta);
                const double minimum=!parallel&&sign*delta<0?1.:0.;
                const double measured=sign==0?double(plus):sign*std::max(outward,minimum);
                const auto start=geometry.sample(a,in.arcA[k]),finish=geometry.sample(b,in.arcB[k]);
                const auto c=geometry.seam().transport(start.point,finish.point);
                const double corrected=((measured+double(c))-double(finish.turn))+double(start.turn);
                const double floor=std::floor(corrected),fraction=corrected-floor;
                constexpr double safe=double(std::numeric_limits<int64_t>::max()/4);
                if(!std::isfinite(corrected)||floor>safe||floor<-safe)throw std::invalid_argument("winding target outside supported integer range");
                const double weight=parallel?1.:options.perpendicularWeight;
                const auto append=[&](std::vector<Arc>& rows,int64_t target,double value){
                    rows.push_back(a>b?Arc{b,a,-target,value}:Arc{a,b,target,value});
                };
                append(batch,int64_t(floor),weight*(1-fraction));
                if(fraction>0)append(upper,int64_t(floor)+1,weight*fraction);
                ++part.used;
            }
            batch.insert(batch.end(),upper.begin(),upper.end());part.expanded+=batch.size();
            aggregate(batch);part.arcs.insert(part.arcs.end(),batch.begin(),batch.end());
        }
        aggregate(part.arcs);
    });
    FiberWindingL1Columns out;out.read=m;
    for(const auto& part:parts){out.used+=part.used;out.expanded+=part.expanded;}
    if(!out.used)throw std::invalid_argument("no usable winding constraints");
    // Merge independent sorted job outputs in disjoint A-node ranges. Equal
    // keys are summed in the same job order as concatenation+stable lexsort.
    const size_t ranges=std::min(std::max(size_t(1),n),options.workers);
    std::vector<std::vector<Arc>> merged(ranges);
    vc::objects::parallelFor(ranges,options.workers,[&](size_t range){
        const int64_t low=int64_t(n*range/ranges),high=int64_t(n*(range+1)/ranges);
        struct Cursor {const Arc* current;const Arc* end;size_t job;};
        struct Later {bool operator()(const Cursor& a,const Cursor& b)const{
            if(same(*a.current,*b.current))return a.job>b.job;
            return before(*b.current,*a.current);
        }};
        std::priority_queue<Cursor,std::vector<Cursor>,Later> queue;
        auto& result=merged[range];
        for(size_t j=0;j<parts.size();++j){
            const auto& rows=parts[j].arcs;
            const auto begin=std::lower_bound(rows.begin(),rows.end(),low,[](const Arc& a,int64_t value){return a.a<value;});
            const auto end=std::lower_bound(begin,rows.end(),high,[](const Arc& a,int64_t value){return a.a<value;});
            if(begin!=end)queue.push({&*begin,rows.data()+(end-rows.begin()),j});
        }
        while(!queue.empty()){
            auto cursor=queue.top();queue.pop();
            if(!result.empty()&&same(result.back(),*cursor.current))result.back().weight+=cursor.current->weight;
            else result.push_back(*cursor.current);
            if(++cursor.current!=cursor.end)queue.push(cursor);
        }
    });
    std::vector<size_t> offset(ranges+1,0);for(size_t i=0;i<ranges;++i)offset[i+1]=offset[i]+merged[i].size();
    out.arcs=offset.back();
    out.a=std::make_unique_for_overwrite<int64_t[]>(out.arcs);out.b=std::make_unique_for_overwrite<int64_t[]>(out.arcs);
    out.target=std::make_unique_for_overwrite<int64_t[]>(out.arcs);out.weight=std::make_unique_for_overwrite<double[]>(out.arcs);
    vc::objects::parallelFor(ranges,options.workers,[&](size_t range){
        const auto& rows=merged[range];for(size_t i=0;i<rows.size();++i){const auto at=offset[range]+i;
            out.a[at]=rows[i].a;out.b[at]=rows[i].b;out.target[at]=rows[i].target;out.weight[at]=rows[i].weight;}
    });
    return out;
}
} // namespace vc::geometry
