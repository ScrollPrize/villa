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
// Per-type constraint weights. A zero weight omits that type's arcs; the
// enclosure pass derives exact H->V relations from signed perpendicular
// events regardless of the perpendicular weight.
struct FiberWindingGraphOptions {
    double windingScale=1,perpendicularOffset=.5,perpendicularWeight=1,parallelWeight=1;
    bool enclosures=false;
    size_t constraintBatch=500000,readBatch=32768,workers=32;
};
struct FiberWindingL1Columns {
    std::unique_ptr<int64_t[]> a,b,target;
    std::unique_ptr<double[]> weight;
    size_t arcs{},read{},used{},expanded{},withoutDistance{};
    // Enclosure relations (unweighted, H node -> V node, exact integer step),
    // present when FiberWindingGraphOptions::enclosures is set.
    std::unique_ptr<int64_t[]> enclosureA,enclosureB,enclosureTarget;
    size_t enclosureArcs{},enclosurePairs{},enclosureOneSided{},enclosureContradictory{},enclosureNonunique{};
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
// One signed perpendicular bound on the H->V step of a pair, as
// fiber_enclosure_graph.extract_relations defines it: outward evidence
// (sign>0 from H towards V) bounds the step from below, inward evidence
// from above (step-1), with the seam/connector transport already applied.
struct EnclosureBound {int64_t h,v;int sign;int64_t bound;};
inline bool enclosureBefore(const EnclosureBound& x,const EnclosureBound& y){
    if(x.h!=y.h)return x.h<y.h;
    if(x.v!=y.v)return x.v<y.v;
    if(x.sign!=y.sign)return x.sign<y.sign;
    return x.bound<y.bound;
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

// One measured relation, as both the bulk preparation below and a single-event
// caller (a manual scaffold's crossings) state it: the two catalog nodes with
// their winding families and whole-fiber normal signs, the anchors, the
// measured winding distance in turns (already scaled; NaN for a relation
// without one) and the connector and sign-free normals at the event.
struct WindingEvent {
    int64_t a=0,b=0,familyA=0,familyB=0,signA=1,signB=1;
    double arcA=0,arcB=0,distance=0;
    bool parallel=false;
    Point connector{},normalA{},normalB{};
    // Weight of this relation's type; zero witnesses enclosures only.
    double weight=0;
    bool enclosure=false; // collect the signed enclosure bound of this event
    double perpendicularOffset=.5;
};
// What one event contributes: its signed order, the arcs of the fractional
// target expansion in node order (a < b) and its enclosure bound.
struct WindingEventRelation {
    int sign=0;
    bool withoutDistance=false,usable=false,hasEnclosure=false,hasUpper=false,hasTransport=false;
    winding_graph_detail::EnclosureBound enclosure{};
    winding_graph_detail::Arc lower{},upper{};
    // The whole-fiber transport of the a->b step (connector crossings minus
    // b's turns plus a's), present whenever the event yielded an arc or an
    // enclosure bound: target = anchor step + transport.
    int64_t transport=0;
};
inline WindingEventRelation windingEventRelation(const PackedFiberGeometry& geometry,const WindingEvent& event)
{
    using namespace winding_graph_detail;
    const auto a=event.a,b=event.b;
    WindingEventRelation out;
    out.sign=int(fiberSignedOrder(event.connector,
        Point{event.normalA[0]*double(event.signA),event.normalA[1]*double(event.signA),event.normalA[2]*double(event.signA)},
        Point{event.normalB[0]*double(event.signB),event.normalB[1]*double(event.signB),event.normalB[2]*double(event.signB)}));
    const int delta=int(event.familyB-event.familyA);const bool parallel=event.parallel;
    const double weight=event.weight;
    const bool enclosure=event.enclosure&&!parallel&&out.sign!=0;
    if(weight<=0&&!enclosure)return out;
    const double distance=event.distance;
    // NaN is a measured relation without a winding distance: it yields no
    // automatic arc but still witnesses enclosures.
    const bool hasDistance=!std::isnan(distance);
    if(hasDistance&&(!std::isfinite(distance)||distance<0))throw std::invalid_argument("invalid measured winding separation");
    out.withoutDistance=!hasDistance&&weight>0;
    const auto plus=hasDistance?integerTarget(distance,1,delta,parallel,event.perpendicularOffset):0;
    const auto minus=hasDistance?integerTarget(distance,-1,delta,parallel,event.perpendicularOffset):0;
    const bool usable=hasDistance&&weight>0&&!(out.sign==0&&plus!=minus);
    if(!usable&&!enclosure)return out;
    const auto start=geometry.sample(a,event.arcA),finish=geometry.sample(b,event.arcB);
    const auto c=geometry.seam().transport(start.point,finish.point);
    out.transport=(int64_t(c)-finish.turn)+start.turn;
    out.hasTransport=true;
    if(enclosure){
        // Transport-only shift of the a->b step; oriented from the H fiber.
        const bool forward=event.familyA==0;
        const int64_t shift=(int64_t(c)-finish.turn)+start.turn;
        const int oriented=forward?out.sign:-out.sign;
        const int64_t step=forward?shift:-shift;
        out.enclosure={forward?a:b,forward?b:a,oriented,oriented>0?step:step-1};
        out.hasEnclosure=true;
    }
    if(!usable)return out;
    const double outward=distance-out.sign*(parallel?0.:event.perpendicularOffset*delta);
    const double minimum=!parallel&&out.sign*delta<0?1.:0.;
    const double measured=out.sign==0?double(plus):out.sign*std::max(outward,minimum);
    const double corrected=((measured+double(c))-double(finish.turn))+double(start.turn);
    const double floor=std::floor(corrected),fraction=corrected-floor;
    constexpr double safe=double(std::numeric_limits<int64_t>::max()/4);
    if(!std::isfinite(corrected)||floor>safe||floor<-safe)throw std::invalid_argument("winding target outside supported integer range");
    const auto arc=[&](int64_t target,double value){
        return a>b?Arc{b,a,-target,value}:Arc{a,b,target,value};
    };
    out.lower=arc(int64_t(floor),weight*(1-fraction));
    if(fraction>0){out.upper=arc(int64_t(floor)+1,weight*fraction);out.hasUpper=true;}
    out.usable=true;
    return out;
}

inline FiberWindingL1Columns prepareFiberWindingGraph(const PackedFiberGeometry& geometry,
    const FiberWindingGraphInput& in,const FiberWindingGraphOptions& options={})
{
    using namespace winding_graph_detail;
    const size_t m=in.a.count,n=in.family.size();
    if(!options.workers||!options.constraintBatch||!options.readBatch||
       !std::isfinite(options.windingScale)||options.windingScale<0||
       !std::isfinite(options.perpendicularOffset)||!std::isfinite(options.perpendicularWeight)||options.perpendicularWeight<0||
       !std::isfinite(options.parallelWeight)||options.parallelWeight<0)
        throw std::invalid_argument("invalid winding graph preparation settings");
    if(options.perpendicularWeight<=0&&options.parallelWeight<=0&&!options.enclosures)
        throw std::invalid_argument("winding graph preparation needs a positive parallel or perpendicular weight, or enclosures");
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
    struct Job {std::vector<Arc> arcs;std::vector<EnclosureBound> enclosures;size_t used{},expanded{},withoutDistance{};};
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
                WindingEvent event;
                event.a=a;event.b=b;event.familyA=in.family[a];event.familyB=in.family[b];
                event.signA=in.normalSign[a];event.signB=in.normalSign[b];
                event.arcA=in.arcA[k];event.arcB=in.arcB[k];
                event.distance=in.distance[k]*options.windingScale;
                event.parallel=in.parallel[k];
                event.connector=point(in.connector,k);
                event.normalA=point(in.normalA,k);event.normalB=point(in.normalB,k);
                event.weight=event.parallel?options.parallelWeight:options.perpendicularWeight;
                event.enclosure=options.enclosures;
                event.perpendicularOffset=options.perpendicularOffset;
                const auto relation=windingEventRelation(geometry,event);
                part.withoutDistance+=relation.withoutDistance;
                if(relation.hasEnclosure)part.enclosures.push_back(relation.enclosure);
                if(!relation.usable)continue;
                batch.push_back(relation.lower);
                if(relation.hasUpper)upper.push_back(relation.upper);
                ++part.used;
            }
            batch.insert(batch.end(),upper.begin(),upper.end());part.expanded+=batch.size();
            aggregate(batch);part.arcs.insert(part.arcs.end(),batch.begin(),batch.end());
        }
        aggregate(part.arcs);
    });
    FiberWindingL1Columns out;out.read=m;
    for(const auto& part:parts){out.used+=part.used;out.expanded+=part.expanded;out.withoutDistance+=part.withoutDistance;}
    if(options.enclosures){
        // Exact relations need every bound of a pair: gather, sort by pair,
        // then accept pairs whose largest lower bound equals the smallest
        // upper bound. Rejections are counted, never silently dropped arcs.
        std::vector<EnclosureBound> bounds;
        for(auto& part:parts){bounds.insert(bounds.end(),part.enclosures.begin(),part.enclosures.end());std::vector<EnclosureBound>().swap(part.enclosures);}
        std::sort(bounds.begin(),bounds.end(),enclosureBefore);
        std::vector<Arc> relations;
        for(size_t i=0;i<bounds.size();){
            size_t j=i;bool lowerSeen=false,upperSeen=false;int64_t lower=0,upper=0;
            for(;j<bounds.size()&&bounds[j].h==bounds[i].h&&bounds[j].v==bounds[i].v;++j){
                if(bounds[j].sign>0){lower=lowerSeen?std::max(lower,bounds[j].bound):bounds[j].bound;lowerSeen=true;}
                else{upper=upperSeen?std::min(upper,bounds[j].bound):bounds[j].bound;upperSeen=true;}
            }
            ++out.enclosurePairs;
            if(!lowerSeen||!upperSeen)++out.enclosureOneSided;
            else if(lower>upper)++out.enclosureContradictory;
            else if(lower<upper)++out.enclosureNonunique;
            else relations.push_back(Arc{bounds[i].h,bounds[i].v,lower,1.});
            i=j;
        }
        out.enclosureArcs=relations.size();
        out.enclosureA=std::make_unique_for_overwrite<int64_t[]>(out.enclosureArcs);
        out.enclosureB=std::make_unique_for_overwrite<int64_t[]>(out.enclosureArcs);
        out.enclosureTarget=std::make_unique_for_overwrite<int64_t[]>(out.enclosureArcs);
        for(size_t i=0;i<out.enclosureArcs;++i){out.enclosureA[i]=relations[i].a;out.enclosureB[i]=relations[i].b;out.enclosureTarget[i]=relations[i].target;}
    }
    if(!out.used&&!out.enclosureArcs)throw std::invalid_argument("no usable winding constraints");
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
