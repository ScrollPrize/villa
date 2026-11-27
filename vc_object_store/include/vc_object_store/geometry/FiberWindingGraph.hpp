#pragma once
#include "vc_object_store/geometry/FiberCrossingFilter.hpp"
#include "vc_object_store/geometry/FiberCrossingWeight.hpp"
#include "vc_object_store/geometry/FiberWinding.hpp"
#include <memory>
#include <queue>
#include <tuple>

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
    // The crossings' normal-surface findings, as the extraction stored them
    // with the constraints; empty for a build that does not filter
    // (WindingConstraintOptions::crossingFilter is set from their presence).
    std::span<const uint8_t> crossingFindings;
    std::span<const double> crossingAlternative;
};
// Per-type constraint weights. A zero weight omits that type's arcs; the
// enclosure pass derives exact H->V relations from signed perpendicular
// events regardless of the perpendicular weight.
// What a measured crossing is turned into, the same for every consumer: the
// traced volume's winding stage and a manual scaffold's solve build their
// constraints from these options through windingEventConstraints below.
// Experiment: what a perpendicular crossing of a known signed order states,
// learned from crossings whose true step is known. Per crossing class (0: the
// V fiber outside the H fiber, 1: the H fiber outside the V fiber) and bin of
// the measured separation, the weights of the terms |n - (stepMin + k)| on
// the anchor-level outward step n: together a convex cost of every whole
// step, in place of the two integers around the measurement. The separation's
// own distance damping is not applied on top: the table states it.
struct WindingStepCostTable {
    double binWidth=0;
    int64_t bins=0,stepMin=0,steps=0;
    std::vector<double> weights; // [class][bin][step]
};
// Experiment: a measured separation d of one perpendicular crossing class is
// read as scale * d + offset turns (never below 0) before it is turned into a
// winding step. The evidence damping keeps the separation as measured.
struct WindingDistanceCalibration {double scale=1,offset=0;};
struct WindingConstraintOptions {
    std::shared_ptr<const WindingStepCostTable> stepCosts;
    // Per class under the given normal signs: the V fiber outside the H fiber,
    // and the H fiber outside the V fiber. A crossing whose normals state no
    // order has no class and keeps its measurement.
    WindingDistanceCalibration calibrationVOutside,calibrationHOutside;
    // Experiment: a measured separation as mostly a lower bound. Placing the
    // two fibers of a signed perpendicular crossing farther apart than
    // measured costs this share of the crossing's weight, placing them closer
    // the whole weight: the equation keeps the share and a one-sided
    // constraint (WindingEventConstraints::bounds) carries the rest. 1: off.
    double fartherApartFactor=1;
    double windingScale=1,perpendicularOffset=.5,perpendicularWeight=1,parallelWeight=1;
    // How much each crossing counts as evidence, on top of its type weight
    // (FiberCrossingWeight.hpp, the rule the sign solve uses): |dot|^power,
    // and with `dampByWindingDistance` also min(1, 1/d^2) of its measured
    // separation. Power 0 without damping is the flat per-type weight.
    double dotWeightPower=0;
    bool dampByWindingDistance=false;
    // Collect the signed enclosure bound of every perpendicular crossing.
    bool enclosures=false;
    // The local sign every signed perpendicular crossing states on its own:
    // the V fiber lies outside the H fiber there, or inside, which bounds the
    // pair's H->V step from one side. Each becomes a one-sided constraint of
    // this weight times the crossing's evidence, costing only when a solve
    // breaks it; zero states none. It needs no winding separation.
    double signWeight=0;
    // Pass every perpendicular crossing through the shared rule
    // (fiberCrossingUse) under the two fibers' signs before it states
    // anything: a crossing the rule drops, or one that would have to take
    // another crossing's dot (which has no measured separation), yields no
    // constraint.
    bool crossingFilter=false;
};
struct FiberWindingGraphOptions : WindingConstraintOptions {
    size_t constraintBatch=500000,readBatch=32768,workers=32;
    // Also return one row per usable event with its fractional target, for a
    // readout that scores a solve against the measurements themselves.
    bool eventTargets=false;
};
struct FiberWindingL1Columns {
    std::unique_ptr<int64_t[]> a,b,target;
    std::unique_ptr<double[]> weight;
    size_t arcs{},read{},used{},expanded{},withoutDistance{};
    // Crossings the findings took out (FiberWindingGraphInput::crossingFindings).
    size_t crossingFiltered{};
    // One-sided sign constraints (WindingConstraintOptions::signWeight), in
    // node order a < b: the step b - a is at least (side +1) or at most
    // (side -1) the target. Identical ones are summed.
    std::unique_ptr<int64_t[]> signA,signB,signTarget,signSide;
    std::unique_ptr<double[]> signWeight;
    size_t signArcs{},signConstraints{};
    // Enclosure relations (unweighted, H node -> V node, exact integer step),
    // present when FiberWindingGraphOptions::enclosures is set.
    std::unique_ptr<int64_t[]> enclosureA,enclosureB,enclosureTarget;
    size_t enclosureArcs{},enclosurePairs{},enclosureOneSided{},enclosureContradictory{},enclosureNonunique{};
    // One row per usable event (FiberWindingGraphOptions::eventTargets): the
    // node pair in arc order and the measured target in turns, unexpanded and
    // unaggregated.
    std::unique_ptr<int64_t[]> eventA,eventB;
    std::unique_ptr<double[]> eventTarget,eventMeasured,eventDistance,eventWeight;
    size_t events{};
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
    // The crossing's normal-surface findings (FiberCrossingFilter.hpp); zero
    // for none recorded.
    uint8_t crossingFindings=0;
    double crossingAlternative=0;
};
// What one event contributes: its signed order, the arcs of the fractional
// target expansion in node order (a < b) and its enclosure bound.
struct WindingEventRelation {
    int sign=0;
    bool withoutDistance=false,usable=false,hasEnclosure=false,hasUpper=false,hasTransport=false;
    winding_graph_detail::EnclosureBound enclosure{};
    winding_graph_detail::Arc lower{},upper{};
    // Under a step cost table: the crossing's further terms, and the flag
    // that lower, upper and these are all stated as they are.
    std::vector<winding_graph_detail::Arc> extra;
    bool costTable=false;
    // The whole-fiber transport of the a->b step (connector crossings minus
    // b's turns plus a's), present whenever the event yielded an arc or an
    // enclosure bound: target = anchor step + transport.
    int64_t transport=0;
    // What the measurement says before the integer expansion: the corrected
    // target in turns, oriented like the arcs (negated when a>b). A readout
    // that scores a solve against the measurements wants this, not the two
    // integer arcs the L1 problem is built from.
    double fractionalTarget=0;
    // The same target with the measurement left as measured: no clamp to the
    // half-winding minimum, so a crossing inside one winding keeps its
    // fractional separation. Only a readout wants this; the L1 problem does
    // not, because a target below the minimum is not a relation it can state.
    double measuredTarget=0;
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
    out.fractionalTarget=a>b?-corrected:corrected;
    const double unclamped=((out.sign==0?double(plus):out.sign*outward)+double(c))-double(finish.turn)+double(start.turn);
    out.measuredTarget=a>b?-unclamped:unclamped;
    out.lower=arc(int64_t(floor),weight*(1-fraction));
    if(fraction>0){out.upper=arc(int64_t(floor)+1,weight*fraction);out.hasUpper=true;}
    out.usable=true;
    return out;
}

// A one-sided statement on the step between two nodes in node order (a < b):
// b - a is at least the target (side +1) or at most it (side -1).
struct WindingSignConstraint {
    int64_t a=0,b=0,target=0;
    double weight=0;
    int side=0;
};
// Everything one measured crossing contributes under the options: its
// relation (the equality arcs of its measured separation and its enclosure
// bound), the weight of its type times its evidence, and its sign constraint.
struct WindingEventConstraints {
    WindingEventRelation relation;
    // |dot|^power times the distance damping; 1 when both are off.
    double evidence=1;
    // Type weight times evidence: what the equality arcs carry in total.
    double weight=0;
    bool hasSign=false;
    WindingSignConstraint sign{};
    // The one-sided parts of the separation (WindingConstraintOptions::fartherApartFactor).
    std::vector<WindingSignConstraint> bounds;
    // The crossing filter took the crossing out (WindingConstraintOptions::crossingFilter).
    bool filtered=false;
};
// `event` states the crossing as measured: its `distance` is unscaled, and
// its weight, enclosure flag and perpendicular offset are set here.
inline WindingEventConstraints windingEventConstraints(const PackedFiberGeometry& geometry,WindingEvent event,
    const WindingConstraintOptions& options)
{
    WindingEventConstraints out;
    event.distance*=options.windingScale;
    const double dot=event.normalA[0]*event.normalB[0]+event.normalA[1]*event.normalB[1]+event.normalA[2]*event.normalB[2];
    if(options.crossingFilter&&!event.parallel){
        const auto use=fiberCrossingUse(event.crossingFindings,event.crossingAlternative,dot,int(event.signA*event.signB));
        if(!use.used||use.replaced){out.filtered=true;return out;}
    }
    out.evidence=fiberCrossingDotWeight(dot,options.dotWeightPower)
                *(options.dampByWindingDistance?fiberCrossingDistanceFactor(event.distance):1.);
    out.weight=(event.parallel?options.parallelWeight:options.perpendicularWeight)*out.evidence;
    event.weight=out.weight;
    if(!event.parallel&&std::isfinite(event.distance)){
        const int order=int(fiberSignedOrder(event.connector,
            Point{event.normalA[0]*double(event.signA),event.normalA[1]*double(event.signA),event.normalA[2]*double(event.signA)},
            Point{event.normalB[0]*double(event.signB),event.normalB[1]*double(event.signB),event.normalB[2]*double(event.signB)}));
        if(order&&!options.stepCosts){
            const auto& calibration=order*int(event.familyB-event.familyA)>0?options.calibrationVOutside:options.calibrationHOutside;
            event.distance=std::max(0.,calibration.scale*event.distance+calibration.offset);
        }
    }
    event.enclosure=options.enclosures||options.signWeight>0;
    event.perpendicularOffset=options.perpendicularOffset;
    out.relation=windingEventRelation(geometry,event);
    if(options.stepCosts&&!event.parallel&&out.relation.usable&&out.relation.sign!=0){
        const auto& table=*options.stepCosts;auto& r=out.relation;
        const size_t kind=r.sign*int(event.familyB-event.familyA)>0?0:1;
        const auto bin=size_t(std::clamp<int64_t>(int64_t(event.distance/table.binWidth),0,table.bins-1));
        out.evidence=fiberCrossingDotWeight(dot,options.dotWeightPower);
        out.weight=options.perpendicularWeight*out.evidence;
        const double* weights=&table.weights[(kind*size_t(table.bins)+bin)*size_t(table.steps)];
        std::vector<winding_graph_detail::Arc> arcs;
        for(int64_t k=0;k<table.steps;++k){
            if(!(weights[k]>0))continue;
            const int64_t target=r.sign*(table.stepMin+k)+r.transport;
            const double weight=out.weight*weights[k];
            arcs.push_back(event.a>event.b?winding_graph_detail::Arc{event.b,event.a,-target,weight}
                                          :winding_graph_detail::Arc{event.a,event.b,target,weight});
        }
        r.costTable=true;r.hasUpper=arcs.size()>1;r.usable=!arcs.empty();
        if(!arcs.empty())r.lower=arcs[0];
        if(arcs.size()>1)r.upper=arcs[1];
        if(arcs.size()>2)r.extra.assign(arcs.begin()+2,arcs.end());
    }
    if(options.fartherApartFactor<1&&!event.parallel&&out.relation.usable&&out.relation.sign!=0&&!out.relation.costTable){
        auto& r=out.relation;
        const int side=r.sign*(event.a>event.b?-1:1);
        const auto split=[&](winding_graph_detail::Arc& arc){
            out.bounds.push_back({arc.a,arc.b,arc.target,(1-options.fartherApartFactor)*arc.weight,side});
            arc.weight*=options.fartherApartFactor;
        };
        split(r.lower);
        if(r.hasUpper)split(r.upper);
    }
    if(out.relation.hasEnclosure&&options.signWeight>0){
        // The H -> V step is at least its bound (the V fiber outside) or at
        // most it; stated in node order.
        const auto& bound=out.relation.enclosure;
        const bool ordered=bound.h<bound.v;
        out.sign={ordered?bound.h:bound.v,ordered?bound.v:bound.h,ordered?bound.bound:-bound.bound,
                  options.signWeight*out.evidence,ordered?bound.sign:-bound.sign};
        out.hasSign=true;
    }
    return out;
}

// The exact H -> V relations of the pairs bounded from both sides: a pair is
// accepted when its largest lower bound equals its smallest upper bound.
// `bounds` carries one entry per witnessing crossing and is sorted here;
// `accept(step, witness)` receives the accepted step and the index (into the
// sorted bounds) of the crossing that set the lower bound. Rejections are
// counted, never turned into a weaker constraint.
struct WindingEnclosureCounts {size_t pairs{},accepted{},oneSided{},contradictory{},nonunique{};};
template<class Bound,class BoundOf,class Accept>
WindingEnclosureCounts acceptWindingEnclosures(std::vector<Bound>& bounds,BoundOf&& boundOf,Accept&& accept)
{
    using winding_graph_detail::EnclosureBound;
    std::stable_sort(bounds.begin(),bounds.end(),[&](const Bound& x,const Bound& y){
        return winding_graph_detail::enclosureBefore(boundOf(x),boundOf(y));});
    WindingEnclosureCounts counts;
    for(size_t i=0;i<bounds.size();){
        size_t j=i,witness=i;bool lowerSeen=false,upperSeen=false;int64_t lower=0,upper=0;
        const EnclosureBound first=boundOf(bounds[i]);
        for(;j<bounds.size();++j){
            const EnclosureBound bound=boundOf(bounds[j]);
            if(bound.h!=first.h||bound.v!=first.v)break;
            if(bound.sign>0){
                if(!lowerSeen||bound.bound>lower){lower=bound.bound;witness=j;}
                lowerSeen=true;
            }else{upper=upperSeen?std::min(upper,bound.bound):bound.bound;upperSeen=true;}
        }
        ++counts.pairs;
        if(!lowerSeen||!upperSeen)++counts.oneSided;
        else if(lower>upper)++counts.contradictory;
        else if(lower<upper)++counts.nonunique;
        else{accept(lower,witness);++counts.accepted;}
        i=j;
    }
    return counts;
}

inline FiberWindingL1Columns prepareFiberWindingGraph(const PackedFiberGeometry& geometry,
    const FiberWindingGraphInput& in,const FiberWindingGraphOptions& options={})
{
    using namespace winding_graph_detail;
    const size_t m=in.a.count,n=in.family.size();
    if(!options.workers||!options.constraintBatch||!options.readBatch||
       !std::isfinite(options.windingScale)||options.windingScale<0||
       !std::isfinite(options.perpendicularOffset)||!std::isfinite(options.perpendicularWeight)||options.perpendicularWeight<0||
       !std::isfinite(options.parallelWeight)||options.parallelWeight<0||
       !std::isfinite(options.dotWeightPower)||options.dotWeightPower<0||
       !std::isfinite(options.signWeight)||options.signWeight<0)
        throw std::invalid_argument("invalid winding graph preparation settings");
    if(options.perpendicularWeight<=0&&options.parallelWeight<=0&&!options.enclosures&&options.signWeight<=0)
        throw std::invalid_argument("winding graph preparation needs a positive parallel, perpendicular or sign weight, or enclosures");
    if(in.b.count!=m||in.arcA.count!=m||in.arcB.count!=m||in.distance.count!=m||
       in.parallel.size()!=m||in.validA.size()!=m||in.validB.size()!=m||in.validConnector.size()!=m||
       in.normalA.size()!=m*3||in.normalB.size()!=m*3||in.connector.size()!=m*3||in.normalSign.size()!=n||
       in.a.stride<1||in.b.stride<1||in.arcA.stride<1||in.arcB.stride<1||in.distance.stride<1||
       (!in.crossingFindings.empty()&&in.crossingFindings.size()!=m)||in.crossingAlternative.size()!=in.crossingFindings.size())
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
    // The input carries findings exactly when the build filters by them.
    WindingConstraintOptions shared=options;shared.crossingFilter=!in.crossingFindings.empty();
    struct Job {std::vector<Arc> arcs;std::vector<EnclosureBound> enclosures;std::vector<WindingSignConstraint> signs;
                std::vector<int64_t> eventA,eventB;std::vector<double> eventTarget,eventMeasured,eventDistance,eventWeight;
                size_t used{},expanded{},withoutDistance{},crossingFiltered{};};
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
                event.distance=in.distance[k];
                event.parallel=in.parallel[k];
                event.connector=point(in.connector,k);
                event.normalA=point(in.normalA,k);event.normalB=point(in.normalB,k);
                if(!in.crossingFindings.empty()){event.crossingFindings=in.crossingFindings[k];event.crossingAlternative=in.crossingAlternative[k];}
                const auto constraints=windingEventConstraints(geometry,event,shared);
                if(constraints.filtered){++part.crossingFiltered;continue;}
                const auto& relation=constraints.relation;
                part.withoutDistance+=relation.withoutDistance;
                if(relation.hasEnclosure&&options.enclosures)part.enclosures.push_back(relation.enclosure);
                if(constraints.hasSign)part.signs.push_back(constraints.sign);
                part.signs.insert(part.signs.end(),constraints.bounds.begin(),constraints.bounds.end());
                if(!relation.usable)continue;
                batch.push_back(relation.lower);
                if(relation.hasUpper)upper.push_back(relation.upper);
                upper.insert(upper.end(),relation.extra.begin(),relation.extra.end());
                if(options.eventTargets){
                    part.eventA.push_back(relation.lower.a);part.eventB.push_back(relation.lower.b);
                    part.eventTarget.push_back(relation.fractionalTarget);
                    part.eventMeasured.push_back(relation.measuredTarget);
                    part.eventDistance.push_back(event.distance*options.windingScale);part.eventWeight.push_back(constraints.weight);
                }
                ++part.used;
            }
            batch.insert(batch.end(),upper.begin(),upper.end());part.expanded+=batch.size();
            aggregate(batch);part.arcs.insert(part.arcs.end(),batch.begin(),batch.end());
        }
        aggregate(part.arcs);
    });
    FiberWindingL1Columns out;out.read=m;
    for(const auto& part:parts){out.used+=part.used;out.expanded+=part.expanded;out.withoutDistance+=part.withoutDistance;out.crossingFiltered+=part.crossingFiltered;}
    if(options.enclosures){
        // Exact relations need every bound of a pair: gather, sort by pair,
        // then accept pairs whose largest lower bound equals the smallest
        // upper bound. Rejections are counted, never silently dropped arcs.
        std::vector<EnclosureBound> bounds;
        for(auto& part:parts){bounds.insert(bounds.end(),part.enclosures.begin(),part.enclosures.end());std::vector<EnclosureBound>().swap(part.enclosures);}
        std::vector<Arc> relations;
        const auto counts=acceptWindingEnclosures(bounds,[](const EnclosureBound& bound){return bound;},
            [&](int64_t step,size_t witness){relations.push_back(Arc{bounds[witness].h,bounds[witness].v,step,1.});});
        out.enclosurePairs=counts.pairs;out.enclosureOneSided=counts.oneSided;
        out.enclosureContradictory=counts.contradictory;out.enclosureNonunique=counts.nonunique;
        out.enclosureArcs=relations.size();
        out.enclosureA=std::make_unique_for_overwrite<int64_t[]>(out.enclosureArcs);
        out.enclosureB=std::make_unique_for_overwrite<int64_t[]>(out.enclosureArcs);
        out.enclosureTarget=std::make_unique_for_overwrite<int64_t[]>(out.enclosureArcs);
        for(size_t i=0;i<out.enclosureArcs;++i){out.enclosureA[i]=relations[i].a;out.enclosureB[i]=relations[i].b;out.enclosureTarget[i]=relations[i].target;}
    }
    if(options.signWeight>0){
        std::vector<WindingSignConstraint> signs;
        for(auto& part:parts){signs.insert(signs.end(),part.signs.begin(),part.signs.end());std::vector<WindingSignConstraint>().swap(part.signs);}
        out.signConstraints=signs.size();
        const auto key=[](const WindingSignConstraint& x){return std::tuple{x.a,x.b,x.side,x.target};};
        std::stable_sort(signs.begin(),signs.end(),[&](const auto& x,const auto& y){return key(x)<key(y);});
        size_t kept=0;
        for(size_t i=0;i<signs.size();++i){
            if(kept&&key(signs[kept-1])==key(signs[i]))signs[kept-1].weight+=signs[i].weight;
            else signs[kept++]=signs[i];
        }
        signs.resize(kept);out.signArcs=kept;
        out.signA=std::make_unique_for_overwrite<int64_t[]>(kept);out.signB=std::make_unique_for_overwrite<int64_t[]>(kept);
        out.signTarget=std::make_unique_for_overwrite<int64_t[]>(kept);out.signSide=std::make_unique_for_overwrite<int64_t[]>(kept);
        out.signWeight=std::make_unique_for_overwrite<double[]>(kept);
        for(size_t i=0;i<kept;++i){
            out.signA[i]=signs[i].a;out.signB[i]=signs[i].b;out.signTarget[i]=signs[i].target;
            out.signSide[i]=signs[i].side;out.signWeight[i]=signs[i].weight;
        }
    }
    if(options.eventTargets){
        for(const auto& part:parts)out.events+=part.eventA.size();
        out.eventA=std::make_unique_for_overwrite<int64_t[]>(out.events);
        out.eventB=std::make_unique_for_overwrite<int64_t[]>(out.events);
        out.eventTarget=std::make_unique_for_overwrite<double[]>(out.events);
        out.eventMeasured=std::make_unique_for_overwrite<double[]>(out.events);
        out.eventDistance=std::make_unique_for_overwrite<double[]>(out.events);
        out.eventWeight=std::make_unique_for_overwrite<double[]>(out.events);
        size_t at=0;
        for(auto& part:parts){
            for(size_t i=0;i<part.eventA.size();++i,++at){
                out.eventA[at]=part.eventA[i];out.eventB[at]=part.eventB[i];
                out.eventTarget[at]=part.eventTarget[i];out.eventMeasured[at]=part.eventMeasured[i];
                out.eventDistance[at]=part.eventDistance[i];
                out.eventWeight[at]=part.eventWeight[i];
            }
            std::vector<int64_t>().swap(part.eventA);std::vector<int64_t>().swap(part.eventB);
            std::vector<double>().swap(part.eventTarget);std::vector<double>().swap(part.eventMeasured);
            std::vector<double>().swap(part.eventDistance);
            std::vector<double>().swap(part.eventWeight);
        }
    }
    if(!out.used&&!out.enclosureArcs&&!out.signArcs)throw std::invalid_argument("no usable winding constraints");
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
