#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include "vc/fiber_tracer/FiberChunkTrace.hpp"

namespace vc::fiber_tracer::detail {
// Ownership of emitted geometry only, in original cached-path arc coordinates.
// A component has one deterministic writer; no locks or first-worker races.
class FiberRouteOwnership {
public:
    explicit FiberRouteOwnership(double length=0):length_(length) {}
    const auto& intervals() const {return spans_;}
    double tolerance() const {return std::max(1e-9,128*std::numeric_limits<double>::epsilon()*std::max(1.,length_));}
    double available(double arc,bool reverse) const {
        const double eps=tolerance();
        if(reverse){
            for(auto i=spans_.rbegin();i!=spans_.rend();++i){
                if(i->first>=arc-eps)continue;
                return i->second>=arc-eps?0:arc-i->second;
            }
        }else for(auto [lo,hi]:spans_){
            if(hi<=arc+eps)continue;
            return lo<=arc+eps?0:lo-arc;
        }
        return std::numeric_limits<double>::infinity();
    }
    void add(double lo,double hi) {
        const double eps=tolerance();
        if(!std::isfinite(lo)||!std::isfinite(hi)||lo < -eps||hi>length_+eps||hi<lo-eps)
            throw std::runtime_error("invalid owned route interval");
        lo=std::clamp(lo,0.,length_);hi=std::clamp(hi,0.,length_);
        if(hi-lo<=eps)return;
        for(auto [a,b]:spans_)if(std::min(hi,b)-std::max(lo,a)>eps)
            throw std::runtime_error("cached path interval exported twice");
        spans_.emplace_back(lo,hi);std::sort(spans_.begin(),spans_.end());
        std::vector<std::pair<double,double>> merged;
        for(auto span:spans_){
            if(!merged.empty()&&span.first<=merged.back().second+eps)merged.back().second=std::max(merged.back().second,span.second);
            else merged.push_back(span);
        }
        spans_=std::move(merged);
    }
    std::optional<std::pair<double,double>> uncovered(double lo,double hi) const {
        if(hi-lo<=tolerance())return {};
        for(auto [a,b]:spans_){
            if(b<=lo+tolerance())continue;
            if(a>lo+tolerance())return std::pair{lo,std::min(a,hi)};
            lo=std::max(lo,b);if(lo>=hi-tolerance())return {};
        }
        if(hi-lo>tolerance())return std::pair{lo,hi};return {};
    }
private:
    double length_;
    std::vector<std::pair<double,double>> spans_;
};

// The storage scope is chosen by the caller: one export component, or one
// independent benchmark segment. Coordinate conversion and alias semantics are
// identical. Optional recording receives only newly owned original-orientation arcs.
inline FiberChunkRouteOwnership fiberRouteOwnershipPolicy(
    std::function<std::pair<FiberRouteOwnership*,double>(const FiberChunkRouteJoin&)> lookup,
    std::function<void(const FiberChunkRouteJoin&,double,double,bool)> record={})
{
    FiberChunkRouteOwnership policy;
    policy.available=[lookup](const auto& j){
        const auto [arcs,length]=lookup(j);
        return arcs->available(j.match.reverse?length-j.match.arc:j.match.arc,j.match.reverse);
    };
    policy.consume=[lookup,record](const auto& j,double begin,double end){
        const auto [arcs,length]=lookup(j);
        const double lo=j.match.reverse?length-end:begin,hi=j.match.reverse?length-begin:end;
        arcs->add(lo,hi);if(record)record(j,lo,hi,false);
    };
    policy.alias=[lookup,record](const auto& j,double begin,double end){
        const auto [arcs,length]=lookup(j);
        const double lo=j.match.reverse?length-end:begin,hi=j.match.reverse?length-begin:end;
        while(auto gap=arcs->uncovered(lo,hi)){
            arcs->add(gap->first,gap->second);
            if(record)record(j,gap->first,gap->second,true);
        }
    };
    return policy;
}
}
