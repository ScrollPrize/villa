#pragma once
#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include "vc/fiber_tracer/FiberChunkTrace.hpp"

namespace vc::fiber_tracer::detail {
struct FiberPathEnd {
    std::array<int,3> key{};
    size_t line=0;
    bool reverse=false; // true=front end, false=back end in stored orientation
    auto operator<=>(const FiberPathEnd&) const = default;
};

// Complete, immutable-after-construction connection graph. Degree is measured
// BEFORE filtering; deleting one ambiguous edge cannot make a rival acceptable.
class FiberContinuationGraph {
public:
    using Edge=std::pair<FiberPathEnd,FiberPathEnd>;
    void add(FiberPathEnd a,FiberPathEnd b){
        if(b<a)std::swap(a,b);
        if(edges_.emplace(a,b).second){++degrees_[a];++degrees_[b];}
    }
    size_t degree(const FiberPathEnd& p) const {
        auto it=degrees_.find(p);return it==degrees_.end()?0:it->second;
    }
    std::string reason(FiberPathEnd a,FiberPathEnd b) const {
        if(degree(a)>1||degree(b)>1)return "chunk_ambiguous_continuation";
        if(b<a)std::swap(a,b);
        if(!edges_.contains({a,b}))return "chunk_unqualified_continuation";
        return {};
    }
    std::string reasonForJoin(const FiberChunkRouteJoin& previous,const FiberChunkRouteJoin& next) const {
        return reason(FiberPathEnd{previous.key,previous.match.line,previous.match.reverse},
            FiberPathEnd{next.key,next.match.line,!next.match.reverse});
    }
    const auto& edges() const {return edges_;}
    const auto& degrees() const {return degrees_;}
private:
    std::set<Edge> edges_;
    std::map<FiberPathEnd,size_t> degrees_;
};
}
