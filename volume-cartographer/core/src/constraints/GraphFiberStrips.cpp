#include "GraphInternal.hpp"
#include "vc/constraints/FiberReview.hpp"
#include "vc/lasagna/RobustLineStrip.hpp"

namespace vc::constraints {
namespace {
double columnAt(const std::vector<double>& arc,double position) {
    auto it=std::lower_bound(arc.begin(),arc.end(),position);
    if(it==arc.begin())return 0.;
    if(it==arc.end())return double(arc.size()-1);
    const size_t hi=size_t(it-arc.begin());
    return hi-1+(position-arc[hi-1])/std::max(arc[hi]-arc[hi-1],1e-12);
}
double arcAt(const std::vector<double>& arc,double column) {
    column=std::clamp(column,0.,double(arc.size()-1));
    const size_t lo=std::min(size_t(column),arc.size()-2);const double f=column-lo;
    return arc[lo]*(1-f)+arc[lo+1]*f;
}
}
void detail::remapFiberLocation(Location& location,const Node& oldStrip,const Node& newStrip) {
    if(location.node!=oldStrip.id)return;
    if(oldStrip.rows==newStrip.rows&&oldStrip.cols==newStrip.cols&&oldStrip.columnArclength==newStrip.columnArclength)return;
    if(location.lineIndex>=0){location.grid=fiberLineLocation(newStrip,size_t(location.lineIndex)).grid;return;}
    location.grid={columnAt(newStrip.columnArclength,arcAt(oldStrip.columnArclength,location.grid[0])),
                   location.grid[1]*(newStrip.rows-1)/double(oldStrip.rows-1)};
}
void detail::replaceFiberStrip(Graph& graph,const std::shared_ptr<const Node>& replacement) {
    auto found=graph.nodes.find(replacement->id);if(found==graph.nodes.end())return;
    const auto old=found->second.source;if(old==replacement)return;
    if(old->rows==replacement->rows&&old->cols==replacement->cols&&old->columnArclength==replacement->columnArclength) {
        found->second.source=replacement;return; // preserve stored UV and links bit-for-bit
    }
    std::vector<cv::Vec2d> uv;uv.reserve(replacement->xyz.size());
    for(int row=0;row<replacement->rows;++row)for(int col=0;col<replacement->cols;++col) {
        const cv::Vec2d at{columnAt(old->columnArclength,replacement->columnArclength[col]),row*(old->rows-1)/double(replacement->rows-1)};
        uv.push_back(interpolate(found->second.uv,weights(*old,at)));
    }
    found->second={replacement,std::move(uv)};
    for(auto& link:graph.links){remapFiberLocation(link.a,*old,*replacement);remapFiberLocation(link.b,*old,*replacement);}
}
bool regenerateFiberStrips(Graph& graph,const lasagna::NormalSampler& sampler,const std::atomic_bool* cancel) {
    Graph candidate=graph;bool changed=false;
    for(const auto& [id,p]:graph.nodes)if(p.source->fiber&&p.source->stripGeneration<lasagna::kRobustLineStripVersion) {
        detail::cancelled(cancel);const auto& n=*p.source;
        Fiber fiber;fiber.id=id;fiber.path=n.path;fiber.direction=n.direction;fiber.line=n.line;fiber.controls=n.controls;
        detail::replaceFiberStrip(candidate,makeFiberNode(fiber,graph.settings,sampler,cancel));changed=true;
    }
    if(changed)graph=std::move(candidate);return changed;
}
bool regenerateReviewFiber(ReviewFiber& fiber,const lasagna::NormalSampler& sampler,const std::atomic_bool* cancel) {
    if(fiber.base.nodes.empty())return false;
    ReviewFiber candidate=fiber;const auto previous=fiber.base.nodes.at(fiber.base.seed).source;
    if(!regenerateFiberStrips(candidate.base,sampler,cancel))return false;
    const auto replacement=candidate.base.nodes.at(candidate.base.seed).source;
    if(candidate.pending)detail::replaceFiberStrip(*candidate.pending,replacement);
    for(auto& [id,decision]:candidate.decisions)detail::replaceFiberStrip(decision.alignment,replacement);
    for(auto& proposal:candidate.proposals)for(auto& link:proposal.contacts) {
        detail::remapFiberLocation(link.a,*previous,*replacement);detail::remapFiberLocation(link.b,*previous,*replacement);
    }
    fiber=std::move(candidate);return true;
}
} // namespace vc::constraints
