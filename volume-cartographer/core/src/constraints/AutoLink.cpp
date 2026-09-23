#include "vc/constraints/AutoLink.hpp"
#include "GraphInternal.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include <algorithm>
#include <future>
#include <limits>
#include <thread>

namespace vc::constraints {
namespace {
struct Surface {
    std::shared_ptr<const Node> node;
    std::shared_ptr<QuadSurface> quad;
    SurfacePatchIndex index;
    std::vector<cv::Vec3d> vertexNormal;   // area-weighted, per grid vertex
    std::vector<int> validVertices;
    cv::Vec3d low=cv::Vec3d::all(std::numeric_limits<double>::infinity()),high=cv::Vec3d::all(-std::numeric_limits<double>::infinity());
    explicit Surface(std::shared_ptr<const Node> n):node(std::move(n)) {
        quad=detail::sourceSurface(*node);index.setSamplingStride(1);index.rebuild({quad});
        vertexNormal.assign(node->xyz.size(),cv::Vec3d{0,0,0});
        for(const auto& t:node->triangles) {
            const auto n=(node->xyz[t[1]]-node->xyz[t[0]]).cross(node->xyz[t[2]]-node->xyz[t[0]]);
            for(int k=0;k<3;++k)vertexNormal[t[k]]+=n;
        }
        for(auto& n:vertexNormal)n=detail::unit(n);
        for(size_t i=0;i<node->xyz.size();++i)if(detail::valid(node->xyz[i])) {
            validVertices.push_back(int(i));
            for(int d=0;d<3;++d){low[d]=std::min(low[d],node->xyz[i][d]);high[d]=std::max(high[d],node->xyz[i][d]);}
        }
    }
    bool inBox(const cv::Vec3d& p,double margin) const {
        for(int d=0;d<3;++d)if(p[d]<low[d]-margin||p[d]>high[d]+margin)return false;
        return true;
    }
    struct Hit { double distance; double normalDot; cv::Vec2d grid; };
    // Nearest surface point within tolerance, with |cos| between the query normal and the hit triangle.
    std::optional<Hit> nearest(const cv::Vec3d& p,const cv::Vec3d& normal,double tolerance) const {
        SurfacePatchIndex::PointQuery q;q.worldPoint=cv::Vec3f(p);q.tolerance=float(tolerance);
        const auto hit=index.locate(q);if(!hit)return {};
        const cv::Vec2d grid(quad->ptrToGrid(hit->ptr));
        detail::Weights w;
        try{w=detail::weights(*node,grid);}catch(const std::exception&){return {};}
        const auto n=detail::unit((node->xyz[w.indices[1]]-node->xyz[w.indices[0]]).cross(node->xyz[w.indices[2]]-node->xyz[w.indices[0]]));
        return Hit{double(hit->distance),std::abs(n.dot(normal)),grid};
    }
};
// Per-vertex best relation of one patch to a growing reference set.
struct Relation {
    std::vector<double> distance,normalDot;
    explicit Relation(size_t n):distance(n,std::numeric_limits<double>::infinity()),normalDot(n,0) {}
    void update(const Surface& self,const Surface& other,double margin,double tolerance,const std::atomic_bool* cancel) {
        if(!self.inBox(other.low,margin)&&!other.inBox(self.low,margin)) {
            // Cheap reject when the boxes are disjoint.
            bool overlap=true;
            for(int d=0;d<3;++d)if(self.high[d]+margin<other.low[d]||other.high[d]+margin<self.low[d])overlap=false;
            if(!overlap)return;
        }
        for(size_t k=0;k<self.validVertices.size();++k) {
            if((k&1023)==0)detail::cancelled(cancel);
            const int i=self.validVertices[k];const auto& p=self.node->xyz[i];
            if(!other.inBox(p,margin))continue;
            const auto hit=other.nearest(p,self.vertexNormal[i],tolerance);
            if(hit&&hit->distance<distance[k]){distance[k]=hit->distance;normalDot[k]=hit->normalDot;}
        }
    }
    std::pair<double,double> fractions(const AutoLinkSettings& s) const {
        if(distance.empty())return {0,0};
        size_t same=0,adjacent=0;
        for(size_t k=0;k<distance.size();++k) {
            if(distance[k]<=s.sameSheet&&normalDot[k]>s.sameNormal)++same;
            else if(distance[k]>=s.adjacentLow&&distance[k]<=s.adjacentHigh&&normalDot[k]>s.adjacentNormal)++adjacent;
        }
        return {double(same)/distance.size(),double(adjacent)/distance.size()};
    }
};
double contactLength(const Node& fiber,const Candidate& c,const AutoLinkSettings& s) {
    std::vector<int64_t> lines;
    for(const auto& l:c.contacts){const auto& at=l.a.node==fiber.id?l.a:l.b;if(at.lineIndex>=0&&size_t(at.lineIndex)<fiber.lineArclength.size())lines.push_back(at.lineIndex);}
    if(lines.empty())return 0;
    std::sort(lines.begin(),lines.end());lines.erase(std::unique(lines.begin(),lines.end()),lines.end());
    // Median line spacing stands in for one sample's own length.
    std::vector<double> steps;for(size_t i=1;i<fiber.lineArclength.size();++i)steps.push_back(fiber.lineArclength[i]-fiber.lineArclength[i-1]);
    double step=2;if(!steps.empty()){std::nth_element(steps.begin(),steps.begin()+steps.size()/2,steps.end());step=steps[steps.size()/2];}
    double total=0;size_t start=0;
    for(size_t i=1;i<=lines.size();++i) {
        if(i==lines.size()||fiber.lineArclength[lines[i]]-fiber.lineArclength[lines[i-1]]>s.contactGap) {
            total+=fiber.lineArclength[lines[i-1]]-fiber.lineArclength[lines[start]]+step;start=i;
        }
    }
    return total;
}
std::pair<double,double> stripAgreement(const Surface& strip,const Surface& patch,const AutoLinkSettings& s,const std::atomic_bool* cancel) {
    const int rows=strip.node->rows,center=rows/2;
    double weight=0,same=0,adjacent=0;size_t selected=0;
    for(size_t k=0;k<patch.validVertices.size();++k) {
        if((k&1023)==0)detail::cancelled(cancel);
        const int i=patch.validVertices[k];const auto& p=patch.node->xyz[i];
        if(!strip.inBox(p,s.stripMargin))continue;
        ++selected;
        const auto hit=strip.nearest(p,patch.vertexNormal[i],s.adjacentHigh);
        // Unmatched vertices are farther than any sheet of interest; they
        // still count toward the total, at the outermost trust level.
        int row=hit?int(std::lround(hit->grid[1])):0;
        const int k_=hit?std::abs(std::clamp(row,0,rows-1)-center):int(s.stripWeights.size());
        const double w=k_<int(s.stripWeights.size())?s.stripWeights[k_]:.05;
        weight+=w;
        if(!hit)continue;
        if(hit->distance<=s.sameSheet&&hit->normalDot>s.sameNormal)same+=w;
        else if(hit->distance>=s.adjacentLow&&hit->distance<=s.adjacentHigh&&hit->normalDot>s.adjacentNormal)adjacent+=w;
    }
    if(selected<5||weight<=0)return {0,0};
    return {same/weight,adjacent/weight};
}
}
using Json=nlohmann::json;
Json autoLinkSettingsToJson(const AutoLinkSettings& s) {
    return {{"contact_gap",s.contactGap},{"anchor_contact",s.anchorContact},{"same_sheet",s.sameSheet},{"adjacent_low",s.adjacentLow},{"adjacent_high",s.adjacentHigh},
            {"same_normal",s.sameNormal},{"adjacent_normal",s.adjacentNormal},{"grow_same",s.growSame},{"grow_ratio",s.growRatio},
            {"strip_weights",s.stripWeights},{"strip_gate_ratio",s.stripGateRatio},{"strip_gate_margin",s.stripGateMargin},{"strip_margin",s.stripMargin},{"max_rounds",s.maxRounds}};
}
AutoLinkSettings autoLinkSettingsFromJson(const Json& j) {
    AutoLinkSettings s;
    s.contactGap=j.value("contact_gap",s.contactGap);s.anchorContact=j.value("anchor_contact",s.anchorContact);
    s.sameSheet=j.value("same_sheet",s.sameSheet);s.adjacentLow=j.value("adjacent_low",s.adjacentLow);s.adjacentHigh=j.value("adjacent_high",s.adjacentHigh);
    s.sameNormal=j.value("same_normal",s.sameNormal);s.adjacentNormal=j.value("adjacent_normal",s.adjacentNormal);
    s.growSame=j.value("grow_same",s.growSame);s.growRatio=j.value("grow_ratio",s.growRatio);
    if(j.contains("strip_weights")){const auto& w=j.at("strip_weights");if(w.size()!=4)throw std::runtime_error("strip_weights needs four values");for(size_t i=0;i<4;++i)s.stripWeights[i]=w[i].get<double>();}
    s.stripMargin=j.value("strip_margin",s.stripMargin);s.maxRounds=j.value("max_rounds",s.maxRounds);
    s.stripGateRatio=j.value("strip_gate_ratio",s.stripGateRatio);s.stripGateMargin=j.value("strip_gate_margin",s.stripGateMargin);
    if(s.anchorContact<0||s.sameSheet<=0||s.adjacentLow<=s.sameSheet||s.adjacentHigh<=s.adjacentLow||s.growSame<0||s.growRatio<0||s.maxRounds<1||s.stripGateRatio<=0||s.stripGateMargin<0)
        throw std::runtime_error("Invalid auto-link settings");
    return s;
}
AutoLinkResult autoLinkFiber(const Node& fiber,const std::vector<Candidate>& proposals,const PatchLoader& load,
                             const AutoLinkSettings& s,const std::atomic_bool* cancel,Progress progress) {
    if(!fiber.fiber)throw std::runtime_error("Auto-link needs a fiber strip node");
    AutoLinkResult result;
    // One entry per patch; legacy split proposals contribute all their contacts.
    std::map<std::string,size_t> index;
    std::vector<Candidate> merged;
    for(const auto& c:proposals) {
        if(c.fiber)continue;
        auto [it,inserted]=index.emplace(c.id,merged.size());
        if(inserted)merged.push_back(c);else merged[it->second].contacts.insert(merged[it->second].contacts.end(),c.contacts.begin(),c.contacts.end());
    }
    if(progress)progress("Loading "+std::to_string(merged.size())+" candidate patches");
    Surface strip(std::make_shared<Node>(fiber));
    std::vector<std::unique_ptr<Surface>> surfaces(merged.size());
    {
        const size_t width=std::min<size_t>({merged.size(),std::max(1u,std::thread::hardware_concurrency()),8});
        for(size_t first=0;first<merged.size();first+=width) {
            detail::cancelled(cancel);
            std::vector<std::future<std::unique_ptr<Surface>>> futures;
            for(size_t j=first;j<std::min(first+width,merged.size());++j)
                futures.push_back(std::async(std::launch::async,[&,j]{auto n=load(merged[j].id);return n?std::make_unique<Surface>(n):nullptr;}));
            for(size_t j=first;j<std::min(first+width,merged.size());++j)surfaces[j]=futures[j-first].get();
        }
    }
    result.candidates.resize(merged.size());
    for(size_t i=0;i<merged.size();++i) {
        auto& c=result.candidates[i];c.id=merged[i].id;
        if(!surfaces[i])continue;
        c.contact=contactLength(fiber,merged[i],s);
    }
    if(progress)progress("Checking strip agreement");
    for(size_t i=0;i<merged.size();++i) {
        auto& c=result.candidates[i];if(!surfaces[i]||c.contact<s.anchorContact)continue;
        std::tie(c.stripSame,c.stripAdjacent)=stripAgreement(strip,*surfaces[i],s,cancel);
        c.anchor=c.stripSame>s.stripGateRatio*c.stripAdjacent+s.stripGateMargin&&c.stripSame>0;
        if(c.anchor){c.linked=true;c.round=0;++result.anchors;++result.linked;}
    }
    // Growth: every unlinked candidate tracks its per-vertex nearest linked surface.
    std::vector<std::unique_ptr<Relation>> relations(merged.size());
    for(size_t i=0;i<merged.size();++i)if(surfaces[i]&&!result.candidates[i].linked)relations[i]=std::make_unique<Relation>(surfaces[i]->validVertices.size());
    std::vector<size_t> fresh;for(size_t i=0;i<merged.size();++i)if(result.candidates[i].linked)fresh.push_back(i);
    for(int round=1;round<=s.maxRounds&&!fresh.empty();++round) {
        if(progress)progress("Growing from "+std::to_string(result.linked)+" linked patches (round "+std::to_string(round)+")");
        std::vector<size_t> pending;for(size_t i=0;i<merged.size();++i)if(relations[i]&&!result.candidates[i].linked)pending.push_back(i);
        const size_t width=std::min<size_t>({pending.size(),std::max(1u,std::thread::hardware_concurrency()),8});
        for(size_t first=0;first<pending.size();first+=width) {
            detail::cancelled(cancel);
            std::vector<std::future<void>> futures;
            for(size_t j=first;j<std::min(first+width,pending.size());++j)
                futures.push_back(std::async(std::launch::async,[&,i=pending[j]]{for(size_t f:fresh)relations[i]->update(*surfaces[i],*surfaces[f],s.stripMargin,s.adjacentHigh,cancel);}));
            for(auto& f:futures)f.get();
        }
        fresh.clear();
        for(size_t i:pending) {
            auto& c=result.candidates[i];
            std::tie(c.linkedSame,c.linkedAdjacent)=relations[i]->fractions(s);
            if(c.linkedSame>=s.growSame&&c.linkedSame>s.growRatio*c.linkedAdjacent){c.linked=true;c.round=round;++result.linked;fresh.push_back(i);}
        }
    }
    return result;
}
} // namespace vc::constraints
