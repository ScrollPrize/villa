#pragma once
#include "GraphInternal.hpp"

namespace vc::constraints::detail {
// Derived geometry only: horizontal traces locate the front of the sheet;
// linked vertical traces can follow its back. Never rewrite either source.
class FiberJunctions {
    struct Knot { double distance; cv::Vec3d offset; };
    std::map<std::string,std::vector<Knot>> profiles_;
    std::map<std::string,std::vector<cv::Vec3d>> points_;
    double spacing_;

    double distance(const Node& node,double column) const {
        if(node.columnArclength.empty())return column*spacing_;
        const int lo=std::clamp(int(std::floor(column)),0,node.cols-2);
        const double t=std::clamp(column-lo,0.0,1.0);
        return node.columnArclength[lo]*(1-t)+node.columnArclength[lo+1]*t;
    }
public:
    explicit FiberJunctions(const Graph& graph,bool enabled=true):spacing_(graph.settings.spacing) {
        if(!enabled)return;
        std::map<std::string,std::map<double,cv::Vec3d>> anchors;
        for(const auto& link:graph.links) {
            const auto [h,v]=horizontalVerticalLink(graph,link);if(!h)continue;
            const auto& horizontal=*graph.nodes.at(h->node).source;
            const auto& vertical=*graph.nodes.at(v->node).source;
            const auto hw=weights(horizontal,*h),vw=weights(vertical,*v);
            double column=0;for(int i=0;i<3;++i)column+=(vw.indices[i]%vertical.cols)*vw.values[i];
            const auto delta=interpolate(horizontal.xyz,hw)-interpolate(vertical.xyz,vw);
            auto [it,inserted]=anchors[v->node].emplace(distance(vertical,column),delta);
            if(!inserted&&cv::norm(it->second-delta)>1e-6)
                throw std::runtime_error("Linked horizontals disagree at the same vertical control: "+v->node);
        }
        const double reach=4*spacing_;
        for(const auto& [id,locations]:anchors) {
            auto& profile=profiles_[id];
            profile.push_back({locations.begin()->first-reach,{0,0,0}});
            for(auto it=locations.begin();it!=locations.end();++it) {
                if(it!=locations.begin()) {
                    const auto previous=std::prev(it);
                    if(it->first-previous->first>2*reach) {
                        profile.push_back({previous->first+reach,{0,0,0}});
                        profile.push_back({it->first-reach,{0,0,0}});
                    }
                }
                profile.push_back({it->first,it->second});
            }
            profile.push_back({locations.rbegin()->first+reach,{0,0,0}});
            const auto& node=*graph.nodes.at(id).source;
            auto& points=points_[id];points=node.xyz;
            for(int x=0;x<node.cols;++x) {
                const auto delta=offset(node,x);
                for(int y=0;y<node.rows;++y) {
                    const int i=y*node.cols+x;
                    if(valid(points[i]))points[i]+=delta;
                }
            }
        }
    }
    cv::Vec3d offset(const Node& node,double column) const {
        const auto found=profiles_.find(node.id);if(found==profiles_.end())return {0,0,0};
        const auto& profile=found->second;const double at=distance(node,column);
        auto hi=std::upper_bound(profile.begin(),profile.end(),at,
            [](double value,const Knot& knot){return value<knot.distance;});
        if(hi==profile.begin()||hi==profile.end())return {0,0,0};
        const auto lo=std::prev(hi);
        const double t=(at-lo->distance)/(hi->distance-lo->distance);
        const double blend=t*t*(3-2*t);
        return lo->offset*(1-blend)+hi->offset*blend;
    }
    const std::vector<cv::Vec3d>& points(const Node& node) const {
        const auto found=points_.find(node.id);
        return found==points_.end()?node.xyz:found->second;
    }
};
} // namespace vc::constraints::detail
