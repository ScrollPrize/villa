#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct MotionBlur { Scalar length=Uniform{1,5};std::vector<double> direction;int samples=9; ChannelOptions options; };
inline ChannelPlan channel_parameters(const MotionBlur& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double length=draw(t.length,r);require(length>=0 && length<=1024 && t.samples>=2 && t.samples<=256,"invalid motion blur length or samples");auto direction=volume_detail::sample_direction(t.direction,r);p.values={length};p.values.insert(p.values.end(),direction.begin(),direction.end());
    p.seed=r.next();return p;
}
struct MotionBlurKernel {
    double length,direction[3];int samples;
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        if(length==0)return v[i];double sum=0;for(int j=0;j<samples;++j){double q[3],offset=length*(double(j)/(samples-1)-.5);for(int a=0;a<3;++a)q[a]=double(c.grid.coord(i,a))+direction[a]*offset;sum+=interpolate(v,c.grid,q);}return sum/samples;
    }
};
inline MotionBlurKernel make_kernel(const MotionBlur& t,const ChannelPlan& p) { return {p.values[0],{p.values[1],p.values[2],p.values[3]},t.samples}; }
inline void channel_apply(const MotionBlur& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const MotionBlur& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
