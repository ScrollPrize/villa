#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct ZoomBlur { Scalar strength=Uniform{0,.2};std::vector<double> center;int samples=9; ChannelOptions options; };
inline ChannelPlan channel_parameters(const ZoomBlur& t,const Array& a,Random& r) {
    volume_detail::require_image(a);ChannelPlan p;
    double strength=draw(t.strength,r);require(strength>=0 && strength<=64 && t.samples>=2 && t.samples<=256,"invalid zoom strength or samples");auto center=volume_detail::relative_center(t.center,a.shape.size()-1);p.values={strength};p.values.insert(p.values.end(),center.begin(),center.end());
    p.seed=r.next();return p;
}
struct ZoomBlurKernel {
    double strength,center[3];int samples;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;
        if(strength==0)return v[i];T sum=0;for(int j=0;j<samples;++j){double q[3],factor=1+strength*double(j)/(samples-1);for(int a=0;a<D;++a){double origin=center[a]*(c.grid.n[a]-1);q[a]=origin+(double(c.grid.coord(i,a))-origin)/factor;}sum+=interpolate(v,c.grid,q);}return sum/samples;
    }
};
inline ZoomBlurKernel make_kernel(const ZoomBlur& t,const ChannelPlan& p) { return {p.values[0],{p.values[1],p.values[2],p.values[3]},t.samples}; }
template<class T> inline void channel_apply(const ZoomBlur& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel<ZoomBlur>(v,g,make_kernel(t,p));
}
inline void apply(const ZoomBlur& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
