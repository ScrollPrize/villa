#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct Vignetting { Scalar strength=Uniform{.1,.5}, radius=.8;std::vector<double> center; ChannelOptions options; };
inline ChannelPlan channel_parameters(const Vignetting& t,const Array& a,Random& r) {
    volume_detail::require_image(a);ChannelPlan p;
    double strength=draw(t.strength,r),radius=draw(t.radius,r);probability(strength);require(radius>0,"radius must be positive");auto center=volume_detail::relative_center(t.center,a.shape.size()-1);p.values={strength,radius};p.values.insert(p.values.end(),center.begin(),center.end());
    p.seed=r.next();return p;
}
struct VignettingKernel {
    double strength,radius,center[3];
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;
        T distance=0;for(int a=0;a<D;++a){T q=(c.grid.n[a]>1?T(c.grid.coord(i,a))/T(c.grid.n[a]-1):T(.5))-T(center[a]);distance+=q*q;}return v[i]*(T(1)-T(strength)*clamp(distance/(T(radius)*T(radius)),T(0),T(1)));
    }
};
inline VignettingKernel make_kernel(const Vignetting& t,const ChannelPlan& p) { return {p.values[0],p.values[1],{p.values[2],p.values[3],p.values[4]}}; }
template<class T> inline void channel_apply(const Vignetting& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel<Vignetting>(v,g,make_kernel(t,p));
}
inline void apply(const Vignetting& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
