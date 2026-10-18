#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct Illumination { Scalar strength=Uniform{-.3,.3}, radius=.4;int mode=0;std::vector<double> direction,center; ChannelOptions options; };
inline ChannelPlan channel_parameters(const Illumination& t,const Array& a,Random& r) {
    volume_detail::require_image(a);ChannelPlan p;
    double strength=draw(t.strength,r),radius=draw(t.radius,r);require(fabs(strength)<=1 && radius>0,"illumination strength must be [-1,1], radius positive");auto direction=volume_detail::sample_direction(t.direction,r,a.shape.size()-1),center=volume_detail::relative_center(t.center,a.shape.size()-1);if(t.mode==1 && t.center.empty())for(std::size_t j=0;j<a.shape.size()-1;++j)center[j]=double(r.index(2));p.values={strength,radius};p.values.insert(p.values.end(),direction.begin(),direction.end());p.values.insert(p.values.end(),center.begin(),center.end());
    p.seed=r.next();return p;
}
struct IlluminationKernel {
    double strength,radius;int mode;double direction[3],center[3];
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;
        T field=0;
        if(mode==0){T denom=0;for(int a=0;a<D;++a){field+=T(direction[a])*(c.grid.n[a]>1?2*T(c.grid.coord(i,a))/T(c.grid.n[a]-1)-1:0);denom+=fabs(T(direction[a]));}field/=denom;}
        else {T distance=0;for(int a=0;a<D;++a){T q=(c.grid.n[a]>1?T(c.grid.coord(i,a))/T(c.grid.n[a]-1):T(.5))-T(center[a]);distance+=q*q;}field=mode==1?1-sqrt(distance/T(D)):exp(-distance/(T(2)*T(radius)*T(radius)));}
        return v[i]*(T(1)+T(strength)*field);
    }
};
inline IlluminationKernel make_kernel(const Illumination& t,const ChannelPlan& p) { return {p.values[0],p.values[1],t.mode,{p.values[2],p.values[3],p.values[4]},{p.values[5],p.values[6],p.values[7]}}; }
template<class T> inline void channel_apply(const Illumination& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel<Illumination>(v,g,make_kernel(t,p));
}
inline void apply(const Illumination& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
