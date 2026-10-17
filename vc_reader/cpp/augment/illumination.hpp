#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct Illumination { Scalar strength=Uniform{-.3,.3}, radius=.4;int mode=0;std::vector<double> direction,center; ChannelOptions options; };
inline ChannelPlan channel_parameters(const Illumination& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double strength=draw(t.strength,r),radius=draw(t.radius,r);require(fabs(strength)<=1 && radius>0,"illumination strength must be [-1,1], radius positive");auto direction=volume_detail::sample_direction(t.direction,r),center=volume_detail::relative_center(t.center);if(t.mode==1 && t.center.empty())for(auto& x:center)x=double(r.index(2));p.values={strength,radius};p.values.insert(p.values.end(),direction.begin(),direction.end());p.values.insert(p.values.end(),center.begin(),center.end());
    p.seed=r.next();return p;
}
struct IlluminationKernel {
    double strength,radius;int mode;double direction[3],center[3];
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        double field=0;
        if(mode==0){double denom=0;for(int a=0;a<3;++a){field+=direction[a]*(c.grid.n[a]>1?2*double(c.grid.coord(i,a))/double(c.grid.n[a]-1)-1:0);denom+=fabs(direction[a]);}field/=denom;}
        else {double distance=0;for(int a=0;a<3;++a){double q=(c.grid.n[a]>1?double(c.grid.coord(i,a))/double(c.grid.n[a]-1):.5)-center[a];distance+=q*q;}field=mode==1?1-sqrt(distance/3):exp(-distance/(2*radius*radius));}
        return v[i]*(1+strength*field);
    }
};
inline IlluminationKernel make_kernel(const Illumination& t,const ChannelPlan& p) { return {p.values[0],p.values[1],t.mode,{p.values[2],p.values[3],p.values[4]},{p.values[5],p.values[6],p.values[7]}}; }
inline void channel_apply(const Illumination& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const Illumination& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
