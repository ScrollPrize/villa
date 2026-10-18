#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct RingingOvershoot {Scalar cutoff=Uniform{.5,.9},radius=4.;ChannelOptions options;};
inline ChannelPlan channel_parameters(const RingingOvershoot& t,const Array& a,Random& r){
    volume_detail::require_image(a);double cutoff=draw(t.cutoff,r);int radius=draw_integer(t.radius,r);
    require(cutoff>0 && cutoff<=1 && radius>=1 && radius<=64,"cutoff must be (0,1], radius [1,64]");ChannelPlan p;p.values={cutoff,double(radius)};return p;
}
struct RingingOvershootKernel {
    double cutoff;int radius,axis;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;if(cutoff==1)return v[i];T sum=0,total=0;
        for(int j=-radius;j<=radius;++j){double weight=j?sin(pi*cutoff*j)/(pi*j):cutoff;weight*=.5+.5*cos(pi*j/(radius+1));
            auto at=(long long)c.grid.coord(i,axis)+j;at=at<0?0:at>=(long long)c.grid.n[axis]?(long long)c.grid.n[axis]-1:at;
            auto index=i-c.grid.coord(i,axis)*c.grid.stride[axis]+std::size_t(at)*c.grid.stride[axis];sum+=T(weight)*v[index];total+=T(weight);}
        return sum/total;
    }
};
template<int D,class T> inline void ringing_channel(std::vector<T>& v,const Grid& g,const ChannelPlan& p){
    const double cutoff=p.values[0];const int radius=int(p.values[1]);
    if(cutoff==1)return;
    std::vector<T> weights(2*radius+1);double total=0;
    for(int j=-radius;j<=radius;++j){
        double weight=j?sin(volume_detail::pi*cutoff*j)/(volume_detail::pi*j):cutoff;
        weight*=.5+.5*cos(volume_detail::pi*j/(radius+1));
        total+=weights[j+radius]=weight;
    }
    std::vector<T> output(v.size());
    for(int a=0;a<D;++a){
        convolve_axis(v,output,g,a,weights,Boundary::Nearest);
        // Divide after accumulation, matching the original rounding order.
        for(auto& value:output){value/=total;require(std::isfinite(value),"augmentation produced nonfinite values");}
        v.swap(output);
    }
}
template<class T> inline void channel_apply(const RingingOvershoot&,std::vector<T>& v,const Grid& g,const ChannelPlan& p){
    if(g.shape.size()==3) ringing_channel<3>(v,g,p); else ringing_channel<2>(v,g,p);
}
inline void apply(const RingingOvershoot& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
