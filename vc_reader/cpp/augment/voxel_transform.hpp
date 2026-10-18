#pragma once
#include "spatial.hpp"
#include "histogram.hpp"
#include "intensity_fields.hpp"
namespace vcr::augment::volume_detail {
template<class T> struct ChannelContextT {
    Grid3 grid;
    Stats input;
    const double* table=nullptr;
    Stats reference{};
    const T* auxiliary=nullptr;
};
using ChannelContext=ChannelContextT<double>;
template<class Kernel,class T> void map_channel(std::vector<T>& values,const Grid& grid,const Kernel& kernel,
                                      const double* table=nullptr,Stats reference={},const T* auxiliary=nullptr) {
    ChannelContextT<T> context{Grid3(grid.shape),stats(values.data(),values.size()),table,reference,auxiliary};
    std::vector<T> output(values.size());
    for(std::size_t i=0;i<values.size();++i){output[i]=kernel(values.data(),i,context);require(std::isfinite(output[i]),"augmentation produced nonfinite values");}
    values.swap(output);
}
template<class Transform> void apply_checked_channels(const Transform& t,Sample& s,const Parameters& p) {
    apply_channels(s,p,[&](auto& values,const Grid& grid,const ChannelPlan& channel) {
        channel_apply(t,values,grid,channel);
        const double limit=s.at(channel.key).array.bytes==4 ? std::numeric_limits<float>::max() : std::numeric_limits<double>::max();
        for(double value:values)
            require(std::isfinite(value) && std::abs(value)<=limit,"augmentation output is not finite in the image dtype");
    });
}
inline void require_volume(const Array& a) { Grid3 grid(a.spatial()); }
inline Point sample_direction(const std::vector<double>& fixed,Random& r) {
    require(fixed.empty() || fixed.size()==3,"direction needs three ZYX components");
    Point direction{};double norm=0;for(int a=0;a<3;++a){direction[a]=fixed.empty()?r.normal():fixed[a];norm+=direction[a]*direction[a];}
    require(std::isfinite(norm) && norm>0,"direction must be finite and nonzero");for(auto& x:direction)x/=sqrt(norm);return direction;
}
inline Point relative_center(const std::vector<double>& fixed) {
    require(fixed.empty() || fixed.size()==3,"center needs three relative ZYX coordinates");Point center{.5,.5,.5};
    for(int a=0;a<3;++a)if(!fixed.empty()){probability(fixed[a]);center[a]=fixed[a];}return center;
}
template<class T> Parameters sample_noise(const T& t,const Sample& s,Random& r) {
    auto p=sample_channels(s,t.options,r,[&](const Array& a,Random& rng){return channel_parameters(t,a,rng);});
    if(t.shared_noise){auto seed=r.next();for(auto& c:p.channels)c.seed=split_seed(seed,c.key);}return p;
}
struct ReferenceVolume {Shape shape;std::vector<double> values;bool wide=true;};
inline std::vector<double> reference_channel(const ReferenceVolume& ref,std::size_t channel) {
    require(ref.shape.size()==4 && product(ref.shape)>0,"reference must be a nonempty (C,Z,Y,X) volume");
    auto n=ref.values.size()/ref.shape[0],c=ref.shape[0]==1?0:channel;
    require(c<ref.shape[0],"reference needs one channel or matching image channels");return {ref.values.begin()+c*n,ref.values.begin()+(c+1)*n};
}
} // namespace vcr::augment::volume_detail
