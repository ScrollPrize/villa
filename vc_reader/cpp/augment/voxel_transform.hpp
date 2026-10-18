#pragma once
#include "spatial.hpp"
#include "precision.hpp"
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
        using Value=typename std::decay_t<decltype(values)>::value_type;
        bool wide=false;
        if constexpr(std::is_same_v<Value,float>) {
            wide=channel_wide_parameters(t,channel);
            if constexpr(channel_range_guard<Transform>) {
                const auto limit=channel_float_limit(t,channel,grid.size);
                wide |= std::any_of(values.begin(),values.end(),[&](float x){return std::abs(double(x))>limit;});
            }
        }
        if(wide) {
            std::vector<double> promoted(values.begin(),values.end());
            channel_apply(t,promoted,grid,channel);
            std::copy(promoted.begin(),promoted.end(),values.begin());
        } else channel_apply(t,values,grid,channel);
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
struct ReferenceVolume {Shape shape;Array values;bool wide=true;};
template<class T> inline std::vector<T> reference_channel(const ReferenceVolume& ref,std::size_t channel) {
    require(ref.shape.size()==4 && product(ref.shape)>0,"reference must be a nonempty (C,Z,Y,X) volume");
    const auto c=ref.shape[0]==1?0:channel;
    require(c<ref.shape[0],"reference needs one channel or matching image channels");
    return ref.values.channel<T>(c);
}
template<class Function> decltype(auto) visit_reference_channel(const ReferenceVolume& ref,std::size_t channel,Function&& fn) {
    if (ref.wide) return fn(reference_channel<double>(ref,channel));
    return fn(reference_channel<float>(ref,channel));
}
} // namespace vcr::augment::volume_detail
