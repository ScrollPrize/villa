#pragma once
#include "../ops/options.hpp"
#include "../cpu/spatial.hpp"
#include "precision.hpp"
#include "channel_requirements.hpp"
#include "histogram.hpp"
#include "intensity_fields.hpp"
namespace vcr::transforms::volume_detail {
template<class T,int D=3> struct ChannelContextT {
    GridND<D> grid;
    Stats input;
    const double* table=nullptr;
    Stats reference{};
    const T* auxiliary=nullptr;
};
using ChannelContext=ChannelContextT<double>;
template<class Transform,class Kernel,class T> void map_channel(std::vector<T>& values,const Grid& grid,const Kernel& kernel,
                                      const double* table=nullptr,Stats reference={},const T* auxiliary=nullptr) {
    Stats input{};
    if constexpr(channel_bounds<Transform>) input=stats<channel_moments<Transform>>(values.data(),values.size());
    with_grid(grid.shape,[&](const auto& g) {
    ChannelContextT<T,std::decay_t<decltype(g)>::dim> context{g,input,table,reference,auxiliary};
    std::vector<T> output(values.size());
    for(std::size_t i=0;i<values.size();++i){output[i]=kernel(values.data(),i,context);require(std::isfinite(output[i]),"transform produced nonfinite values");}
    values.swap(output);
    });
}
// A uint8 channel's value counts and bounds: stats<false> of its float32 values. bins(n)[x] is
// bin() of value x, i.e. bin_indices' result for every voxel holding x.
struct NarrowValues {
    std::array<std::uint32_t,256> counts{};
    int low=0,high=0;
    NarrowValues(const std::uint8_t* v,std::size_t n) {
        // Four interleaved tables avoid store-to-load stalls on runs of one value.
        std::array<std::array<std::uint32_t,256>,4> partial{};
        std::size_t i=0;
        for(;i+4<=n;i+=4){++partial[0][v[i]];++partial[1][v[i+1]];++partial[2][v[i+2]];++partial[3][v[i+3]];}
        for(;i<n;++i)++partial[0][v[i]];
        for(int x=0;x<256;++x)counts[std::size_t(x)]=partial[0][std::size_t(x)]+partial[1][std::size_t(x)]+partial[2][std::size_t(x)]+partial[3][std::size_t(x)];
        low=0;while(low<255 && !counts[std::size_t(low)])++low;
        high=255;while(high>low && !counts[std::size_t(high)])--high;
    }
    Stats bounds() const { return Stats{double(low),double(high),0,0}; }
    std::array<std::uint16_t,256> bins(int bins) const {
        std::array<std::uint16_t,256> result{};
        for(int x=low;x<=high;++x)result[std::size_t(x)]=std::uint16_t(bin(double(x),double(low),double(high),bins));
        return result;
    }
    // A histogram of the bins: count_bins of the voxels' bin indices.
    void histogram(const std::array<std::uint16_t,256>& bin_of,double* table) const {
        for(int x=low;x<=high;++x)table[bin_of[std::size_t(x)]]+=double(counts[std::size_t(x)]);
    }
};
template<class T> VCR_TARGET_CLONES inline void lookup(const std::uint8_t* __restrict in,T* __restrict out,std::size_t n,const T* __restrict table) {
    for(std::size_t i=0;i<n;++i)out[i]=table[in[i]];
}
template<class Transform> void apply_checked_channels(const Transform& t,Sample& s,const Parameters& planned) {
    // Transforms with a narrow_apply read uint8 images directly (see narrow_channels).
    constexpr bool narrow_reader=requires(const ChannelPlan& c,const Grid& g,const std::uint8_t* in,float* out){narrow_apply(t,c,g,in,out);};
    Parameters narrowed;
    if constexpr(narrow_reader)
        narrowed=narrow_channels(s,planned,[&](const ChannelPlan& c,const Grid& g,const std::uint8_t* in,float* out){narrow_apply(t,c,g,in,out);});
    const Parameters& p=narrow_reader?narrowed:planned;
    apply_channels(s,p,[&](auto& values,const Grid& grid,const ChannelPlan& channel) {
        using Value=typename std::decay_t<decltype(values)>::value_type;
        bool wide=false;
        if constexpr(std::is_same_v<Value,float>) {
            wide=channel_wide_parameters(t,channel);
            if constexpr(channel_range_guard<Transform>) {
                const auto limit=channel_float_limit(t,channel,grid.size);
                wide |= magnitude_exceeds(values.data(),values.size(),limit);
            }
        }
        if(wide) {
            std::vector<double> promoted(values.begin(),values.end());
            channel_apply(t,promoted,grid,channel);
            std::copy(promoted.begin(),promoted.end(),values.begin());
        } else channel_apply(t,values,grid,channel);
        const double limit=s.at(channel.key).array.bytes==4 ? std::numeric_limits<float>::max() : std::numeric_limits<double>::max();
        if constexpr(std::is_same_v<Value,float>) {
            // Every finite float is within the float limit: an exponent-bit OR reduction
            // (vectorizable, as in validate) is the same test.
            std::uint32_t invalid=0;
            for(const float value:values)invalid|=std::uint32_t((std::bit_cast<std::uint32_t>(value)&0x7f800000u)==0x7f800000u);
            require(!invalid,"transform output is not finite in the image dtype");
        } else for(double value:values)
            require(std::isfinite(value) && std::abs(value)<=limit,"transform output is not finite in the image dtype");
    });
}
inline void require_image(const Array& a) { require(a.shape.size()==3 || a.shape.size()==4,"expected (C,Y,X) or (C,Z,Y,X)"); }
inline Point sample_direction(const std::vector<double>& fixed,Random& r,std::size_t dim) {
    require(fixed.empty() || fixed.size()==dim,"direction must match spatial dimensions (YX or ZYX)");
    Point direction{};double norm=0;for(std::size_t a=0;a<dim;++a){direction[a]=fixed.empty()?r.normal():fixed[a];norm+=direction[a]*direction[a];}
    require(std::isfinite(norm) && norm>0,"direction must be finite and nonzero");for(auto& x:direction)x/=sqrt(norm);return direction;
}
inline Point relative_center(const std::vector<double>& fixed,std::size_t dim) {
    require(fixed.empty() || fixed.size()==dim,"center must match spatial dimensions (YX or ZYX)");Point center{.5,.5,.5};
    for(std::size_t a=0;a<dim;++a)if(!fixed.empty()){probability(fixed[a]);center[a]=fixed[a];}return center;
}
template<class T> Parameters sample_noise(const T& t,const Sample& s,Random& r) {
    auto p=sample_channels(s,t.options,r,[&](const Array& a,Random& rng){return channel_parameters(t,a,rng);});
    if(t.shared_noise){auto seed=r.next();for(auto& c:p.channels)c.seed=split_seed(seed,c.key);}return p;
}
template<class T> inline std::vector<T> reference_channel(const ReferenceVolume& ref,std::size_t channel) {
    require((ref.shape.size()==3 || ref.shape.size()==4) && product(ref.shape)>0,"reference must be a nonempty (C,Y,X) image or (C,Z,Y,X) volume");
    const auto c=ref.shape[0]==1?0:channel;
    require(c<ref.shape[0],"reference needs one channel or matching image channels");
    return ref.values.channel<T>(c);
}
template<class Function> decltype(auto) visit_reference_channel(const ReferenceVolume& ref,std::size_t channel,Function&& fn) {
    if (ref.wide) return fn(reference_channel<double>(ref,channel));
    return fn(reference_channel<float>(ref,channel));
}
} // namespace vcr::transforms::volume_detail
