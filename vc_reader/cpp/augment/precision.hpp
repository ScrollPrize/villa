#pragma once
#include "core.hpp"

namespace vcr::augment {
struct OrientedBlur; struct GaussianBlur; struct Sharpen; struct LocalSmoothing; struct Smear;
struct UnsharpMask; struct Defocus; struct MotionBlur; struct ZoomBlur;
struct RingingOvershoot; struct GibbsRinging; struct FDA;
struct Illumination; struct Vignetting;

template<class T> constexpr bool channel_range_guard =
    std::is_same_v<T,OrientedBlur> || std::is_same_v<T,GaussianBlur> || std::is_same_v<T,Sharpen> || std::is_same_v<T,LocalSmoothing> ||
    std::is_same_v<T,Smear> || std::is_same_v<T,UnsharpMask> || std::is_same_v<T,Defocus> ||
    std::is_same_v<T,MotionBlur> || std::is_same_v<T,ZoomBlur> || std::is_same_v<T,RingingOvershoot> ||
    std::is_same_v<T,GibbsRinging> || std::is_same_v<T,FDA>;

// These are bounds on unnormalized filter sums, FFT coefficients and sharpening
// intermediates, not accuracy knobs. CPU and CUDA use the same overflow policy.
template<class T> double channel_float_limit(const T& t,const ChannelPlan& c,std::size_t size) {
    double limit=std::numeric_limits<float>::max()/1024.;
    if constexpr(std::is_same_v<T,GaussianBlur> || std::is_same_v<T,Sharpen> || std::is_same_v<T,LocalSmoothing>)
        limit=std::numeric_limits<float>::max()/64.;
    if constexpr(std::is_same_v<T,Smear>) limit=std::numeric_limits<float>::max()/std::max(t.previous,1)/4.;
    if constexpr(std::is_same_v<T,Defocus>) limit=std::numeric_limits<float>::max()/1000000.;
    if constexpr(std::is_same_v<T,GibbsRinging> || std::is_same_v<T,FDA>) limit=std::numeric_limits<float>::max()/(4.*size);
    if constexpr(std::is_same_v<T,Sharpen> || std::is_same_v<T,UnsharpMask>) limit/=std::max(1.,std::abs(c.values[0]));
    return limit;
}
template<class T> bool channel_wide_parameters(const T& t,const ChannelPlan& c) {
    for(double x:c.values)
        if(std::abs(x)>std::numeric_limits<float>::max() || (x!=0 && std::abs(x)<std::numeric_limits<float>::denorm_min())) return true;
    if constexpr(std::is_same_v<T,Illumination> || std::is_same_v<T,Vignetting>)
        if(c.values[1]<std::sqrt(std::numeric_limits<float>::min())) return true;
    if constexpr(std::is_same_v<T,FDA>) if(t.reference.wide) return true;
    return false;
}
}
