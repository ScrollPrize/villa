#pragma once
#include "resident.cuh"
namespace vcr::augment {
struct UnsharpMask;struct Defocus;struct MotionBlur;struct ZoomBlur;struct RingingOvershoot;struct GibbsRinging;struct FDA;
}
namespace vcr::augment::cuda {
template<class T, class Channel> void apply_channels(const T& t, Sample& s, const Parameters& p, Session& session, Channel&& channel) {
    for (const auto& c : p.channels) {
        auto& a = session.arrays.at(c.key); Grid g(a.spatial());
        if (a.bytes == 8) { channel(t,a.channel<double>(c.channel),g,c); continue; }
        auto input = a.channel<float>(c.channel);
        bool wide = false;
        if constexpr (std::is_same_v<T,Smear> || std::is_same_v<T,GaussianBlur> || std::is_same_v<T,Sharpen> || std::is_same_v<T,LocalSmoothing> ||
                      std::is_same_v<T,UnsharpMask> || std::is_same_v<T,Defocus> || std::is_same_v<T,MotionBlur> ||
                      std::is_same_v<T,ZoomBlur> || std::is_same_v<T,RingingOvershoot> || std::is_same_v<T,GibbsRinging> || std::is_same_v<T,FDA>) {
            // Protect large-value intermediates; only four statistics,
            // never image data, cross back to the host for this dispatch.
            auto summary = statistics(input);
            double limit = std::numeric_limits<float>::max()/1024.;
            if constexpr(std::is_same_v<T,GaussianBlur> || std::is_same_v<T,Sharpen> || std::is_same_v<T,LocalSmoothing>)limit=std::numeric_limits<float>::max()/64.;
            if constexpr(std::is_same_v<T,Smear>)limit=std::numeric_limits<float>::max()/std::max(t.previous,1)/4.;
            if constexpr(std::is_same_v<T,Defocus>)limit=std::numeric_limits<float>::max()/1000000.;
            if constexpr(std::is_same_v<T,GibbsRinging> || std::is_same_v<T,FDA>)limit=std::numeric_limits<float>::max()/(4.*g.size);
            if constexpr(std::is_same_v<T,FDA>)wide |= t.reference.wide;
            if constexpr (std::is_same_v<T,Sharpen> || std::is_same_v<T,UnsharpMask>) {
                limit /= std::max(1.,std::abs(c.values[0]));
                wide |= std::abs(c.values[0]) > std::numeric_limits<float>::max();
            }
            // Asynchronous sessions keep float32 and flag inputs that would
            // need the float64 path, rather than reading statistics back.
            if (session.async) flag_magnitude(summary,limit,session.check_flags(),4);
            else { Statistics stats; summary.download(&stats); wide |= std::max(std::abs(stats.low),std::abs(stats.high)) > limit; }
        }
        if (wide) {
            Buffer<double> promoted(input.size); View<double> v{promoted.ptr,input.size};
            convert(v,input); channel(t,v,g,c); convert(input,v);
        } else {
            channel(t,input,g,c);
        }
    }
    session.restore_valid(s);
}
} // namespace vcr::augment::cuda
