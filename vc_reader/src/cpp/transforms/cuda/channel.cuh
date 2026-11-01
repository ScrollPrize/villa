#pragma once
#include "resident.cuh"
#include "../shared/precision.hpp"
namespace vcr::transforms {
struct UnsharpMask;struct Defocus;struct MotionBlur;struct ZoomBlur;struct RingingOvershoot;struct GibbsRinging;struct FDA;
}
namespace vcr::transforms::cuda {
template<class T, class Channel> void apply_channels(const T& t, Sample& s, const Parameters& p, Session& session, Channel&& channel) {
    for (const auto& c : p.channels) {
        auto& a = session.arrays.at(c.key); Grid g(a.spatial());
        if (a.bytes == 8) { channel(t,a.channel<double>(c.channel),g,c); continue; }
        auto input = a.channel<float>(c.channel);
        bool wide = channel_wide_parameters(t,c);
        if constexpr (channel_range_guard<T>) {
            const double limit = channel_float_limit(t,c,g.size);
            // Asynchronous sessions keep float32 and flag inputs that would
            // need the float64 path, rather than reading statistics back.
            if (session.async) flag_magnitude(input,limit,session.check_flags(),4);
            else {
                Buffer<int> flag(1); check(memset_async(flag.ptr,0,sizeof(int),vcr::device::stream()));
                flag_magnitude(input,limit,flag.ptr,1);
                int exceeded = 0; flag.download(&exceeded); wide |= exceeded != 0;
            }
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
} // namespace vcr::transforms::cuda
