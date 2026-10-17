#pragma once
#include "../intensity.hpp"
#include "common.cuh"
namespace vcr::augment::cuda {
template<class T, class Channel> void apply_channels(const T& t, Sample& s, const Parameters& p, Channel&& channel) {
    {
        for (const auto& c : p.channels) {
            auto& a = s.at(c.key).array; Grid g(a.spatial());
            if constexpr (std::is_same_v<T,Smear>) {
                auto v = a.channel(c.channel); channel(t,v,g,c); a.channel(c.channel,v);
            } else {
                bool wide = a.bytes == 8;
                if constexpr (std::is_same_v<T,BlankRectangle>) wide |= !t.statistic.empty();
                if constexpr (std::is_same_v<T,GaussianBlur> || std::is_same_v<T,Sharpen> || std::is_same_v<T,LocalSmoothing>) {
                    // Preserve the CPU's wider overflow-protection path.
                    if (a.bytes == 4) {
                        const auto* ptr = static_cast<const float*>(std::as_const(a).data()) + c.channel*g.size;
                        double limit = std::numeric_limits<float>::max()/64.;
                        if constexpr (std::is_same_v<T,Sharpen>) { limit /= std::max(1.,std::abs(c.values[0])); wide |= std::abs(c.values[0]) > std::numeric_limits<float>::max(); }
                        wide |= std::any_of(ptr,ptr+g.size,[&](float x) { return std::abs(double(x)) > limit; });
                    }
                }
                if (wide) { auto v = a.channel(c.channel); channel(t,v,g,c); a.channel(c.channel,v); }
                else {
                    auto* ptr = static_cast<float*>(a.data()) + c.channel*g.size;
                    std::vector<float> v(ptr,ptr+g.size); channel(t,v,g,c); std::copy(v.begin(),v.end(),ptr);
                }
            }
        }
        restore_valid(s);
    }
}

} // namespace vcr::augment::cuda
