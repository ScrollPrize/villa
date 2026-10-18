#pragma once
#include "resident.cuh"
namespace vcr::augment::cuda {
std::shared_ptr<Buffer<float>> local_field(const LocalOptions&, const Grid&, Random&, bool centered);
template<class T> Parameters sample_local(const T& t, const Sample& s, Random& r, bool record) {
    return sample_channels(s,t.options,r,[&](const Array& a, Random& rng) {
        ChannelPlan c;
        if constexpr (std::is_same_v<T,BrightnessGradient>) c.values = {draw(t.strength,rng)};
        if constexpr (std::is_same_v<T,LocalContrast>) c.values = {draw(t.factor,rng)};
        if constexpr (std::is_same_v<T,LocalGamma>) {
            auto gamma = draw(t.gamma,rng); require(gamma > 0,"gamma must be positive"); c.values = {gamma};
        }
        if constexpr (std::is_same_v<T,LocalSmoothing>) {
            auto strength = draw(t.strength,rng); probability(strength);
            auto sigma = draw(t.sigma,rng); require(sigma >= 0,"smoothing sigma must be nonnegative"); c.values = {strength,sigma};
        }
        bool centered = false;
        if constexpr (std::is_same_v<T,BrightnessGradient>) centered = t.mean_centered;
        auto field = local_field(t.local,Grid(a.spatial()),rng,centered);
        c.device_field = field;
        if (record) { c.field.resize(field->size); field->download(c.field.data()); }
        return c;
    });
}
Parameters sample_elastic(const Elastic&, const Sample&, Random&, bool record);
}
