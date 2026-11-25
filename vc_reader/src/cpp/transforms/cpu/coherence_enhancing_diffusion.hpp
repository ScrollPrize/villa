#pragma once
#include "../shared/coherence_enhancing_diffusion.hpp"

namespace vcr::transforms {
inline Parameters sample(const CoherenceEnhancingDiffusion& t, const Sample& s, Random& r) {
    return sample_channels(s, t.options, r, [](const Array&, Random&) { return ChannelPlan{}; });
}
inline void apply(const CoherenceEnhancingDiffusion& t, Sample& s, const Parameters& p) {
    if (t.num_steps == 0) return;
    using namespace structure_tensor_detail;
    const auto st = diffusion_detail::tensor_options(t);
    for (const auto& c : p.channels) {
        auto& a = s.at(c.key).array; Grid g(a.spatial());
        const int d = int(g.shape.size()), count = component_count(d);
        const auto integration = integration_taps(st,d);
        volume_detail::with_grid(g.shape,[&](const auto& layout) {
            const auto run = [&](auto tag) {
                using T = decltype(tag);
                auto input = a.channel<T>(c.channel);
                std::vector<T> tensor(count*g.size), component(g.size), scratch(g.size), flux(d*g.size);
                for (int step = 0; step < t.num_steps; ++step) {
                    const auto gradients = structure_gradients(st,input,g);
                    for (int k = 0; k < count; ++k) {
                        int x, y; component_axes(k,d,x,y);
                        for (std::size_t i = 0; i < g.size; ++i) component[i] = gradients[x][i]*gradients[y][i];
                        for (int axis = 0; axis < d; ++axis) {
                            if (integration[axis].radius == 0) continue;
                            correlate(component.data(),scratch.data(),g,axis,integration[axis],st.boundary);
                            component.swap(scratch);
                        }
                        std::copy(component.begin(),component.end(),tensor.begin()+k*g.size);
                    }
                    for (std::size_t i = 0; i < g.size; ++i)
                        diffusion_detail::diffusion_tensor(tensor.data(),g.size,i,d,T(t.lambda),T(t.exponent),T(t.alpha));
                    for (std::size_t i = 0; i < g.size; ++i)
                        diffusion_detail::flux(input.data(),tensor.data(),flux.data(),layout,i,d);
                    // Once flux is complete the update only reads input[i], so it is safe in place.
                    for (std::size_t i = 0; i < g.size; ++i)
                        input[i] = diffusion_detail::update(input.data(),flux.data(),layout,i,d,T(t.step_size));
                }
                a.channel(c.channel,input);
            };
            if (a.bytes == 4) run(float{}); else run(double{});
        });
    }
    restore_valid(s);
}
}
