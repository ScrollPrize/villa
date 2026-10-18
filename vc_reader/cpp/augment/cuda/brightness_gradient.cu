#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double strength; bool clip;
    template<class T> __device__ T operator()(T x, std::size_t i, Layout g, LocalFieldView field, const Statistics* stats) const { T y=x+T(field[i])*T(strength); return clip ? fmin(fmax(y,T(stats->low)),T(stats->high)) : y; }
};
}
void apply_cuda(const BrightnessGradient& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        local_pointwise(v,g,c,Function{c.values[0],t.clip},t.clip,false);
    });
}
} // namespace vcr::augment::cuda
