#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double strength; bool clip;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const float* field, const Statistics* stats) const { double y = double(x)+field[i]*strength; return clip ? clamp_value(y,stats->low,stats->high) : y; }
};
}
void apply_cuda(const BrightnessGradient& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        pointwise(v,g,c,Function{c.values[0],t.clip},t.clip,false);
    });
}
} // namespace vcr::augment::cuda
