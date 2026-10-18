#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double factor; bool clip;
    template<class T> __device__ T operator()(T x, std::size_t i, Layout g, const float* field, const Statistics* stats) const { T m=T(stats->mean), y=(x-m)*T(factor)+m; return clip ? fmin(fmax(y,T(stats->low)),T(stats->high)) : y; }
};
}
void apply_cuda(const Contrast& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        pointwise(v,g,c,Function{c.values[0],t.preserve_range},true,false);
    });
}
} // namespace vcr::augment::cuda
