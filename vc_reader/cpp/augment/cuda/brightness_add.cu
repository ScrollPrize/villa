#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double shift;
    template<class T> __device__ T operator()(T x, std::size_t i, Layout g, const float* field, const Statistics* stats) const { return x+T(shift); }
};
}
void apply_cuda(const BrightnessAdd& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        pointwise(v,g,c,Function{c.values[0]},false,false);
    });
}
} // namespace vcr::augment::cuda
