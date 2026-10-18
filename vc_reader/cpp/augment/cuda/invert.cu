#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {

    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const float* field, const Statistics* stats) const { return 2*stats->mean-double(x); }
};
}
void apply_cuda(const Invert& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        pointwise(v,g,c,Function{},true,false);
    });
}
} // namespace vcr::augment::cuda
