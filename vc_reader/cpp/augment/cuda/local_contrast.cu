#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double factor;
    template<class T> __device__ T operator()(T x, std::size_t i, Layout g, const float* field, const Statistics* stats) const { T w=T(field[i]), m=T(stats->mean); return x*(T(1)-w)+((x-m)*T(factor)+m)*w; }
};
}
void apply_cuda(const LocalContrast& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        pointwise(v,g,c,Function{c.values[0]},true,true);
    });
}
} // namespace vcr::augment::cuda
