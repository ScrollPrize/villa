#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double gamma;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const float* field, const Statistics* stats) const { double span = fmax(stats->high-stats->low,1e-8), norm = (double(x)-stats->low)/span; return (norm*(1-field[i])+power<T>(norm,gamma)*field[i])*span+stats->low; }
};
}
void apply_cuda(const LocalGamma& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        pointwise(v,g,c,Function{c.values[0]},true,false);
    });
}
} // namespace vcr::augment::cuda
