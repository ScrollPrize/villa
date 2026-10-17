#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double factor;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const double* field, const Statistics* stats) const { return double(x)*(1-field[i])+((double(x)-stats->mean)*factor+stats->mean)*field[i]; }
};
}
void apply_cuda(const LocalContrast& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        pointwise(v,g,c,Function{c.values[0]},true,true);
    });
}
} // namespace vcr::augment::cuda
