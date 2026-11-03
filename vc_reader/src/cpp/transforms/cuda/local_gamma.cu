#include "pointwise.cuh"
namespace vcr::transforms::cuda {
namespace {
struct Function {
    double gamma;
    template<class T> __device__ T operator()(T x, std::size_t i, Layout g, LocalFieldView field, const Statistics* stats) const { T low=T(stats->low), span=fmax(T(stats->high)-low,T(1e-8)), norm=(x-low)/span, w=T(field[i]); return (norm*(T(1)-w)+pow(norm,T(gamma))*w)*span+low; }
};
}
void apply_cuda(const LocalGamma& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& g, const ChannelPlan& c) {
        local_pointwise(v,g,c,Function{c.values[0]},true,false);
    });
}
} // namespace vcr::transforms::cuda
