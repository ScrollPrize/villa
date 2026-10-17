#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double gamma, low, span;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const double* field) const { double norm = (double(x)-low)/span; return (norm*(1-field[i])+power<T>(norm,gamma)*field[i])*span+low; }
};
}
void apply_cuda(const LocalGamma& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto& t, auto& v, const Grid& g, const ChannelPlan& c) {
        auto bounds = std::minmax_element(v.begin(),v.end());
        pointwise(v,g,c,Function{c.values[0],double(*bounds.first),std::max(double(*bounds.second)-double(*bounds.first),1e-8)});
    });
}
} // namespace vcr::augment::cuda
