#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double factor;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const double* field) const { return double(x)*factor; }
};
}
void apply_cuda(const BrightnessMultiply& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto& t, auto& v, const Grid& g, const ChannelPlan& c) {
        
        pointwise(v,g,c,Function{c.values[0]});
    });
}
} // namespace vcr::augment::cuda
