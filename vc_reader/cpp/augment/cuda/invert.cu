#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double m;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const double* field) const { return 2*m-double(x); }
};
}
void apply_cuda(const Invert& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto& t, auto& v, const Grid& g, const ChannelPlan& c) {
        
        pointwise(v,g,c,Function{mean(v)});
    });
}
} // namespace vcr::augment::cuda
