#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double factor, m;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const double* field) const { return double(x)*(1-field[i])+((double(x)-m)*factor+m)*field[i]; }
};
}
void apply_cuda(const LocalContrast& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto& t, auto& v, const Grid& g, const ChannelPlan& c) {
        
        pointwise(v,g,c,Function{c.values[0],local_mean(v,c.field)});
    });
}
} // namespace vcr::augment::cuda
