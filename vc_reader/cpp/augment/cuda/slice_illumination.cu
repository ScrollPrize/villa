#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    int axis;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const double* field) const { return double(x)*field[g.coord(i,axis)]; }
};
}
void apply_cuda(const SliceIllumination& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto& t, auto& v, const Grid& g, const ChannelPlan& c) {
        
        pointwise(v,g,c,Function{int(c.values[0])});
    });
}
} // namespace vcr::augment::cuda
