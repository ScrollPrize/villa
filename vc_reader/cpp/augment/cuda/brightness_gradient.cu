#include "pointwise.cuh"
namespace vcr::augment::cuda {
namespace {
struct Function {
    double strength, low, high; bool clip;
    template<class T> __device__ double operator()(T x, std::size_t i, Layout g, const double* field) const { double y = double(x)+field[i]*strength; return clip ? clamp_value(y,low,high) : y; }
};
}
void apply_cuda(const BrightnessGradient& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto& t, auto& v, const Grid& g, const ChannelPlan& c) {
        auto bounds = std::minmax_element(v.begin(),v.end());
        pointwise(v,g,c,Function{c.values[0],double(*bounds.first),double(*bounds.second),t.clip});
    });
}
} // namespace vcr::augment::cuda
