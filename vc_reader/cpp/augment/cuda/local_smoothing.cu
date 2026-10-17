#include "gaussian.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void blend(T* data, const T* smoothed, const double* field, std::size_t n, double strength) {
    EACH_INDEX(n) { T w = T(field[i] * strength); data[i] = data[i] * (T(1)-w) + smoothed[i]*w; }
}
void apply_cuda(const LocalSmoothing& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto&, auto& v, const Grid& g, const ChannelPlan& c) {
        using V = typename std::decay_t<decltype(v)>::value_type;
        Buffer<V> data(v), smooth(v); Buffer<double> field(c.field);
        blur(smooth,g,std::vector<double>(g.shape.size(),c.values[1]),4,false,false);
        blend<<<blocks(g.size),256>>>(data.ptr,smooth.ptr,field.ptr,g.size,c.values[0]);
        check(cudaGetLastError()); data.download(v.data());
    });
}

} // namespace vcr::augment::cuda
