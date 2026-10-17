#include "gaussian.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void blend(T* data, const T* smoothed, const double* field, std::size_t n, double strength) {
    EACH_INDEX(n) { T w = T(field[i] * strength); data[i] = data[i] * (T(1)-w) + smoothed[i]*w; }
}
void apply_cuda(const LocalSmoothing& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto&, auto v, const Grid& g, const ChannelPlan& c) {
        using V = typename decltype(v)::value_type;
        Buffer<V> smooth(v.size); View<V> smoothed{smooth.ptr,v.size}; copy_device(smoothed,v);
        auto field = field_buffer(c.field,c.device_field);
        blur(smoothed,g,std::vector<double>(g.shape.size(),c.values[1]),4,false,false);
        blend<<<blocks(g.size),256>>>(v.ptr,smooth.ptr,field->ptr,g.size,c.values[0]);
        check(cudaGetLastError());
    });
}
} // namespace vcr::augment::cuda
