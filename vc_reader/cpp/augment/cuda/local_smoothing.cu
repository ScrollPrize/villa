#include "gaussian.cuh"
namespace vcr::augment::cuda {
template<class T> __device__ void blend(KernelIndex vcr_launch, T* data, const T* smoothed, LocalFieldView field, std::size_t n, double strength) {
    EACH_INDEX(n) { T w = T(field[i]) * T(strength); data[i] = data[i] * (T(1)-w) + smoothed[i]*w; }
}
struct blend_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { blend(index,args...); }
};
void apply_cuda(const LocalSmoothing& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto&, auto v, const Grid& g, const ChannelPlan& c) {
        using V = typename decltype(v)::value_type;
        Buffer<V> smooth(v.size); View<V> smoothed{smooth.ptr,v.size};
        auto field = field_buffer(c.field,c.device_field);
        blur_into(v,smoothed,g,std::vector<double>(g.shape.size(),c.values[1]),4,Boundary::Reflect,false);
        launch(blocks(g.size),256,0,vcr::device::stream(), blend_launch{}, v.ptr,smooth.ptr,LocalFieldView(c.field,g,field->ptr),g.size,c.values[0]);
        check(cudaGetLastError());
    });
}
} // namespace vcr::augment::cuda
