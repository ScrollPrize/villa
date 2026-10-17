#include "noise.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const RicianNoise& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto&, auto& v, const Grid& g, const ChannelPlan& c) {
        using V = typename std::decay_t<decltype(v)>::value_type;
        Buffer<V> data(v);
        noise_kernel<<<blocks(g.size),256>>>(data.ptr,g.size,c.values[0],c.seed,true);
        check(cudaGetLastError()); data.download(v.data());
    });
}
} // namespace vcr::augment::cuda
