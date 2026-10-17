#include "noise.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const GaussianNoise& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto&, auto v, const Grid& g, const ChannelPlan& c) {
        noise_kernel<<<blocks(g.size),256>>>(v.ptr,g.size,c.values[0],c.seed,false);
        check(cudaGetLastError());
    });
}
} // namespace vcr::augment::cuda
