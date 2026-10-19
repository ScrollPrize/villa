#include "noise.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const GaussianNoise& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto&, auto v, const Grid& g, const ChannelPlan& c) {
        launch(blocks(g.size),256,0,vcr::device::stream(), noise_kernel_launch{}, v.ptr,g.size,c.values[0],c.seed,false);
        check(cudaGetLastError());
    });
}
} // namespace vcr::augment::cuda
