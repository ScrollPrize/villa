#include "geometry.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const LowResolution& t, Sample& s, const Parameters& p) {
    apply_resolution(t,s,p,resize_cuda);
}
} // namespace vcr::augment::cuda
