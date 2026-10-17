#include "geometry.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const Transpose&, Sample& s, const Parameters& p) {
    apply_geometry_with(s,p,resample_cuda);
}
} // namespace vcr::augment::cuda
