#include "geometry.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const Flip&, Sample& s, const Parameters& p, Session& session) {
    geometry_cuda(s,p,session);
}
} // namespace vcr::transforms::cuda
