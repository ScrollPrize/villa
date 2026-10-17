#include "../multiplicative_noise.hpp"
#include "voxel_transform.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const MultiplicativeNoise& t,Sample& s,const Parameters& p,Session& session) {
    apply_voxel_channels(t,s,p,session);
}
} // namespace vcr::augment::cuda
