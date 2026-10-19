#include "../cpu/salt_and_pepper.hpp"
#include "voxel_transform.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const SaltAndPepper& t,Sample& s,const Parameters& p,Session& session) {
    apply_voxel_channels(t,s,p,session);
}
} // namespace vcr::transforms::cuda
