#include "../cpu/vignetting.hpp"
#include "voxel_transform.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const Vignetting& t,Sample& s,const Parameters& p,Session& session) {
    apply_voxel_channels(t,s,p,session);
}
} // namespace vcr::transforms::cuda
