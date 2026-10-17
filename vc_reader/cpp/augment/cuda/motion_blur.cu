#include "../motion_blur.hpp"
#include "voxel_transform.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const MotionBlur& t,Sample& s,const Parameters& p,Session& session) {
    apply_voxel_channels(t,s,p,session);
}
} // namespace vcr::augment::cuda
