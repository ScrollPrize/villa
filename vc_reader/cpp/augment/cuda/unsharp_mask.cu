#include "../unsharp_mask.hpp"
#include "voxel_transform.cuh"
#include "gaussian.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const UnsharpMask& t,Sample& s,const Parameters& p,Session& session){
    with_double_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        Buffer<double> blurred(input.size);copy_device(View<double>{blurred.ptr,blurred.size},input);blur(View<double>{blurred.ptr,blurred.size},g,std::vector<double>(3,c.values[1]),4,Boundary::Nearest,false);
        launch_voxel_transform(UnsharpMaskKernel{c.values[0]},g,input,output,stats,error,blurred.ptr);
    });
}
}
