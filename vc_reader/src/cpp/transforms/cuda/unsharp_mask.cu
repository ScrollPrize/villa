#include "../cpu/unsharp_mask.hpp"
#include "voxel_transform.cuh"
#include "gaussian.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const UnsharpMask& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        using T=typename decltype(input)::value_type;
        Buffer<T> blurred(input.size);blur_into(input,View<T>{blurred.ptr,blurred.size},g,std::vector<double>(g.shape.size(),c.values[1]),4,Boundary::Nearest,false);
        launch_voxel_transform(UnsharpMaskKernel{c.values[0]},g,input,output,stats,error,nullptr,nullptr,blurred.ptr);
    });
}
}
