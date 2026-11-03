#include "../cpu/ringing_overshoot.hpp"
#include "voxel_transform.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const RingingOvershoot& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        const auto result=output;
        for(std::size_t a=0;a<g.shape.size();++a){launch_voxel_transform(RingingOvershootKernel{c.values[0],int(c.values[1]),int(a)},g,input,output,stats,error);if(a+1<g.shape.size())std::swap(input,output);}
        if(output.ptr!=result.ptr)copy_device(result,output);
    });
}
}
