#include "../ringing_overshoot.hpp"
#include "voxel_transform.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const RingingOvershoot& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        for(int a=0;a<3;++a){launch_voxel_transform(RingingOvershootKernel{c.values[0],int(c.values[1]),a},g,input,output,stats,error);if(a<2)std::swap(input,output);}
    });
}
}
