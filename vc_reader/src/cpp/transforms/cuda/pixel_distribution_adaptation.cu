#include "../cpu/pixel_distribution_adaptation.hpp"
#include "voxel_transform.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const PixelDistributionAdaptation& t,Sample& s,const Parameters& p,Session& session){
    PixelDistributionAdaptation settings;
    const PixelDistributionAdaptation* effective=&t;
    if(!t.reference_key.empty()){
        settings=t;settings.reference.wide=vd::reference_target(t.reference_key,s,p).bytes==8;effective=&settings;
    }
    with_voxel_channels(*effective,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        Buffer<vd::Stats> rs(1);
        if(t.reference_key.empty()){
            with_reference_channel(t.reference,c.channel,[&](auto ref){summarize_channel(ref,rs.ptr,t.mode==0);});
        }else{
            auto& ref=session.arrays.at(t.reference_key);auto channel=ref.shape[0]==1?0:c.channel;
            if(ref.bytes==4)summarize_channel(ref.channel<float>(channel),rs.ptr,t.mode==0);
            else summarize_channel(ref.channel<double>(channel),rs.ptr,t.mode==0);
        }
        launch_voxel_transform(PixelDistributionAdaptationKernel{c.values[0],t.mode},g,input,output,stats,error,nullptr,rs.ptr);
    });
}
}
