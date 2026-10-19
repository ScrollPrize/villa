#include "../cpu/pixel_distribution_adaptation.hpp"
#include "voxel_transform.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const PixelDistributionAdaptation& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        Buffer<vd::Stats> rs(1);
        with_reference_channel(t.reference,c.channel,[&](auto ref){summarize_channel(ref,rs.ptr,t.mode==0);});
        launch_voxel_transform(PixelDistributionAdaptationKernel{c.values[0],t.mode},g,input,output,stats,error,nullptr,rs.ptr);
    });
}
}
