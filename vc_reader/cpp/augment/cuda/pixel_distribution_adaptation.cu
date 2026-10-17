#include "../pixel_distribution_adaptation.hpp"
#include "voxel_transform.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const PixelDistributionAdaptation& t,Sample& s,const Parameters& p,Session& session){
    with_double_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        auto reference=vd::reference_channel(t.reference,c.channel);Buffer<double> ref(reference);Buffer<vd::Stats> rs(1);
        summarize_channel<<<1,1,0,vcr::device::stream()>>>(ref.ptr,ref.size,rs.ptr);check(cudaGetLastError());
        launch_voxel_transform(PixelDistributionAdaptationKernel{c.values[0],t.mode},g,input,output,stats,error,nullptr,rs.ptr);
    });
}
}
