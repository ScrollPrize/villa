#include "../histogram_matching.hpp"
#include "histogram.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const HistogramMatching& t,Sample& s,const Parameters& p,Session& session){
    with_double_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        auto reference=vd::reference_channel(t.reference,c.channel);Buffer<double> ref(reference),table(2*t.bins);Buffer<vd::Stats> rs(1);
        vd::HistogramGrid layout(vd::Grid3(g.shape),g.shape,t.bins,0);build_histograms(input,g,stats,layout,table.ptr);
        reference_histogram<<<1,1,0,vcr::device::stream()>>>(ref.ptr,ref.size,rs.ptr,table.ptr+t.bins,t.bins);check(cudaGetLastError());
        launch_voxel_transform(HistogramMatchingKernel{c.values[0],t.bins},g,input,output,stats,error,table.ptr,rs.ptr);
    });
}
}
