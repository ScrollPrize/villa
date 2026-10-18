#include "../histogram_matching.hpp"
#include "histogram.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const HistogramMatching& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        Buffer<double> table(2*t.bins);Buffer<vd::Stats> rs(1);
        auto layout=vd::with_grid(g.shape,[&](const auto& grid){return vd::HistogramGrid(grid,g.shape,t.bins,0);});build_histograms(input,g,stats,layout,table.ptr);
        with_reference_channel(t.reference,c.channel,[&](auto ref){
            summarize_channel(ref,rs.ptr,false);
            build_global_histogram(ref,rs.ptr,t.bins,table.ptr+t.bins);
        });
        launch_voxel_transform(HistogramMatchingKernel{c.values[0],t.bins},g,input,output,stats,error,table.ptr,rs.ptr);
    });
}
}
