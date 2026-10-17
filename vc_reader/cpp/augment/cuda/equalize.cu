#include "../equalize.hpp"
#include "histogram.cuh"
namespace vcr::augment::cuda {
void apply_cuda(const Equalize& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan&,auto input,auto output,const auto* stats,int* error){
        vd::HistogramGrid layout(vd::Grid3(g.shape),g.shape,t.bins,0);Buffer<double> table(t.bins);build_histograms(input,g,stats,layout,table.ptr);
        launch_voxel_transform(EqualizeKernel{t.bins},g,input,output,stats,error,table.ptr);
    });
}
}
