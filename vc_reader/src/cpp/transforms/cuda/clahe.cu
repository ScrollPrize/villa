#include "../cpu/clahe.hpp"
#include "histogram.cuh"
namespace vcr::transforms::cuda {
void apply_cuda(const CLAHE& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& g,const ChannelPlan& c,auto input,auto output,const auto* stats,int* error){
        auto layout=vd::with_grid(g.shape,[&](const auto& grid){return vd::HistogramGrid(grid,t.tile_shape,t.bins,c.values[0]);});Buffer<double> table(layout.count*t.bins);build_histograms(input,g,stats,layout,table.ptr);
        launch_voxel_transform(CLAHEKernel{layout},g,input,output,stats,error,table.ptr);
    });
}
}
