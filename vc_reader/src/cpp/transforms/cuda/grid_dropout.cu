#include "../cpu/grid_dropout.hpp"
#include "dropout_mask.cuh"
namespace vcr::transforms::cuda {
Parameters sample_dropout(const GridDropout& t,const Sample& s,Random& r,bool record,Session&) {
    auto kernel=sample_holes(t,r,sample_shape(s).size());auto p=volume_detail::select_dropout_channels(t.options,s,r);auto field=std::make_shared<Buffer<float>>(product(p.shape));
    volume_detail::with_grid(p.shape,[&](const auto& g){
    launch(blocks(g.size),256,0,vcr::device::stream(), periodic_dropout_launch{}, kernel,g,field->ptr);check(cudaGetLastError());});record_dropout(p,field,record);return p;
}
void apply_cuda(const GridDropout& t,Sample& s,const Parameters& p,Session& session){erase_mask(s,p,session,t.fill);}
}
