#include "../grid_dropout.hpp"
#include "dropout_mask.cuh"
namespace vcr::augment::cuda {
Parameters sample_dropout(const GridDropout& t,const Sample& s,Random& r,bool record,Session&) {
    auto kernel=sample_holes(t,r);auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::Grid3 g(p.shape);auto field=std::make_shared<Buffer<float>>(g.size);
    periodic_dropout<<<blocks(g.size),256,0,vcr::device::stream()>>>(kernel,g,field->ptr);check(cudaGetLastError());record_dropout(p,field,record);return p;
}
void apply_cuda(const GridDropout& t,Sample& s,const Parameters& p,Session& session){erase_mask(s,p,session,t.fill);}
}
