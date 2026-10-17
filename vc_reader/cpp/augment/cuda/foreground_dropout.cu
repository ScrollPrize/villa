#include "../foreground_dropout.hpp"
#include "dropout_mask.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void foreground_holes(ForegroundDropoutKernel kernel,const T* mask,volume_detail::Grid3 grid,double* field){if(!threadIdx.x && !blockIdx.x)kernel(mask,grid,field);}
Parameters sample_dropout(const ForegroundDropout& t,const Sample& s,Random& r,bool record,Session& session){
    auto kernel=sample_holes(t,r);volume_detail::mask_field(s,t.mask,t.mask_channel);auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::Grid3 g(p.shape);auto field=std::make_shared<Buffer<double>>(g.size);
    session.arrays.at(t.mask).typed([&](auto v){foreground_holes<<<1,1,0,vcr::device::stream()>>>(kernel,v.ptr+t.mask_channel*g.size,g,field->ptr);});check(cudaGetLastError());record_dropout(p,field,record);return p;
}
void apply_cuda(const ForegroundDropout& t,Sample& s,const Parameters& p,Session& session){erase_mask(s,p,session,t.fill);}
}
