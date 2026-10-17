#include "../object_dropout.hpp"
#include "dropout_mask.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void object_holes(ObjectDropoutKernel kernel,const T* mask,volume_detail::Grid3 grid,double* field,std::size_t* parents){if(!threadIdx.x && !blockIdx.x)kernel(mask,grid,field,parents);}
Parameters sample_dropout(const ObjectDropout& t,const Sample& s,Random& r,bool record,Session& session){
    auto kernel=sample_objects(t,r);volume_detail::mask_field(s,t.mask,t.mask_channel);auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::Grid3 g(p.shape);auto field=std::make_shared<Buffer<double>>(g.size);Buffer<std::size_t> parents(g.size);
    session.arrays.at(t.mask).typed([&](auto v){object_holes<<<1,1,0,vcr::device::stream()>>>(kernel,v.ptr+t.mask_channel*g.size,g,field->ptr,parents.ptr);});check(cudaGetLastError());record_dropout(p,field,record);return p;
}
void apply_cuda(const ObjectDropout& t,Sample& s,const Parameters& p,Session& session){erase_mask(s,p,session,t.fill);}
}
