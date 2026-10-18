#include "../object_dropout.hpp"
#include "dropout_mask.cuh"
#include "components.cuh"
namespace vcr::augment::cuda {
__global__ void object_holes(ObjectDropoutKernel kernel,const ComponentIndex* parents,std::size_t n,float* field){
    EACH_INDEX(n)field[i]=parents[i]!=n && volume_detail::uniform(kernel.seed,parents[i])<kernel.probability;
}
Parameters sample_dropout(const ObjectDropout& t,const Sample& s,Random& r,bool record,Session& session){
    auto kernel=sample_objects(t,r);volume_detail::mask_field(s,t.mask,t.mask_channel);auto p=volume_detail::select_dropout_channels(t.options,s,r);Grid g(p.shape);auto field=std::make_shared<Buffer<float>>(g.size);Buffer<ComponentIndex> parents(g.size);
    session.arrays.at(t.mask).typed([&](auto v){label_components<true>(v.ptr+t.mask_channel*g.size,g,kernel.threshold,false,parents);});
    object_holes<<<blocks(g.size),256,0,vcr::device::stream()>>>(kernel,parents.ptr,g.size,field->ptr);check(cudaGetLastError());record_dropout(p,field,record);return p;
}
void apply_cuda(const ObjectDropout& t,Sample& s,const Parameters& p,Session& session){erase_mask(s,p,session,t.fill);}
}
