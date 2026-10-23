#pragma once
#include "resident.cuh"
#include "../shared/dropout_mask.hpp"
namespace vcr::transforms::cuda {
void erase_mask(Sample&,const Parameters&,Session&,const volume_detail::EraseValues&);
inline void record_dropout(Parameters& p,const std::shared_ptr<Buffer<float>>& field,bool record){
    p.device_field=field;if(record){p.field.resize(field->size);field->download(p.field.data());}
}
template<class Kernel,int D> __device__ void periodic_dropout(KernelIndex vcr_launch, Kernel kernel,volume_detail::GridND<D> grid,float* field){EACH_INDEX(grid.size)field[i]=kernel(grid,i);}
struct periodic_dropout_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { periodic_dropout(index,args...); }
};
}
