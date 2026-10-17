#pragma once
#include "resident.cuh"
#include "../dropout_mask.hpp"
namespace vcr::augment::cuda {
void erase_mask(Sample&,const Parameters&,Session&,const volume_detail::EraseValues&);
inline void record_dropout(Parameters& p,const std::shared_ptr<Buffer<double>>& field,bool record){
    p.device_field=field;if(record){p.field.resize(field->size);field->download(p.field.data());}
}
template<class Kernel> __global__ void periodic_dropout(Kernel kernel,volume_detail::Grid3 grid,double* field){EACH_INDEX(grid.size)field[i]=kernel(grid,i);}
}
