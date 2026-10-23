#include "dropout_mask.cuh"
namespace vcr::transforms::cuda {
template<class T> __device__ void erase_field(KernelIndex vcr_launch, T* data,std::size_t n,std::size_t spatial,const float* field,T fill){EACH_INDEX(n)if(field[i%spatial])data[i]=fill;}
struct erase_field_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { erase_field(index,args...); }
};
void erase_mask(Sample& s,const Parameters& p,Session& session,const volume_detail::EraseValues& fill){
    auto field=field_buffer(p.field,p.device_field);
    for(const auto& c:p.channels){auto& a=session.arrays.at(c.key);auto n=product(a.spatial());a.typed([&](auto v){using T=typename decltype(v)::value_type;launch(blocks(n),256,0,vcr::device::stream(), erase_field_launch{}, v.ptr+c.channel*n,n,n,field->ptr,Array::checked_cast<T>(fill.image));});check(cudaGetLastError());}
    if(fill.labels)for(const auto& [key,f]:s)if(f.target.role==Role::Label){auto& a=session.arrays.at(key);auto n=product(a.spatial());a.typed([&](auto v){using T=typename decltype(v)::value_type;launch(blocks(v.size),256,0,vcr::device::stream(), erase_field_launch{}, v.ptr,v.size,n,field->ptr,Array::checked_cast<T>(fill.label));});check(cudaGetLastError());}
    session.restore_valid(s);
}
}
