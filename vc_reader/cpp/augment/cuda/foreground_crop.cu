#include "../foreground_crop.hpp"
#include "geometry.cuh"
namespace vcr::augment::cuda {
namespace vd=volume_detail;
template<class T> __global__ void choose_foreground(const T* mask,std::size_t n,double threshold,std::uint64_t seed,std::size_t* chosen){if(!threadIdx.x && !blockIdx.x)*chosen=vd::foreground_index(mask,n,threshold,seed);}
Parameters sample_foreground(const ForegroundCrop& t,const Sample& s,Random& r,Session& session){
    require(!session.async,"ForegroundCrop requires sync=True for foreground selection");
    const auto& f=vd::mask_field(s,t.mask,t.channel);vd::Grid3 g(f.array.spatial());auto seed=r.next();Buffer<std::size_t> chosen(1);auto& mask=session.arrays.at(t.mask);
    mask.typed([&](auto v){choose_foreground<<<1,1,0,vcr::device::stream()>>>(v.ptr+t.channel*g.size,g.size,t.threshold,seed,chosen.ptr);});check(cudaGetLastError());std::size_t index;chosen.download(&index);return vd::foreground_geometry(t,f.array.spatial(),index,r);
}
void apply_cuda(const ForegroundCrop&,Sample& s,const Parameters& p,Session& session){geometry_cuda(s,p,session);}
}
