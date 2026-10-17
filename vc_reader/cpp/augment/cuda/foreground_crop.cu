#include "../foreground_crop.hpp"
#include "geometry.cuh"
#include "foreground_selection.cuh"
namespace vcr::augment::cuda {
namespace vd=volume_detail;
Parameters sample_foreground(const ForegroundCrop& t,const Sample& s,Random& r,Session& session){
    require(!session.async,"ForegroundCrop requires sync=True for foreground selection");
    const auto& f=vd::mask_field(s,t.mask,t.channel);vd::Grid3 g(f.array.spatial());auto seed=r.next();Buffer<ForegroundCount> prefix(g.size),chosen(1);auto& mask=session.arrays.at(t.mask);
    mask.typed([&](auto v){foreground_prefix(v.ptr+t.channel*g.size,g.size,t.threshold,prefix);});choose_foreground_indices(prefix,seed,chosen,false);ForegroundCount index;chosen.download(&index);return vd::foreground_geometry(t,f.array.spatial(),index?std::size_t(index-1):g.size,r);
}
void apply_cuda(const ForegroundCrop&,Sample& s,const Parameters& p,Session& session){geometry_cuda(s,p,session);}
}
