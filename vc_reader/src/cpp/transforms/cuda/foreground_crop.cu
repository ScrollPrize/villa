#include "../cpu/foreground_crop.hpp"
#include "geometry.cuh"
#include "foreground_selection.cuh"
#include "random.cuh"
namespace vcr::transforms::cuda {
namespace vd=volume_detail;
struct ForegroundOffset {
    __device__ void operator()(KernelIndex,const ForegroundCount* chosen,Layout input,Layout output,std::uint64_t seed,long long* offset) const {
        Random random{seed}; const auto index=*chosen?std::size_t(*chosen-1):input.size;
        for (int a=0;a<input.dim;++a) {
            const auto delta=static_cast<long long>(input.shape[a])-static_cast<long long>(output.shape[a]);
            if (delta<0) offset[a]=delta/2-(delta%2!=0);
            else if(index==input.size) offset[a]=static_cast<long long>(random_index(random,std::size_t(delta)+1));
            else {
                const auto x=static_cast<long long>(input.coord(index,a));
                const auto lo=max(0ll,x-static_cast<long long>(output.shape[a])+1),hi=min(x,delta);
                offset[a]=lo+static_cast<long long>(random_index(random,std::size_t(hi-lo+1)));
            }
        }
    }
};
Parameters sample_foreground(const ForegroundCrop& t,const Sample& s,Random& r,bool record,Session& session){
    const auto& f=vd::mask_field(s,t.mask,t.channel);Grid g(f.array.spatial());
    require(t.shape.size()==g.shape.size() && product(t.shape)>0,"crop shape must match spatial dimensions");
    auto seed=r.next();Buffer<ForegroundCount> prefix(g.size),chosen(1);auto& mask=session.arrays.at(t.mask);
    mask.typed([&](auto v){foreground_prefix(v.ptr+t.channel*g.size,g.size,t.threshold,prefix);});
    choose_foreground_indices(prefix,seed,chosen,false);
    auto p=geometry(t.shape);p.exact=true;
    auto offset=std::make_shared<Buffer<long long>>(g.shape.size());
    launch(1,1,0,vcr::device::stream(),ForegroundOffset{},chosen.ptr,Layout(g),Layout(Grid(t.shape)),r.state,offset->ptr);
    if(record) {
        std::vector<long long> host(g.shape.size());offset->download(host.data());
        std::copy(host.begin(),host.end(),p.offset.begin());
    } else p.device_offset=offset;
    return p;
}
void apply_cuda(const ForegroundCrop&,Sample& s,const Parameters& p,Session& session){geometry_cuda(s,p,session);}
}
