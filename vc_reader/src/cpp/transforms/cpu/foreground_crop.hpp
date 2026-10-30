#pragma once
#include "../ops/foreground_crop.hpp"
#include "../shared/foreground_selection.hpp"
namespace vcr::transforms {
namespace volume_detail {
inline Parameters foreground_geometry(const ForegroundCrop& t,const Shape& input,std::size_t index,Random& r) {
    require(t.shape.size()==input.size() && product(t.shape)>0,"crop shape must match spatial dimensions");Grid g(input);auto p=geometry(t.shape);p.exact=true;
    for(std::size_t a=0;a<input.size();++a){auto delta=(long long)input[a]-(long long)t.shape[a];
        if(delta<0)p.offset[a]=floor(double(delta)/2);
        else if(index==g.size)p.offset[a]=double(r.index(std::size_t(delta)+1));
        else {auto x=(long long)g.coord(index,a);auto lo=std::max(0ll,x-(long long)t.shape[a]+1),hi=std::min(x,delta);p.offset[a]=double(lo+(long long)r.index(std::size_t(hi-lo+1)));}}
    return p;
}
}
inline Parameters sample(const ForegroundCrop& t,const Sample& s,Random& r) {
    const auto& f=volume_detail::mask_field(s,t.mask,t.channel);auto g=Grid(f.array.spatial());auto seed=r.next();
    auto index=f.array.typed([&](const auto* v){return volume_detail::foreground_index(v+t.channel*g.size,g.size,t.threshold,seed);});
    return volume_detail::foreground_geometry(t,f.array.spatial(),index,r);
}
inline void apply(const ForegroundCrop&,Sample& s,const Parameters& p){apply_geometry(s,p);}

} // namespace vcr::transforms
