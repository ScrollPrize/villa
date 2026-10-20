#pragma once
#include "../ops/options.hpp"
#include "foreground_selection.hpp"
namespace vcr::transforms::volume_detail {
inline Parameters select_dropout_channels(const ChannelOptions& options,const Sample& s,Random& r) {
    auto p=sample_channels(s,options,r,[&](const Array&,Random&){return ChannelPlan{};});p.shape=sample_shape(s);require(p.shape.size()==2 || p.shape.size()==3,"dropout requires 2D or 3D images");return p;
}
inline void erase_mask(Sample& s,const Parameters& p,const EraseValues& fill) {
    for(const auto& c:p.channels){auto& a=s.at(c.key).array;auto n=product(a.spatial());for(std::size_t i=0;i<n;++i)if(p.field[i])a.set(c.channel*n+i,fill.image);}
    if(fill.labels)for(auto& [name,f]:s)if(f.target.role==Role::Label){auto n=product(f.array.spatial());for(std::size_t i=0;i<f.array.size();++i)if(p.field[i%n])f.array.set(i,fill.label);}
    restore_valid(s);
}
struct PeriodicHoles {
    std::size_t size[3],offset[3];bool enabled[3];double ratio;
    PeriodicHoles(const Shape& unit_size,const std::string& axes,double ratio,std::uint64_t seed,std::size_t dim):ratio(ratio) {
        const auto sizes=spatial_sizes(unit_size,dim,8);const auto names=vcr::transforms::axes(dim);
        const auto selected=axes.empty()?names:axes;
        require(selected.find_first_not_of(names)==std::string::npos,"axis is not present in the sample");probability(ratio);
        for(std::size_t a=0;a<dim;++a){require(sizes[a]>0 && sizes[a]<=65536,"unit sizes must be [1,65536]");size[a]=sizes[a];offset[a]=std::size_t(uniform(seed,a)*size[a]);enabled[a]=selected.find(names[a])!=std::string::npos;}
    }
    template<int D> VCR_HD bool hole(const GridND<D>& g,std::size_t i,int a) const {return (g.coord(i,a)+offset[a])%size[a]<std::size_t(floor(ratio*size[a]));}
};
} // namespace vcr::transforms::volume_detail
