#pragma once
#include "foreground_selection.hpp"
namespace vcr::augment::volume_detail {
struct EraseValues {double image=0,label=0;bool labels=false;};
inline Parameters select_dropout_channels(const ChannelOptions& options,const Sample& s,Random& r) {
    auto p=sample_channels(s,options,r,[&](const Array&,Random&){return ChannelPlan{};});p.shape=sample_shape(s);Grid3 g(p.shape);return p;
}
inline void erase_mask(Sample& s,const Parameters& p,const EraseValues& fill) {
    for(const auto& c:p.channels){auto& a=s.at(c.key).array;auto n=product(a.spatial());for(std::size_t i=0;i<n;++i)if(p.field[i])a.set(c.channel*n+i,fill.image);}
    if(fill.labels)for(auto& [name,f]:s)if(f.target.role==Role::Label){auto n=product(f.array.spatial());for(std::size_t i=0;i<f.array.size();++i)if(p.field[i%n])f.array.set(i,fill.label);}
    restore_valid(s);
}
struct PeriodicHoles {
    std::size_t size[3],offset[3];bool enabled[3];double ratio;
    PeriodicHoles(const Shape& unit_size,const std::string& axes,double ratio,std::uint64_t seed):ratio(ratio) {
        require(unit_size.size()==3 && !axes.empty() && axes.find_first_not_of("zyx")==std::string::npos,"invalid unit_size or axes");probability(ratio);
        for(int a=0;a<3;++a){require(unit_size[a]>0 && unit_size[a]<=65536,"unit sizes must be [1,65536]");size[a]=unit_size[a];offset[a]=std::size_t(uniform(seed,a)*size[a]);enabled[a]=axes.find("zyx"[a])!=std::string::npos;}
    }
    VCR_HD bool hole(const Grid3& g,std::size_t i,int a) const {return (g.coord(i,a)+offset[a])%size[a]<std::size_t(floor(ratio*size[a]));}
};
} // namespace vcr::augment::volume_detail
