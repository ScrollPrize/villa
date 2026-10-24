#pragma once
#include "../ops/object_dropout.hpp"
#include "../shared/dropout_mask.hpp"
namespace vcr::transforms {
// Class-preserving, face-connected objects (4 neighbors in 2D, 6 in 3D). Unlike binary connected_components,
// adjacent different label IDs remain distinct; integer comparisons are exact.
struct ObjectDropoutKernel {
    double probability,threshold;std::uint64_t seed;
    VCR_HD static std::size_t root(std::size_t* parents,std::size_t i){while(parents[i]!=i){parents[i]=parents[parents[i]];i=parents[i];}return i;}
    template<class T,int D> VCR_HD void operator()(const T* mask,const volume_detail::GridND<D>& g,float* field,std::size_t* parents)const{
        for(std::size_t i=0;i<g.size;++i)parents[i]=i;
        for(std::size_t i=0;i<g.size;++i)if(double(mask[i])>threshold)for(int a=0;a<D;++a)if(g.coord(i,a)>0){auto j=i-g.stride[a];if(mask[j]==mask[i]){auto x=root(parents,i),y=root(parents,j);if(x!=y)parents[x>y?x:y]=x<y?x:y;}}
        for(std::size_t i=0;i<g.size;++i)field[i]=double(mask[i])>threshold && volume_detail::uniform(seed,root(parents,i))<probability;
    }
};
inline ObjectDropoutKernel sample_objects(const ObjectDropout& t,Random& r){double p=draw(t.drop_probability,r);probability(p);return {p,t.threshold,r.next()};}
inline Parameters sample(const ObjectDropout& t,const Sample& s,Random& r){
    auto kernel=sample_objects(t,r);const auto& f=volume_detail::mask_field(s,t.mask,t.mask_channel);auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::with_grid(p.shape,[&](const auto& g){p.field.resize(g.size);std::vector<std::size_t> parents(g.size);
    f.array.typed([&](const auto* v){kernel(v+t.mask_channel*g.size,g,p.field.data(),parents.data());});});return p;
}
inline void apply(const ObjectDropout& t,Sample& s,const Parameters& p){volume_detail::erase_mask(s,p,t.fill);}
}
