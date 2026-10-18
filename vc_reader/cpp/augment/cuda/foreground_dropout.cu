#include "../foreground_dropout.hpp"
#include "dropout_mask.cuh"
#include "foreground_selection.cuh"
namespace vcr::augment::cuda {
template<class T,int D> __global__ void foreground_holes(ForegroundDropoutKernel kernel,const T* mask,volume_detail::GridND<D> grid,const ForegroundCount* centers,float* field) {
    EACH_INDEX(grid.size) {
        bool erase=false;
        if(double(mask[i])>kernel.threshold)for(std::size_t h=0;h<kernel.count && !erase;++h) {
            if(!centers[h])continue;const auto center=std::size_t(centers[h]-1);bool inside=true;
            for(int a=0;a<D;++a){const auto x=grid.coord(center,a),at=grid.coord(i,a),lo=x>kernel.size[a]/2?x-kernel.size[a]/2:0;inside &= at>=lo && at-lo<kernel.size[a];}
            erase=inside;
        }
        field[i]=erase;
    }
}
Parameters sample_dropout(const ForegroundDropout& t,const Sample& s,Random& r,bool record,Session& session){
    auto kernel=sample_holes(t,r,sample_shape(s).size());volume_detail::mask_field(s,t.mask,t.mask_channel);auto p=volume_detail::select_dropout_channels(t.options,s,r);auto field=std::make_shared<Buffer<float>>(product(p.shape));
    volume_detail::with_grid(p.shape,[&](const auto& g){
    Buffer<ForegroundCount> prefix(g.size),centers(kernel.count);
    session.arrays.at(t.mask).typed([&](auto v){
        foreground_prefix(v.ptr+t.mask_channel*g.size,g.size,t.threshold,prefix);choose_foreground_indices(prefix,kernel.seed,centers,true);
        foreground_holes<<<blocks(g.size),256,0,vcr::device::stream()>>>(kernel,v.ptr+t.mask_channel*g.size,g,centers.ptr,field->ptr);
    });check(cudaGetLastError());});record_dropout(p,field,record);return p;
}
void apply_cuda(const ForegroundDropout& t,Sample& s,const Parameters& p,Session& session){erase_mask(s,p,session,t.fill);}
}
