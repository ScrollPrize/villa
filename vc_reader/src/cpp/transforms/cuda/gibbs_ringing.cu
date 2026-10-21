#include "../cpu/gibbs_ringing.hpp"
#include "voxel_transform.cuh"
#include "fourier.cuh"
namespace vcr::transforms::cuda {
template<class T,int D> __device__ void truncate_frequencies(KernelIndex vcr_launch, vd::ComplexT<T>* spectrum,vd::GridND<D> g,double cutoff){EACH_INDEX(g.size)if(!vd::low_frequency(g,i,cutoff))spectrum[i]={};}
struct truncate_frequencies_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { truncate_frequencies(index,args...); }
};
void apply_cuda(const GibbsRinging& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& grid,const ChannelPlan& c,auto input,auto output,const auto*,int*){
        if(c.values[0]==1){copy_device(output,input);return;}using T=typename decltype(input)::value_type;vd::with_grid(grid.shape,[&](const auto& g){Buffer<vd::ComplexT<T>> spectrum(g.size);to_complex(input,spectrum);fft_volume(spectrum,g,false);
        launch(blocks(g.size),256,0,vcr::device::stream(), truncate_frequencies_launch{}, spectrum.ptr,g,c.values[0]);check(cudaGetLastError());fft_volume(spectrum,g,true);from_complex(spectrum,output);
        });
    });
}
}
