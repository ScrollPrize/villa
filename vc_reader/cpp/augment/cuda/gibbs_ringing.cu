#include "../gibbs_ringing.hpp"
#include "voxel_transform.cuh"
#include "fourier.cuh"
namespace vcr::augment::cuda {
__global__ void truncate_frequencies(vd::Complex* spectrum,vd::Grid3 g,double cutoff){EACH_INDEX(g.size)if(!vd::low_frequency(g,i,cutoff))spectrum[i]={};}
void apply_cuda(const GibbsRinging& t,Sample& s,const Parameters& p,Session& session){
    with_double_channels(t,s,p,session,[&](const Grid& grid,const ChannelPlan& c,auto input,auto output,const auto*,int*){
        if(c.values[0]==1){copy_device(output,input);return;}vd::Grid3 g(grid.shape);Buffer<vd::Complex> spectrum(g.size);to_complex(input,spectrum);fft_volume(spectrum,g,false);
        truncate_frequencies<<<blocks(g.size),256,0,vcr::device::stream()>>>(spectrum.ptr,g,c.values[0]);check(cudaGetLastError());fft_volume(spectrum,g,true);from_complex(spectrum,output);
    });
}
}
