#include "../fourier_domain_adaptation.hpp"
#include "voxel_transform.cuh"
#include "fourier.cuh"
namespace vcr::augment::cuda {
__global__ void adapt_frequencies(vd::Complex* spectrum,const vd::Complex* reference,vd::Grid3 g,double blend,double fraction){EACH_INDEX(g.size)if(vd::low_frequency(g,i,fraction))spectrum[i]=adapt_amplitude(spectrum[i],reference[i],blend);}
void apply_cuda(const FDA& t,Sample& s,const Parameters& p,Session& session){
    with_double_channels(t,s,p,session,[&](const Grid& grid,const ChannelPlan& c,auto input,auto output,const auto*,int*){
        if(c.values[0]==0){copy_device(output,input);return;}vd::Grid3 g(grid.shape);auto reference=vd::reference_channel(t.reference,c.channel);Buffer<double> ref(reference);Buffer<vd::Complex> spectrum(g.size),reference_spectrum(g.size);
        to_complex(input,spectrum);to_complex(View<double>{ref.ptr,ref.size},reference_spectrum);fft_volume(spectrum,g,false);fft_volume(reference_spectrum,g,false);
        adapt_frequencies<<<blocks(g.size),256,0,vcr::device::stream()>>>(spectrum.ptr,reference_spectrum.ptr,g,c.values[0],c.values[1]);check(cudaGetLastError());fft_volume(spectrum,g,true);from_complex(spectrum,output);
    });
}
}
