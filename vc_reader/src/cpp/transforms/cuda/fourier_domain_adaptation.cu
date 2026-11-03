#include "../cpu/fourier_domain_adaptation.hpp"
#include "voxel_transform.cuh"
#include "fourier.cuh"
namespace vcr::transforms::cuda {
template<class T,int D> __device__ void adapt_frequencies(KernelIndex vcr_launch, vd::ComplexT<T>* spectrum,const vd::ComplexT<T>* reference,vd::GridND<D> g,double blend,double fraction){EACH_INDEX(g.size)if(vd::low_frequency(g,i,fraction))spectrum[i]=adapt_amplitude(spectrum[i],reference[i],T(blend));}
struct adapt_frequencies_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { adapt_frequencies(index,args...); }
};
void apply_cuda(const FDA& t,Sample& s,const Parameters& p,Session& session){
    with_voxel_channels(t,s,p,session,[&](const Grid& grid,const ChannelPlan& c,auto input,auto output,const auto*,int*){
        if(c.values[0]==0){copy_device(output,input);return;}using T=typename decltype(input)::value_type;vd::with_grid(grid.shape,[&](const auto& g){auto reference=vd::reference_channel<T>(t.reference,c.channel);Buffer<T> ref(std::vector<T>(reference.begin(),reference.end()));Buffer<vd::ComplexT<T>> spectrum(g.size),reference_spectrum(g.size);
        to_complex(input,spectrum);to_complex(View<T>{ref.ptr,ref.size},reference_spectrum);fft_volume(spectrum,g,false);fft_volume(reference_spectrum,g,false);
        launch(blocks(g.size),256,0,vcr::device::stream(), adapt_frequencies_launch{}, spectrum.ptr,reference_spectrum.ptr,g,c.values[0],c.values[1]);check(cudaGetLastError());fft_volume(spectrum,g,true);from_complex(spectrum,output);
        });
    });
}
}
