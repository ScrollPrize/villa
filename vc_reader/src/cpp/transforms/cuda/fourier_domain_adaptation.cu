#include "../cpu/fourier_domain_adaptation.hpp"
#include "voxel_transform.cuh"
#include "fourier.cuh"
namespace vcr::transforms::cuda {
template<class T,int D> __device__ void adapt_frequencies(KernelIndex vcr_launch, vd::ComplexT<T>* spectrum,const vd::ComplexT<T>* reference,vd::GridND<D> g,double blend,double fraction){EACH_INDEX(g.size)if(vd::low_frequency(g,i,fraction))spectrum[i]=adapt_amplitude(spectrum[i],reference[i],T(blend));}
struct adapt_frequencies_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { adapt_frequencies(index,args...); }
};
void apply_cuda(const FDA& t,Sample& s,const Parameters& p,Session& session){
    FDA settings;
    const FDA* effective=&t;
    if(!t.reference_key.empty()){
        settings=t;settings.reference.wide=vd::reference_target(t.reference_key,s,p).bytes==8;effective=&settings;
    }
    with_voxel_channels(*effective,s,p,session,[&](const Grid& grid,const ChannelPlan& c,auto input,auto output,const auto*,int*){
        if(c.values[0]==0){copy_device(output,input);return;}using T=typename decltype(input)::value_type;vd::with_grid(grid.shape,[&](const auto& g){Buffer<T> ref(g.size);
        if(t.reference_key.empty()){
            auto reference=vd::reference_channel<T>(t.reference,c.channel);ref.upload(reference.data());
        }else{
            auto& source=session.arrays.at(t.reference_key);auto channel=source.shape[0]==1?0:c.channel;
            View<T> destination{ref.ptr,ref.size};
            if(source.bytes==4)convert(destination,source.channel<float>(channel));
            else convert(destination,source.channel<double>(channel));
        }
        Buffer<vd::ComplexT<T>> spectrum(g.size),reference_spectrum(g.size);
        to_complex(input,spectrum);to_complex(View<T>{ref.ptr,ref.size},reference_spectrum);fft_volume(spectrum,g,false);fft_volume(reference_spectrum,g,false);
        launch(blocks(g.size),256,0,vcr::device::stream(), adapt_frequencies_launch{}, spectrum.ptr,reference_spectrum.ptr,g,c.values[0],c.values[1]);check(cudaGetLastError());fft_volume(spectrum,g,true);from_complex(spectrum,output);
        });
    });
}
}
