#include "fourier.cuh"
#include <cufft.h>
#include <list>
namespace vcr::augment::cuda {
namespace vd=volume_detail;
namespace {
void check_fft(cufftResult result){require(result==CUFFT_SUCCESS,"cuFFT error "+std::to_string(int(result)));}
// Plan construction is expensive. A bounded per-thread/device/stream cache also owns
// cuFFT's workspace; stream binding is refreshed before every execution.
struct Plan {
    int device;std::size_t shape[3];bool wide;cufftHandle handle{};
    cudaStream_t stream=vcr::device::stream();pid_t pid=getpid();
    Plan(int d,const vd::Grid3& g,bool w):device(d),shape{g.n[0],g.n[1],g.n[2]},wide(w){
        for(auto n:shape)require(n<=std::size_t(INT_MAX),"FFT dimension exceeds cuFFT limit");
        check_fft(cufftPlan3d(&handle,int(shape[0]),int(shape[1]),int(shape[2]),wide?CUFFT_Z2Z:CUFFT_C2C));
    }
    ~Plan(){
        if(!handle || pid!=getpid())return;
        int previous; if(cudaGetDevice(&previous)!=cudaSuccess || cudaSetDevice(device)!=cudaSuccess)return;
        cufftDestroy(handle);cudaSetDevice(previous);
    }
    bool matches(int d,const vd::Grid3& g,bool w)const{return device==d && stream==vcr::device::stream() && wide==w && shape[0]==g.n[0] && shape[1]==g.n[1] && shape[2]==g.n[2];}
};
cufftHandle plan(const vd::Grid3& g,bool wide){
    thread_local std::list<Plan> cache;int device;check(cudaGetDevice(&device));
    auto p=std::find_if(cache.begin(),cache.end(),[&](const auto& p){return p.matches(device,g,wide);});
    if(p!=cache.end())cache.splice(cache.begin(),cache,p);
    else{if(cache.size()==4)cache.pop_back();cache.emplace_front(device,g,wide);}
    check_fft(cufftSetStream(cache.front().handle,vcr::device::stream()));return cache.front().handle;
}
template<class T> __global__ void make_complex(const T* src,vd::ComplexT<T>* dest,std::size_t n){EACH_INDEX(n)dest[i]={src[i],0};}
template<class T> __global__ void real_part(const vd::ComplexT<T>* src,T* dest,std::size_t n){EACH_INDEX(n)dest[i]=src[i].re;}
template<class T> __global__ void normalize(vd::ComplexT<T>* data,std::size_t n){EACH_INDEX(n)data[i]=data[i].scale(T(1)/T(n));}
}
template<class T> void fft_volume(Buffer<vd::ComplexT<T>>& data,const vd::Grid3& g,bool inverse){
    const auto handle=plan(g,std::is_same_v<T,double>);const int direction=inverse?CUFFT_INVERSE:CUFFT_FORWARD;
    if constexpr(std::is_same_v<T,double>)check_fft(cufftExecZ2Z(handle,reinterpret_cast<cufftDoubleComplex*>(data.ptr),reinterpret_cast<cufftDoubleComplex*>(data.ptr),direction));
    else check_fft(cufftExecC2C(handle,reinterpret_cast<cufftComplex*>(data.ptr),reinterpret_cast<cufftComplex*>(data.ptr),direction));
    if(inverse){normalize<<<blocks(g.size),256,0,vcr::device::stream()>>>(data.ptr,g.size);check(cudaGetLastError());}
}
template<class T> void to_complex(View<T> input,Buffer<vd::ComplexT<T>>& output){make_complex<<<blocks(input.size),256,0,vcr::device::stream()>>>(input.ptr,output.ptr,input.size);check(cudaGetLastError());}
template<class T> void from_complex(const Buffer<vd::ComplexT<T>>& input,View<T> output){real_part<<<blocks(output.size),256,0,vcr::device::stream()>>>(input.ptr,output.ptr,output.size);check(cudaGetLastError());}
template void fft_volume(Buffer<vd::ComplexT<float>>&,const vd::Grid3&,bool);
template void fft_volume(Buffer<vd::ComplexT<double>>&,const vd::Grid3&,bool);
template void to_complex(View<float>,Buffer<vd::ComplexT<float>>&);
template void to_complex(View<double>,Buffer<vd::ComplexT<double>>&);
template void from_complex(const Buffer<vd::ComplexT<float>>&,View<float>);
template void from_complex(const Buffer<vd::ComplexT<double>>&,View<double>);
}
