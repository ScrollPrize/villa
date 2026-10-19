#include "fourier.cuh"
#include <cufft.h>
#include <list>
namespace vcr::augment::cuda {
namespace vd=volume_detail;
namespace {
void check_fft(cufftResult result){require(result==CUFFT_SUCCESS,"cuFFT error "+std::to_string(int(result)));}
// Plan construction is expensive. A bounded per-thread/device/stream cache also owns
// cuFFT's workspace; stream binding is refreshed before every execution.
template<int D> struct Plan {
    int device;std::size_t shape[D];bool wide; int batch; cufftHandle handle{};
    cudaStream_t stream=vcr::device::stream();pid_t pid=getpid();
    Plan(int d,const vd::GridND<D>& g,bool w,int b):device(d),wide(w),batch(b){
        std::copy_n(g.n,D,shape);
        for(auto n:shape)require(n<=std::size_t(INT_MAX),"FFT dimension exceeds cuFFT limit");
        long long sizes[D]; for (int i=0;i<D;++i) sizes[i]=static_cast<long long>(shape[i]);
        check_fft(cufftCreate(&handle));
        std::size_t workspace=0;
        check_fft(cufftMakePlanMany64(handle,D,sizes,nullptr,1,static_cast<long long>(g.size),nullptr,1,static_cast<long long>(g.size),wide?CUFFT_Z2Z:CUFFT_C2C,batch,&workspace));
    }
    ~Plan(){
        if(!handle || pid!=getpid())return;
        int previous; if(cudaGetDevice(&previous)!=cudaSuccess || cudaSetDevice(device)!=cudaSuccess)return;
        cufftDestroy(handle);cudaSetDevice(previous);
    }
    bool matches(int d,const vd::GridND<D>& g,bool w,int b)const{return batch==b && device==d && stream==vcr::device::stream() && wide==w && shape[0]==g.n[0] && shape[1]==g.n[1] && (D==2 || shape[2]==g.n[2]);}
};
template<int D> cufftHandle plan(const vd::GridND<D>& g,bool wide,int batch=1){
    thread_local std::list<Plan<D>> cache;int device;check(cudaGetDevice(&device));
    auto p=std::find_if(cache.begin(),cache.end(),[&](const auto& p){return p.matches(device,g,wide,batch);});
    if(p!=cache.end())cache.splice(cache.begin(),cache,p);
    else{if(cache.size()==4)cache.pop_back();cache.emplace_front(device,g,wide,batch);}
    check_fft(cufftSetStream(cache.front().handle,vcr::device::stream()));return cache.front().handle;
}
template<class T> __device__ void make_complex(KernelIndex vcr_launch, const T* src,vd::ComplexT<T>* dest,std::size_t n){EACH_INDEX(n)dest[i]={src[i],0};}
struct make_complex_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { make_complex(index,args...); }
};
template<class T> __device__ void real_part(KernelIndex vcr_launch, const vd::ComplexT<T>* src,T* dest,std::size_t n){EACH_INDEX(n)dest[i]=src[i].re;}
struct real_part_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { real_part(index,args...); }
};
template<class T> __device__ void normalize(KernelIndex vcr_launch, vd::ComplexT<T>* data,std::size_t n){EACH_INDEX(n)data[i]=data[i].scale(T(1)/T(n));}
struct normalize_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { normalize(index,args...); }
};
}
template<class T,int D> struct FFTJob {
    vd::ComplexT<T>* data;
    vd::GridND<D> grid;
    bool inverse;
    bool compatible(const FFTJob& other) const {
        if (inverse!=other.inverse) return false;
        for (int a=0;a<D;++a) if (grid.n[a]!=other.grid.n[a]) return false;
        return true;
    }
    static void run(const std::vector<const FFTJob*>& jobs) {
        const auto& first=*jobs.front();
        Buffer<vd::ComplexT<T>> packed(jobs.size()>1 ? first.grid.size*jobs.size() : 0);
        auto* values=jobs.size()>1?packed.ptr:first.data;
        if (jobs.size()>1) {
            BatchQueue copies(jobs.size());
            for (std::size_t i=0;i<jobs.size();++i) {
                copies.lane(i);
                check(copy_async(values+i*first.grid.size,jobs[i]->data,first.grid.size*sizeof(*values),cudaMemcpyDeviceToDevice,vcr::device::stream()));
            }
            copies.flush();
        }
        const auto handle=plan(first.grid,std::is_same_v<T,double>,int(jobs.size()));
        const int direction=first.inverse?CUFFT_INVERSE:CUFFT_FORWARD;
        if constexpr(std::is_same_v<T,double>) check_fft(cufftExecZ2Z(handle,reinterpret_cast<cufftDoubleComplex*>(values),reinterpret_cast<cufftDoubleComplex*>(values),direction));
        else check_fft(cufftExecC2C(handle,reinterpret_cast<cufftComplex*>(values),reinterpret_cast<cufftComplex*>(values),direction));
        if (jobs.size()>1) {
            BatchQueue copies(jobs.size());
            for (std::size_t i=0;i<jobs.size();++i) {
                copies.lane(i);
                check(copy_async(jobs[i]->data,values+i*first.grid.size,first.grid.size*sizeof(*values),cudaMemcpyDeviceToDevice,vcr::device::stream()));
            }
            copies.flush();
        }
    }
};
template<class T,int D> void fft_volume(Buffer<vd::ComplexT<T>>& data,const vd::GridND<D>& g,bool inverse){
    FFTJob<T,D> job{data.ptr,g,inverse};
    if (BatchQueue::current) BatchQueue::current->append(job);
    else FFTJob<T,D>::run({&job});
    if(inverse){launch(blocks(g.size),256,0,vcr::device::stream(), normalize_launch{}, data.ptr,g.size);check(cudaGetLastError());}
}
template<class T> void to_complex(View<T> input,Buffer<vd::ComplexT<T>>& output){launch(blocks(input.size),256,0,vcr::device::stream(), make_complex_launch{}, input.ptr,output.ptr,input.size);check(cudaGetLastError());}
template<class T> void from_complex(const Buffer<vd::ComplexT<T>>& input,View<T> output){launch(blocks(output.size),256,0,vcr::device::stream(), real_part_launch{}, input.ptr,output.ptr,output.size);check(cudaGetLastError());}
template void fft_volume(Buffer<vd::ComplexT<float>>&,const vd::GridND<3>&,bool);
template void fft_volume(Buffer<vd::ComplexT<double>>&,const vd::GridND<3>&,bool);
template void fft_volume(Buffer<vd::ComplexT<float>>&,const vd::GridND<2>&,bool);
template void fft_volume(Buffer<vd::ComplexT<double>>&,const vd::GridND<2>&,bool);
template void to_complex(View<float>,Buffer<vd::ComplexT<float>>&);
template void to_complex(View<double>,Buffer<vd::ComplexT<double>>&);
template void from_complex(const Buffer<vd::ComplexT<float>>&,View<float>);
template void from_complex(const Buffer<vd::ComplexT<double>>&,View<double>);
}
